import torch
import torch.nn as nn
from torch.nn import functional as F

class PSTIN(nn.Module):
    def __init__(self,in_channels,hidden_size,num_layers,num_classes,bidirectional=True,**kwargs):
        super(PSTIN, self).__init__()
        
        # spatial encoder: PSAE 
        self.psae = PixelSetAggregationEncoder(in_channels=in_channels)

        # temporal encoder
        self.bilstm = nn.LSTM(input_size = 64,
                            hidden_size = hidden_size,
                            dropout=0.2,
                            num_layers=num_layers,
                            batch_first = True,
                            bidirectional=True) 
        self.attention = nn.Sequential(nn.Linear(in_features = (1*bidirectional+1)*hidden_size,
                                   out_features=1),
                                      nn.ReLU(inplace=True))
        self.fc = nn.Linear(in_features = (1*bidirectional+1)*hidden_size,
                                   out_features=num_classes)
    def forward(self,x):
        
        x,weights = self.psae(x)
     
        
        lstm_out, _ = self.bilstm(x)
        attn_weights = F.softmax(self.attention(lstm_out),dim=1)
        fc_in = attn_weights.permute(0,2,1) @ (lstm_out)
        fc_out = self.fc(fc_in)

        pred = dict({'prediction':fc_out.squeeze(1),
                    #  'weights':weights,
                     'feat':x})
        return pred
       
    
class PixelSetAggregationEncoder(nn.Module):
    def __init__(self,in_channels,**kwargs):
        super(PixelSetAggregationEncoder, self).__init__()
        self.mlp = nn.Sequential( 
                                nn.Linear(in_features = in_channels,
                                        out_features=32),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features =32,
                                        out_features=64),
                                nn.ReLU(inplace=True) ,
                                )
        
        self.query_mlp = nn.Sequential(nn.Linear(in_features =64,
                                            out_features=16),
                                        nn.ReLU(inplace=True))
        self.key_mlp  = nn.Sequential(nn.Linear(in_features =64,
                                            out_features=16),
                                        nn.ReLU(inplace=True))
        self.value_mlp = nn.Sequential(nn.Linear(in_features =64,
                                            out_features=64),
                                        nn.ReLU(inplace=True))
    
    def forward(self,x):
        
        b,t,c,h,w= x.shape 
        x = x.permute(0,3,4,1,2).reshape(b*h*w*t,c)
        x = self.mlp(x)
        
        query = self.query_mlp(x.reshape(b,h,w,t,64)[:,2,2,:,:]).reshape(b,t,1,16) # the spectral features of central pixel
        key = self.key_mlp(x).reshape(b,h*w,t,16).permute(0,2,3,1)
        value = self.value_mlp(x).reshape(b,h*w,t,64).permute(0,2,1,3)
        
        scores = torch.matmul(query,key)
        weights = torch.softmax(scores,-1,dtype=torch.float16)
        
        x = x.reshape(b,h,w,t,64)[:,2,2,:,:] + torch.matmul(weights,value).squeeze(2)
        return x,weights
        
