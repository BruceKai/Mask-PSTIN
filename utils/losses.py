
import torch 
import torch.nn as nn 
from utils.focalloss import FocalLoss

class loss(nn.Module):
    def __init__(self,reduction):
        super().__init__()
        size_average = True if reduction == 'mean' else False

        self.focal_loss = FocalLoss(gamma=1,size_average=size_average)
        
        self.value = []
    def forward(self,pred,ref):

        pred = pred['prediction']
        loss =  self.focal_loss(pred,ref.squeeze(1).long())
      
        acc = (torch.argmax(pred,dim=1) == ref.squeeze(1)).sum()/ref.shape[0]

        self.value = [loss.item(),acc.detach().cpu().numpy()]
        return loss
        


