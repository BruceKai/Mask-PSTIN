
import numpy as np
import torch
from torch.utils.data import Dataset
       
class MyDataset(Dataset):
    def __init__(self,root_dir,data_aug=False,mask_probability=0):
        super(MyDataset, self).__init__()
        data=np.load(root_dir+'/data.npy')
        lbl=np.load(root_dir+'/lbl.npy')
        
        self.mask_probability = mask_probability

        
        self.data_aug = data_aug
        
        self.data = data
        self.lbl = lbl
                
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        
        feat = self.data[index]
        target = self.lbl[index]
        feat[np.isnan(feat)] = 0 

        # (t*c,h,w)->(t,c,h,w)
        tc,h,w = feat.shape # c=5
        feat = feat.reshape(tc//5,5,h,w)/10000. 
        
        # extract the central pixel for the pixel-based model
        # feat = feat[:,:,2,2] (b,t,c)
        feat = torch.from_numpy(feat)
        
        target = np.int16(target)

        target = torch.from_numpy(target)
        
        if self.data_aug:
            p = np.random.random()
            mask = torch.rand(feat.shape[0])<self.mask_probability
            if p <0.5:
                feat[mask] = 0

        return feat,target