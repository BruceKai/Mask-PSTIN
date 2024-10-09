# <p align="center">Improving crop type mapping by integrating LSTM with temporal random masking and pixel-set spatial information</p>

This is an official implementation of the "Improving crop type mapping by integrating LSTM with temporal random masking and pixel-set spatial information".
The overall structure of Mask-PSTIN.
![image](https://github.com/user-attachments/assets/f9767bfd-7a21-4e0b-8830-7c3613dbb72f)

The temporal random masking technique.
![image](https://github.com/user-attachments/assets/f7f31341-855f-442f-ae7a-b3e875318fa8)

The architecture of pixel-set aggregation encoder (PSAE)
![image](https://github.com/user-attachments/assets/1a3b42b0-8dba-42b9-9ddb-4bd5bf30abc6)

## Requirement 
``````
PyTorch
Numpy
tqdm
``````

## Data Format
``````
inputs
└── <train>
    ├── data.npy
    ├── lbl.npy
└── <val>
    ├── data.npy
    ├── lbl.npy
``````

## Usage
``````
import torch
from utils import train
from utils.BalancedDataParallel import BalancedDataParallel
from model import model


batch_size =16
lr = 1e-3
MAX_EPOCH = 300
NUM_WORKERS = 0
GPU0_BSZ = 8
ACC_GRAD = 1

IN_CHANNELS = 5
NUM_CLASSES = 7
HIDDEN_SIZES = 256
NUM_LAYERS = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mask_prob = 0.00
model = model.PSTIN(IN_CHANNELS,HIDDEN_SIZES,NUM_LAYERS,NUM_CLASSES)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
#   dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = BalancedDataParallel(GPU0_BSZ // ACC_GRAD,model,device_ids=[0,1],output_device=0)


train_folder = r'./data/train'
val_folder = r'./data/val'

model_name = 'PSTIN_mask_%03d'%(mask_prob*100)

# 
model = model.to(device)

train_kwargs = dict({'net':model,
                    'devices':device,
                    'batchsize':batch_size,
                    'lr':lr,
                    'num_classes':NUM_CLASSES,
                    'max_epoch':MAX_EPOCH,
                    'train_folder':train_folder,
                    'val_folder':val_folder,
                    'num_workers':NUM_WORKERS,
                    'data_aug':False,
                    'model_name':model_name,
                    'resume':True,
                    'mask_probability':mask_prob
                    })
train.train_model(**train_kwargs)
``````




