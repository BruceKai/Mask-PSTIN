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

## Usage
The ground truth data of Auvergne, France can be downloaded in https://geoservices.ign.fr/rpg
The specific class information in this region is listed as follows:
``````
0: Others
1: Winter wheat
2: Corn
3: Winter rye
4: Winter barley
5: Sunflower
6: Rapeseed
``````

## Data Format of Auvergne, France 
``````
data
└── <train>
    ├── data.npy # time-series satellite image patches with size of (n,t*c,h,w).
    ├── lbl.npy # ground truth 
└── <val>
    ├── data.npy
    ├── lbl.npy
``````





