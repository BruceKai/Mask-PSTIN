import copy
import logging

from utils import losses
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import OrderedDict
from dataset import mydataset
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import autocast as autocast
import datetime
# % validate the module accuracy
def evaluate(val_data,net,loss,devices):

    val_loss = []
    net.eval()
    with torch.no_grad():
        for element in val_data:
            X = [x.to(devices,dtype=torch.float) for x in element[:-1]]
            target = element[-1]
            target = target.to(devices,dtype=torch.float)

            with autocast():
                pred = net(*X)

                l = loss(pred,target)

            val_loss.append(loss.value)
    val_loss = np.array(val_loss).mean(0)

    return  val_loss

# % define get_logger
def get_logger(filename, mode, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(name)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

# %Train model
def train_model(net,
                devices,
                batchsize,
                lr,
                num_classes,
                max_epoch,
                train_folder,
                val_folder,
                num_workers,
                model_name,
                data_aug,
                resume,
                mask_probability):
    """
        model training
    Args:
        net ([torch model]): the deep neural network
        devices ([torch.device]):
        batchsize ([int]): batch size for training
        lr ([float]): learning rate for training
        num_classes ([int]): the number of classes
        max_epoch ([int]): the maximum number of epochs
        train_folder ([str]): the folder containing training data
        val_folder ([str]): the folder containing validation data
        num_workers ([int]): the number of workers
        data_aug ([bool]): data augmentation
        model_name ([str]): the file name of the networks
        resume([bool]): if resuming the training process
        mask_probability([float]): mask probability
    """

    loss = losses.loss(reduction='mean')

    start_epoch = 1
    trainer = torch.optim.Adam([{'params':net.parameters(),'initial_lr':lr}],lr=lr,weight_decay=1e-6)
    scheduler = MultiStepLR(trainer, milestones=[450], gamma=0.3)
    log_path = './logging/'+model_name+'.log'

    if resume:
        check_point_path = r'./result/model/'+model_name+'_optimal.pth'
        check_point = torch.load(check_point_path)
        net.load_state_dict(check_point['model_state_dict'])
        trainer.load_state_dict(check_point['optimizer_state_dict'])
        scheduler.load_state_dict(check_point['lr_state_dict'])
        start_epoch = check_point['epoch']+1
        logger = get_logger(log_path,"a")
        loss_information = np.load(r'./result/loss/'+model_name+'_optimal.npy', allow_pickle=True)
        loss_information = loss_information[()]
        oa = loss_information['val_loss'][-1][-1]

    else:
        logger = get_logger(log_path,"w")
        logger.info('\r start training!')
        logger.info('\r config params')
        logger.info('\r model name: {}'.format(model_name))
        logger.info('\r batch size: {}'.format(batchsize))
        logger.info('\r initial learning rate: {}'.format(lr))
        logger.info('\r num classes: {}'.format(num_classes))
        logger.info('\r mask probability: {}'.format(mask_probability))
        logger.info('\r date time: {}'.format(datetime.datetime.now()))

        loss_information = dict({'train_loss':[],'val_loss':[],'epoch':[]})
        oa = 0
        
    train_dataset = mydataset.MyDataset(train_folder,data_aug=data_aug,mask_probability=mask_probability)
    val_dataset = mydataset.MyDataset(val_folder,data_aug=False)

    for epoch in range(start_epoch,max_epoch+1):

        net.train()

        train_kwargs = dict({'dataset':train_dataset,
                             'batch_size':batchsize,
                             'shuffle':True,
                             'num_workers':num_workers,
                             'pin_memory':False})
        train_data = DataLoader(**train_kwargs)

        val_kwargs = dict({'dataset':val_dataset,
                             'batch_size':batchsize,
                             'shuffle':False,
                             'num_workers':num_workers,
                             'pin_memory':False,})
        val_data = DataLoader(**val_kwargs)

        j=0
        loss1 = 0.
        tmp_loss = []
        with tqdm(iterable=train_data,desc=f'Epoch {epoch}/{max_epoch}', unit='batch')as pbar:

            for element in train_data:

                trainer.zero_grad()
                X = [x.to(devices,dtype=torch.float) for x in element[:-1]]
                target = element[-1]
                target = target.to(devices,dtype=torch.float)

                with autocast():
                    # print(len(X))
                    pred = net(*X)
                    l = loss(pred,target)

                l.backward()
                trainer.step()
                tmp_loss.append(loss.value)

                loss1 += tmp_loss[j-1][0]

                j += 1
                pbar.set_postfix(
                    loss1 = loss1/j,
                    )
                pbar.update()

        scheduler.step()
        tmp_loss = np.array(tmp_loss).mean(0)

        val_loss = evaluate(val_data,net,loss,devices)
        info1 = ' mode\t loss1\t oa\t'
        info2 = ' train\t %.3f\t %.3f\t'%(
                    tmp_loss[0],tmp_loss[1])
        info3 = ' val\t %.3f\t %.3f\t'%(
                    val_loss[0],val_loss[1])

        logger.info('\r Epoch {}/{}:'.format(epoch,max_epoch))
        logger.info(info1)
        logger.info(info2)
        logger.info(info3)

        loss_information['train_loss'].append(tmp_loss)
        loss_information['val_loss'].append(val_loss)
        loss_information['epoch'].append([epoch])

        if val_loss[-1] >= oa:
            oa = val_loss[-1]
            check_point = dict({'model_state_dict':net.state_dict(),
                               'optimizer_state_dict':trainer.state_dict(),
                               'lr_state_dict': scheduler.state_dict(),
                               'epoch':epoch})
            optimal_model_path = './result/model/'+model_name+'_optimal.pth'
            optimal_loss_path = './result/loss/'+model_name+'_optimal.npy'
            torch.save(check_point,optimal_model_path)
            np.save(optimal_loss_path,np.array(loss_information))

    state_dict = dict({'model_state_dict':net.state_dict(),
                        'epoch':epoch})
    model_path = './result/'+model_name+'.pth'
    loss_path = './result/'+model_name+'.npy'
    torch.save(state_dict,model_path)
    np.save(loss_path,np.array(loss_information))
    logger.info('\r finish training')
    logger.info('\r date time: {}'.format(datetime.datetime.now()))
    logger=[]