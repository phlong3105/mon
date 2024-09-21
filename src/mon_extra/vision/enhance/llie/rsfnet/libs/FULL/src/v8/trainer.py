from __future__ import print_function

import json
import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from libs.FULL.datasets.datasets import MyDataset
from libs.FULL.src.v8.model import RRNet
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore", category=FutureWarning)
eps = np.finfo(np.float32).eps


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)


def train(config):
    # READ CONFIG ---------------------------------------
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    p_resDir    = os.path.join(config.p_resDir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    print(f'p_resDir={p_resDir}')
    if not os.path.isdir(p_resDir): os.makedirs(p_resDir)
    n_epochs    = config.epochs
    device      = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config.device = "cuda"
    print(f'device={device}')
    
    # LOGGER CODE HERE -----------------------------------

    # DATALOADERS ----------------------------------------
    dataset_train,_ = MyDataset(config,'train')
    loader_train    = DataLoader(dataset_train, num_workers=config.num_workers, batch_size=config.batch_size)
    print(f'Read {len(loader_train)*config.batch_size} training images.')

    # MODEL -----------------------------------------------
    model = RRNet(config)
    if (torch.cuda.device_count()>1) and len(config.gpuId.split(","))>1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model   = torch.nn.DataParallel(model, device_ids=[int(t) for t in config.gpuId.split(",")])
    model.to(device)
    model.apply(weights_init)
    model_name  = 'RRNet_'+config.dataset
    
    # OPTIMIZERS ----------------------------------------- 
    optimizer = torch.optim.SGD([
                                {'params':model.fuseNet.encoder.parameters(), 'lr':config.lr},
                                {'params':model.fuseNet.decoder.parameters(), 'lr':config.lr},
                                ])
    for i in range(config.factors):
        optimizer.add_param_group({'params':model.factNet.lmbda_A[i].parameters(), 'lr':0.01})  # 0.01
        optimizer.add_param_group({'params':model.factNet.lmbda_E[i].parameters(), 'lr':0.01})  # 0.01
        optimizer.add_param_group({'params':model.factNet.step[i].parameters(), 'lr':0.01})     # 0.01

    # RESUME -----------------------------------------------
    if config.resume:
        if os.path.exists(config.p_model):
            if (torch.cuda.device_count() > 1) and (len(config.gpuId.split(','))>1):
                model.module.load_state_dict(torch.load(config.p_model))
            else:
                model.load_state_dict(torch.load(config.p_model))
            print(f'Model state resumed from {config.p_model}')
            p_resume_json       = os.path.splitext(config.p_model)[0]+'.json'
            with open(p_resume_json,'r') as f:
                resume_json     = json.load(f)
                n_epochs        = config.epochs - resume_json['epoch']+1
                config.__dict__ = resume_json['config']
                optimizer.param_groups[0]['lr'] = resume_json['lr']               
        else:   print(f'ERROR: Model not found at path {config.p_model}. Exitting !')

    # TRAIN -------------------------------------------------
    for epoch in tqdm(range(n_epochs), leave=True, colour='GREEN'):
        dic   = {'train_loss':0, 'L_color':0, 'L_exp':0, 'L_TV':0, 'L_fact':0}
        model.train()
        print(f'*'*75)
        model.factNet.et_mean  = [[] for i in range(config.factors)]
        model.L               = dict.fromkeys(('L_color','L_exp','L_TV','L_fact'))
        if epoch>config.freeze+25:
            optimizer.param_groups[0]['lr']   = optimizer.param_groups[0]['lr']*config.lr_decay
            optimizer.param_groups[1]['lr']   = optimizer.param_groups[1]['lr']*config.lr_decay
                
        for _,data in tqdm(enumerate(loader_train), total=len(loader_train), leave=False, colour='BLUE'):
            imNum, y_labels, imlow  = data['imNum'], data['gtdata'], data['imlow']
            y_labels, imlow         = y_labels.to(device).type(torch.float32), imlow.to(device).type(torch.float32)
            optimizer.zero_grad()

            pred,loss               = model(imlow,epoch,imNum=imNum[0])
            if config.f_OverExp:
                pred = 1-pred
            dic['train_loss']       += (loss.item()/len(loader_train))
            dic['L_color']          += (model.L['L_color']/len(loader_train))
            dic['L_exp']            += (model.L['L_exp']/len(loader_train))
            dic['L_TV']             += (model.L['L_TV']/len(loader_train))
            dic['L_fact']           += (model.L['L_fact']/len(loader_train))
            model.freezeFact(epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   #for LOLv1, LOLv2, LOLsyn
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)    #for LOLve
            optimizer.step()
            del loss, pred
        for i in range(config.factors):
            print(f'''\t E[{i}][0]={model.factNet.lmbda_E[i][0].item():0.9f} \t A[{i}][0]={model.factNet.lmbda_A[i][0].item():0.9f} \t step[{i}][0]={model.factNet.step[i][0].item():0.9f} ''')
        p_model         = os.path.join(p_resDir, model_name+'_'+str(epoch)+'.pt')
        print(f'Saving model at {p_model}')
        if torch.cuda.device_count() and len(config.gpuId.split(',')) > 1:  torch.save(model.module.state_dict(), p_model)
        else:   torch.save(model.state_dict(), p_model)
        with open(os.path.join(p_resDir,model_name+'.json'),'w') as f:
            json.dump({'epoch':epoch, 'lr':optimizer.param_groups[0]['lr'], 'config':config.__dict__}, f, indent=4)

    print(f'COMPLETED TRAINING . SAVED @ {p_resDir}')            
