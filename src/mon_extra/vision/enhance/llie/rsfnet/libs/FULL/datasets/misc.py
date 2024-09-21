import os, torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class misc(Dataset):
    def __init__(self, config, mode):
        self.mode=mode
        self.f_RGB = config.f_RGB
        self.f_OverExp = config.f_OverExp
        self.extn = config.extn
        if mode=='train':
            if (config.p_trainList==None):
                self.imList = os.listdir(config.p_trainDir)
            else:
                with open(config.p_trainList) as f: self.imList  = [l.strip() for l in f.readlines()]
            self.p_inDir     = config.p_trainDir
            self.p_resDir    = os.path.join(config.p_resDir,'train')
            self.augment     = transforms.Compose([transforms.ToTensor()])
        elif mode=='test':
            if (config.p_testList==None):
                self.imList = os.listdir(config.p_testDir)
            else:
                with open(config.p_testList) as f: self.imList  = [l.strip() for l in f.readlines()]
            self.p_inDir     = config.p_testDir
            self.p_resDir    = os.path.join(config.p_resDir,'test')
            self.augment_resize     = transforms.Compose([transforms.Resize(size=(config.imsize)),transforms.ToTensor()])
            self.augment     = transforms.Compose([transforms.ToTensor()])
            
    def __len__(self):
        return len(self.imList)
    
    def __getitem__(self,idx):
        imNum,_ = os.path.splitext(self.imList[idx])
        p_low   = os.path.join(self.p_inDir, imNum+self.extn)
        imlow   = Image.open(p_low)
        if not self.f_RGB: imlow  = imlow.convert('YCbCr')
        # if (imlow.size[0]>1000 or imlow.size[1]>1000): imlow   = self.augment_resize(imlow)
        # else: imlow   = self.augment(imlow)
        imlow   = self.augment(imlow)
        return {'imNum':imNum, 'gtdata':torch.zeros_like(imlow), 'imlow':imlow}
