import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class lolv1(Dataset):
    def __init__(self, config, mode):
        self.mode=mode
        self.f_RGB = config.f_RGB
        self.f_OverExp = config.f_OverExp
        if mode=='train':
            with open(config.p_trainList) as f:
                self.imList  = [l.strip() for l in f.readlines()]
            if (config.f_valFromTrain==True) and config.p_valList==None:  
                with open(config.p_trainList) as f:
                    self.imList    = [l.strip() for l in f.readlines()]
                    trainSize      = int(np.floor(0.95*len(self.imList )))
                    self.imList    = self.imList[:trainSize]
            self.p_inDir     = config.p_trainDir
            self.p_gtDir     = config.p_trainGtDir
            self.p_resDir    = os.path.join(config.p_resDir,'train')
            self.augment     = transforms.Compose([transforms.ToTensor()])
            # self.augment     = transforms.Compose([transforms.Resize(size=(config.imsize,config.imsize)), transforms.ToTensor()])
        elif mode=='val':
            if not config.p_valList==None:
                with open(config.p_valList) as f:
                    self.imList = [l.strip() for l in f.readlines()]
                self.p_inDir     = config.p_valDir
                self.p_gtDir     = config.p_valGtDir
            elif config.f_valFromTrain==True:
                with open(config.p_trainList) as f:
                    self.imList  = [l.strip() for l in f.readlines()]
                    trainSize    = int(np.floor(0.95*len(self.imList )))
                    self.imList  = self.imList[trainSize:]
                self.p_inDir     = config.p_trainDir
                self.p_gtDir     = config.p_trainGtDir
            self.p_resDir    = os.path.join(config.p_resDir,'val')
            self.augment     = transforms.Compose([transforms.ToTensor()])
        elif mode=='test':
            with open(config.p_testList) as f:
                self.imList  = [l.strip() for l in f.readlines()]
            self.p_inDir     = config.p_testDir
            self.p_gtDir     = config.p_testGtDir
            self.p_resDir    = os.path.join(config.p_resDir,'test')
            self.augment     = transforms.Compose([transforms.ToTensor()])
            
    def __len__(self):
        return len(self.imList)
    
    def __getitem__(self,idx):
        imNum,_ = os.path.splitext(self.imList[idx])
        p_low   = os.path.join(self.p_inDir, imNum+'.png')
        imlow   = Image.open(p_low)
        if not self.f_RGB: imlow  = imlow.convert('YCbCr')
        imlow   = self.augment(imlow)
        p_gt    = os.path.join(self.p_gtDir, imNum+'.png')
        gtdata  = Image.open(p_gt)
        if not self.f_RGB:  gtdata  = gtdata.convert('YCbCr')
        gtdata  = self.augment(gtdata)
        return {'imNum':imNum, 'gtdata':gtdata, 'imlow':imlow}

        