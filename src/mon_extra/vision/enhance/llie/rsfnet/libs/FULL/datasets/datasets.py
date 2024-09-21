from libs.FULL.datasets.lolv1 import lolv1
from libs.FULL.datasets.lolv2 import lolv2
from libs.FULL.datasets.lolsyn import lolsyn
from libs.FULL.datasets.lolve import lolve
from libs.FULL.datasets.misc import misc

def MyDataset(config, mode):
    if config.dataset.lower()=='lolv1':
        if mode=='train':
            trainset = lolv1(config,'train')
            # valset = lolv1(config,'val')
            if config.f_valFromTrain: valset = lolv1(config,'val')
            else: valset = []
            return trainset,valset
        elif mode=='test':
            testset = lolv1(config,'test')
            return testset

    elif config.dataset.lower()=='lolv2':
        if mode=='train':
            trainset = lolv2(config,'train')
            if config.f_valFromTrain: valset = lolv2(config,'val')
            else: valset = []
            return trainset,valset
        elif mode=='test':
            testset = lolv2(config,'test')
            return testset
    
    elif config.dataset.lower()=='lolsyn':
        if mode=='train':
            trainset = lolsyn(config,'train')
            if config.f_valFromTrain: valset = lolsyn(config,'val')
            else: valset = []
            return trainset,valset
        elif mode=='test':
            testset = lolsyn(config,'test')
            return testset
    
    elif config.dataset.lower()=='lolve':
        if mode=='train':
            trainset = lolve(config,'train')
            if config.f_valFromTrain: valset = lolve(config,'val')
            else: valset = []
            return trainset,valset
        elif mode=='test':
            testset = lolve(config,'test')
            return testset
        
    elif config.dataset.lower()=='misc':
        if mode=='train':
            trainset = misc(config,'train')
            valset = []
            return trainset,valset
        elif mode=='test':
            testset = misc(config,'test')
            return testset
        
    else:
        print('INCORRECT DATASET SPECIFIED. EXITTING.')
        exit()
