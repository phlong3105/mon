'''
Script for testing the different models in unpaired dataset
'''
import argparse
import os

from archs.retinexformer import RetinexFormer
from options.options import parse

parser = argparse.ArgumentParser(description="Script for testing")
parser.add_argument('-p', '--config', type=str, default='./options/test/RealBlur_Night.yml', help = 'Config file of testing')
args = parser.parse_args()
opt = parse(args.config)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from archs import create_model
from torchvision.transforms import Resize
import torch.multiprocessing as mp
from options.options import parse
from utils.test_utils import *
from tqdm import tqdm
from data import create_test_data
import pyiqa
from ptflops import get_model_complexity_info

import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def pad_tensor(tensor, multiple = 8):
    '''
    Pad the tensor to be multiple of some number (its size).
    '''
    multiple = multiple
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value = 0)
    
    return tensor

def load_model(rank, model, path_weights):
    map_location = 'cpu'
    checkpoints = torch.load(path_weights, map_location='cpu', weights_only=False)
   
    weights = checkpoints['params']
    weights = {'module.' + key: value for key, value in weights.items()}


    macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False, verbose=False)
    print(macs, params)
    model.load_state_dict(weights)
    return model

def create_losses(list_of_losses = ['musiq', 'niqe', 'nrqm', 'brisque'], rank=0):
    losses = {}
    for name in list_of_losses:
        losses[name] = {name: pyiqa.create_metric(name).to(rank)}
    
    return losses

resize = opt['Resize']

def eval_unpaired(rank, world_size):

    setup(rank, world_size=world_size, Master_port='12354')

    test_loader, _ = create_test_data(rank, world_size=world_size, opt = opt['datasets'])

    model, _, _ = create_model(opt['network'], rank)
    model = load_model(rank, model, path_weights = opt['save']['path'])
    print('Using weights in: ', opt['save']['path'])
    
    names = opt['quali']
    losses = create_losses(names, rank)
    if rank==0:
        pbar = tqdm(total = len(test_loader))

    metrics = {name: 0. for name in names}
    model.eval()
    for element, _ in test_loader:

        ind_metric = {name: None for name in names}
        element = element.to(rank)

        _, _, H, W = element.shape
        if resize and (H >=1500 or W>=1500):
            new_size = [int(dim//2) for dim in (H, W)]
            downsample = Resize(new_size)
        else:
            downsample = torch.nn.Identity()    
        element = downsample(element)    

        element = pad_tensor(element)
        
        with torch.no_grad():
            result = model(element, side_loss = False)
            # result = element
        
        if resize:
            upsample = Resize((H, W))
        else: upsample = torch.nn.Identity()
        
        result = upsample(result)
        result = result[:, :, :H, :W]
        
        result = torch.clamp(result, 0., 1.)
        
        for name, loss in losses.items():
            ind_metric[name] = loss[name](result)

        for metric in metrics.keys():
            metrics[metric] = metrics[metric] + ind_metric[metric]

        pbar.update(1)
        
    print('Final results:')
    for name, value in metrics.items():
        print(f'In metric {name} we get: {value / len(test_loader)}')

    if rank == 0:
        pbar.close()    
    cleanup()

def main():
    world_size = 1
    print('Used GPUS:', world_size)
    mp.spawn(eval_unpaired, args =(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
