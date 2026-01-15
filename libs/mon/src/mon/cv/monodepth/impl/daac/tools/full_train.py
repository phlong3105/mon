# pylint: disable=all
# type: ignore

import argparse
import logging
import os
import pprint
import time

import cv2
import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_anything.dpt import DepthAnything_AC
from losses.depth_anything_loss import AffineInvariantLossV2

matplotlib.use("agg")
import yaml

from dataset.semi_depth import SemiDataset

from evaluate_depth import evaluate,evaluate_DA2K
from util.utils import count_params, init_log
from util.dist_helper import setup_distributed
from util.visualize_utils import visualize_geo_prior,save_feature_visualization,save_depth_visualization,save_image
import random
import wandb
from datetime import timedelta
from typing import Tuple


torch.autograd.set_detect_anomaly(True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config',            type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path',         type=str, required=True)
parser.add_argument('--result-path',       type=str, required=True)
parser.add_argument('--local_rank',        type=int, default=0)
parser.add_argument('--port',              type=int, default=None)


def geo_prior_generate(HW_tuple: Tuple[int], depth_map=None):
    """
    Generate geometric prior
    Args:
        HW_tuple: (H,W) represents the height and width of the feature map
        depth_map: depth map tensor with shape [B,H,W]
        weight: weights for position decay and depth decay
    Returns:
        geo_prior: geometric prior tensor with shape [B,HW,HW]
    """
    depth_map = F.interpolate(depth_map.unsqueeze(1), size=HW_tuple,mode='bilinear',align_corners=False).squeeze(1)  

    device = depth_map.device
    B = depth_map.shape[0]
    H, W = HW_tuple

    index_h = torch.arange(H, device=device)
    index_w = torch.arange(W, device=device) 
    grid = torch.meshgrid(index_h, index_w, indexing='ij')
    grid = torch.stack(grid, dim=-1).reshape(H*W, 2)
    mask_pos = grid[:, None, :] - grid[None, :, :]

    mask_pos = torch.sqrt((mask_pos ** 2).sum(dim=-1))  
    mask_pos = mask_pos.unsqueeze(0).expand(B, -1, -1)
    mask_pos = (mask_pos - mask_pos.min()) / (mask_pos.max() - mask_pos.min() + 1e-6) + 1e-6  

    grid_d = depth_map.reshape(B, H*W)
    mask_d = grid_d[:, None, :] - grid_d[:, :, None]
    mask_d = (mask_d.abs())
    
    geo_prior = torch.sqrt(mask_pos ** 2 + mask_d ** 2)

    return geo_prior


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.enabled = True
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    rank, word_size = setup_distributed(port=args.port)
    if rank == 0:
        logger.info('{}'.format(pprint.pformat(cfg)))
    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.result_path, exist_ok=True)
    init_seeds(123, False)

    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'version':'v2'},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'version':'v2'},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'version':'v2'}
    }
    teacher_model = DepthAnything_AC(model_configs['vits'])
    teacher_model.load_state_dict(torch.load(f'./checkpoints/depth_anything_v2_vits.pth'),strict=False)
    teacher_model.cuda().eval()

    for name, param in teacher_model.named_parameters():
        param.requires_grad = False

    model = DepthAnything_AC(model_configs['vits'])
    model.load_state_dict(torch.load(f'./checkpoints/depth_anything_v2_vits.pth'),strict=False)
    model.cuda()
    
    if cfg['encoder_freeze']:
        for param in model.pretrained.parameters():
            param.requires_grad = False
    else:
        for param in model.pretrained.parameters():
            param.requires_grad = True
    for param in model.depth_head.parameters():
        param.requires_grad = True

    use_tqdm = True
    need_save = True

    exp_name = f"your_exp_name"
    if rank == 0:
        logger.info('Model params: {:.1f}M'.format(count_params(model)))
        wandb.init(project="your_project_name", entity="your_entity_name", config=cfg,
                   mode='disabled',
                   name=exp_name,
                   notes=f"your_notes"
                   )

    if cfg['encoder_freeze']:
        optimizer = AdamW(
            model.depth_head.parameters(),  
            lr=cfg['lr'] * cfg['lr_multi'],  
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
    else:
        optimizer = AdamW([{'params': model.pretrained.parameters(), 'lr': cfg['lr']},
                    {'params': model.depth_head.parameters(),
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)
    
    if cfg['criterion']['name'] == 'AffineInvariantLossV2':
        criterion_l = AffineInvariantLossV2().cuda(local_rank)
    elif cfg['criterion']['name'] == 'AffineInvariantLoss':
        criterion_l = AffineInvariantLoss().cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainset_u = SemiDataset(cfg['dataset_u'], cfg['data_root_u'], 'train_u',
                            cfg['crop_size'], args.unlabeled_id_path,argu_mode=cfg['argu_mode'],cfg=cfg)

    valset = SemiDataset(cfg['dataset_val'], cfg['data_root_val'], 'val', cfg['crop_size'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_nyu = SemiDataset(cfg['dataset_val_nyu'], cfg['data_root_nyu'], 'val', cfg['crop_size_nyu'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_sintel = SemiDataset(cfg['dataset_val_sintel'], cfg['data_root_sintel'], 'val', cfg['crop_size_sintel'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_DIODE = SemiDataset(cfg['dataset_val_DIODE'], cfg['data_root_DIODE'], 'val', cfg['crop_size_DIODE'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_ETH3D = SemiDataset(cfg['dataset_val_ETH3D'], cfg['data_root_ETH3D'], 'val', cfg['crop_size_ETH3D'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_robotcar = SemiDataset(cfg['dataset_val_robotcar'], cfg['data_root_robotcar'], 'val', cfg['crop_size_robotcar'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_nuscene = SemiDataset(cfg['dataset_val_nuscene'], cfg['data_root_nuscene'], 'val', cfg['crop_size_nuscene'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_foggy = SemiDataset(cfg['dataset_val_foggy'], cfg['data_root_foggy'], 'val', cfg['crop_size_foggy'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_cloudy = SemiDataset(cfg['dataset_val_cloudy'], cfg['data_root_cloudy'], 'val', cfg['crop_size_cloudy'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_rainy = SemiDataset(cfg['dataset_val_rainy'], cfg['data_root_rainy'], 'val', cfg['crop_size_rainy'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_kitti_c_fog = SemiDataset(cfg['dataset_val_kitti_c_fog'], cfg['data_root_kitti_c_fog'], 'val', cfg['crop_size_kitti_c_fog'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_kitti_c_snow = SemiDataset(cfg['dataset_val_kitti_c_snow'], cfg['data_root_kitti_c_snow'], 'val', cfg['crop_size_kitti_c_snow'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_kitti_c_dark = SemiDataset(cfg['dataset_val_kitti_c_dark'], cfg['data_root_kitti_c_dark'], 'val', cfg['crop_size_kitti_c_dark'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_kitti_c_motion = SemiDataset(cfg['dataset_val_kitti_c_motion'], cfg['data_root_kitti_c_motion'], 'val', cfg['crop_size_kitti_c_motion'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_kitti_c_gaussian = SemiDataset(cfg['dataset_val_kitti_c_gaussian'], cfg['data_root_kitti_c_gaussian'], 'val', cfg['crop_size_kitti_c_gaussian'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_DA2K = SemiDataset(cfg['dataset_val_DA2K'], cfg['data_root_DA2K'], 'val', cfg['crop_size_DA2K'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_DA2K_dark = SemiDataset(cfg['dataset_val_DA2K_dark'], cfg['data_root_DA2K_dark'], 'val', cfg['crop_size_DA2K_dark'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_DA2K_snow = SemiDataset(cfg['dataset_val_DA2K_snow'], cfg['data_root_DA2K_snow'], 'val', cfg['crop_size_DA2K_snow'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_DA2K_fog = SemiDataset(cfg['dataset_val_DA2K_fog'], cfg['data_root_DA2K_fog'], 'val', cfg['crop_size_DA2K_fog'],argu_mode=cfg['argu_mode'],cfg=cfg)
    valset_DA2K_blur = SemiDataset(cfg['dataset_val_DA2K_blur'], cfg['data_root_DA2K_blur'], 'val', cfg['crop_size_DA2K_blur'],argu_mode=cfg['argu_mode'],cfg=cfg)

   
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=False, num_workers=4, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4,
                           drop_last=False, sampler=valsampler)
    
    valsampler_nyu = torch.utils.data.distributed.DistributedSampler(valset_nyu)
    valloader_nyu = DataLoader(valset_nyu, batch_size=1, pin_memory=True, num_workers=4,
                               drop_last=False, sampler=valsampler_nyu)
    
    valsampler_sintel = torch.utils.data.distributed.DistributedSampler(valset_sintel)
    valloader_sintel = DataLoader(valset_sintel, batch_size=1, pin_memory=True, num_workers=4,
                                   drop_last=False, sampler=valsampler_sintel)
    
    valsampler_DIODE = torch.utils.data.distributed.DistributedSampler(valset_DIODE)
    valloader_DIODE = DataLoader(valset_DIODE, batch_size=1, pin_memory=True, num_workers=4,
                                   drop_last=False, sampler=valsampler_DIODE)
    
    valsampler_ETH3D = torch.utils.data.distributed.DistributedSampler(valset_ETH3D)
    valloader_ETH3D = DataLoader(valset_ETH3D, batch_size=1, pin_memory=True, num_workers=4,
                                   drop_last=False, sampler=valsampler_ETH3D)
    
    valsampler_robotcar = torch.utils.data.distributed.DistributedSampler(valset_robotcar)
    valloader_robotcar = DataLoader(valset_robotcar, batch_size=1, pin_memory=True, num_workers=4,
                                   drop_last=False, sampler=valsampler_robotcar)
    valsampler_nuscene = torch.utils.data.distributed.DistributedSampler(valset_nuscene)
    valloader_nuscene = DataLoader(valset_nuscene, batch_size=1, pin_memory=True, num_workers=4,
                               drop_last=False, sampler=valsampler_nuscene)
    
    valsampler_foggy = torch.utils.data.distributed.DistributedSampler(valset_foggy)
    valloader_foggy = DataLoader(valset_foggy, batch_size=1, pin_memory=True, num_workers=4,
                               drop_last=False, sampler=valsampler_foggy)
    
    valsampler_cloudy = torch.utils.data.distributed.DistributedSampler(valset_cloudy)
    valloader_cloudy = DataLoader(valset_cloudy, batch_size=1, pin_memory=True, num_workers=4,
                               drop_last=False, sampler=valsampler_cloudy)
    
    valsampler_rainy = torch.utils.data.distributed.DistributedSampler(valset_rainy)
    valloader_rainy = DataLoader(valset_rainy, batch_size=1, pin_memory=True, num_workers=4,
                               drop_last=False, sampler=valsampler_rainy)
    
    valsampler_kitti_c_fog = torch.utils.data.distributed.DistributedSampler(valset_kitti_c_fog)
    valloader_kitti_c_fog = DataLoader(valset_kitti_c_fog, batch_size=1, pin_memory=True, num_workers=4,
                               drop_last=False, sampler=valsampler_kitti_c_fog)     

    valsampler_kitti_c_snow = torch.utils.data.distributed.DistributedSampler(valset_kitti_c_snow)
    valloader_kitti_c_snow = DataLoader(valset_kitti_c_snow, batch_size=1, pin_memory=True, num_workers=4,
                               drop_last=False, sampler=valsampler_kitti_c_snow)

    valsampler_kitti_c_dark = torch.utils.data.distributed.DistributedSampler(valset_kitti_c_dark)
    valloader_kitti_c_dark = DataLoader(valset_kitti_c_dark, batch_size=1, pin_memory=True, num_workers=4,
                               drop_last=False, sampler=valsampler_kitti_c_dark)

    valsampler_kitti_c_motion = torch.utils.data.distributed.DistributedSampler(valset_kitti_c_motion)
    valloader_kitti_c_motion = DataLoader(valset_kitti_c_motion, batch_size=1, pin_memory=True, num_workers=4,
                               drop_last=False, sampler=valsampler_kitti_c_motion)
    
    valsampler_kitti_c_gaussian = torch.utils.data.distributed.DistributedSampler(valset_kitti_c_gaussian)
    valloader_kitti_c_gaussian = DataLoader(valset_kitti_c_gaussian, batch_size=1, pin_memory=True, num_workers=4,
                               drop_last=False, sampler=valsampler_kitti_c_gaussian)    

    valsampler_DA2K = torch.utils.data.distributed.DistributedSampler(valset_DA2K)
    valloader_DA2K = DataLoader(valset_DA2K, batch_size=1, pin_memory=True, num_workers=4,
                               drop_last=False, sampler=valsampler_DA2K)
    
    valsampler_DA2K_dark = torch.utils.data.distributed.DistributedSampler(valset_DA2K_dark)
    valloader_DA2K_dark = DataLoader(valset_DA2K_dark, batch_size=1, pin_memory=True, num_workers=4,
                               drop_last=False, sampler=valsampler_DA2K_dark)
    
    valsampler_DA2K_snow = torch.utils.data.distributed.DistributedSampler(valset_DA2K_snow)
    valloader_DA2K_snow = DataLoader(valset_DA2K_snow, batch_size=1, pin_memory=True, num_workers=4,
                               drop_last=False, sampler=valsampler_DA2K_snow)
    
    valsampler_DA2K_fog = torch.utils.data.distributed.DistributedSampler(valset_DA2K_fog)
    valloader_DA2K_fog = DataLoader(valset_DA2K_fog, batch_size=1, pin_memory=True, num_workers=4,
                               drop_last=False, sampler=valsampler_DA2K_fog)
    
    valsampler_DA2K_blur = torch.utils.data.distributed.DistributedSampler(valset_DA2K_blur)
    valloader_DA2K_blur = DataLoader(valset_DA2K_blur, batch_size=1, pin_memory=True, num_workers=4,
                               drop_last=False, sampler=valsampler_DA2K_blur)

    total_iters = len(trainloader_u) * (cfg['epochs'])
    previous_best = 0.0

    if cfg['val_init']:
        res_val_init = evaluate(model, valloader, cfg, cfg['dataset_val'], cfg['depth_min_val'], cfg['depth_cap_val'], use_tqdm)
        res_val_nyu_init = evaluate(model, valloader_nyu, cfg, cfg['dataset_val_nyu'], cfg['depth_min_val_nyu'], cfg['depth_cap_val_nyu'], use_tqdm)
        res_val_sintel_init = evaluate(model, valloader_sintel, cfg, cfg['dataset_val_sintel'], cfg['depth_min_val_sintel'], cfg['depth_cap_val_sintel'], use_tqdm)
        res_val_DIODE_init = evaluate(model, valloader_DIODE, cfg, cfg['dataset_val_DIODE'], cfg['depth_min_val_DIODE'], cfg['depth_cap_val_DIODE'], use_tqdm)
        res_val_ETH3D_init = evaluate(model, valloader_ETH3D, cfg, cfg['dataset_val_ETH3D'], cfg['depth_min_val_ETH3D'], cfg['depth_cap_val_ETH3D'], use_tqdm)
        res_val_robotcar_init = evaluate(model, valloader_robotcar, cfg, cfg['dataset_val_robotcar'], cfg['depth_min_val_robotcar'], cfg['depth_cap_val_robotcar'], use_tqdm)
        res_val_nuscene_init = evaluate(model, valloader_nuscene, cfg, cfg['dataset_val_nuscene'], cfg['depth_min_val_nuscene'], cfg['depth_cap_val_nuscene'], use_tqdm)
        res_val_foggy_init = evaluate(model, valloader_foggy, cfg, cfg['dataset_val_foggy'], cfg['depth_min_val_foggy'], cfg['depth_cap_val_foggy'], use_tqdm)
        res_val_cloudy_init = evaluate(model, valloader_cloudy, cfg, cfg['dataset_val_cloudy'], cfg['depth_min_val_cloudy'], cfg['depth_cap_val_cloudy'], use_tqdm)
        res_val_rainy_init = evaluate(model, valloader_rainy, cfg, cfg['dataset_val_rainy'], cfg['depth_min_val_rainy'], cfg['depth_cap_val_rainy'], use_tqdm)

        res_val_kitti_c_fog_init = evaluate(model, valloader_kitti_c_fog, cfg, cfg['dataset_val_kitti_c_fog'], cfg['depth_min_val_kitti_c_fog'], cfg['depth_cap_val_kitti_c_fog'], use_tqdm)
        res_val_kitti_c_snow_init = evaluate(model, valloader_kitti_c_snow, cfg, cfg['dataset_val_kitti_c_snow'], cfg['depth_min_val_kitti_c_snow'], cfg['depth_cap_val_kitti_c_snow'], use_tqdm)
        res_val_kitti_c_dark_init = evaluate(model, valloader_kitti_c_dark, cfg, cfg['dataset_val_kitti_c_dark'], cfg['depth_min_val_kitti_c_dark'], cfg['depth_cap_val_kitti_c_dark'], use_tqdm)
        res_val_kitti_c_motion_init = evaluate(model, valloader_kitti_c_motion, cfg, cfg['dataset_val_kitti_c_motion'], cfg['depth_min_val_kitti_c_motion'], cfg['depth_cap_val_kitti_c_motion'], use_tqdm)
        res_val_kitti_c_gaussian_init = evaluate(model, valloader_kitti_c_gaussian, cfg, cfg['dataset_val_kitti_c_gaussian'], cfg['depth_min_val_kitti_c_gaussian'], cfg['depth_cap_val_kitti_c_gaussian'], use_tqdm)

        res_val_DA2K_init = evaluate_DA2K(model, valloader_DA2K, cfg['depth_cap_val_DA2K'], use_tqdm)
        res_val_DA2K_dark_init = evaluate_DA2K(model, valloader_DA2K_dark, cfg['depth_cap_val_DA2K_dark'], use_tqdm)
        res_val_DA2K_snow_init = evaluate_DA2K(model, valloader_DA2K_snow, cfg['depth_cap_val_DA2K_snow'], use_tqdm)
        res_val_DA2K_fog_init = evaluate_DA2K(model, valloader_DA2K_fog, cfg['depth_cap_val_DA2K_fog'], use_tqdm)
        res_val_DA2K_blur_init = evaluate_DA2K(model, valloader_DA2K_blur, cfg['depth_cap_val_DA2K_blur'], use_tqdm)


        a1_init = res_val_init['a1']
        abs_rel_init = res_val_init['abs_rel']

        a1_nyu_init = res_val_nyu_init['a1']
        abs_rel_nyu_init = res_val_nyu_init['abs_rel']

        a1_sintel_init = res_val_sintel_init['a1']
        abs_rel_sintel_init = res_val_sintel_init['abs_rel']

        a1_DIODE_init = res_val_DIODE_init['a1']
        abs_rel_DIODE_init = res_val_DIODE_init['abs_rel']

        a1_ETH3D_init = res_val_ETH3D_init['a1']
        abs_rel_ETH3D_init = res_val_ETH3D_init['abs_rel']

        a1_robotcar_init = res_val_robotcar_init['a1']
        abs_rel_robotcar_init = res_val_robotcar_init['abs_rel']
        a1_nuscene_init = res_val_nuscene_init['a1']
        abs_rel_nuscene_init = res_val_nuscene_init['abs_rel']

        a1_foggy_init = res_val_foggy_init['a1']
        abs_rel_foggy_init = res_val_foggy_init['abs_rel']
        a1_cloudy_init = res_val_cloudy_init['a1']
        abs_rel_cloudy_init = res_val_cloudy_init['abs_rel']
        a1_rainy_init = res_val_rainy_init['a1']
        abs_rel_rainy_init = res_val_rainy_init['abs_rel']

        a1_kitti_c_fog_init = res_val_kitti_c_fog_init['a1']    
        abs_rel_kitti_c_fog_init = res_val_kitti_c_fog_init['abs_rel']
        a1_kitti_c_snow_init = res_val_kitti_c_snow_init['a1']    
        abs_rel_kitti_c_snow_init = res_val_kitti_c_snow_init['abs_rel']
        a1_kitti_c_dark_init = res_val_kitti_c_dark_init['a1']    
        abs_rel_kitti_c_dark_init = res_val_kitti_c_dark_init['abs_rel']
        a1_kitti_c_motion_init = res_val_kitti_c_motion_init['a1']    
        abs_rel_kitti_c_motion_init = res_val_kitti_c_motion_init['abs_rel']
        a1_kitti_c_gaussian_init = res_val_kitti_c_gaussian_init['a1']    
        abs_rel_kitti_c_gaussian_init = res_val_kitti_c_gaussian_init['abs_rel']

        acc_DA2K_init = res_val_DA2K_init['Accuracy']    
        acc_DA2K_dark_init = res_val_DA2K_dark_init['Accuracy']    
        acc_DA2K_snow_init = res_val_DA2K_snow_init['Accuracy']    
        acc_DA2K_fog_init = res_val_DA2K_fog_init['Accuracy']    
        acc_DA2K_blur_init = res_val_DA2K_blur_init['Accuracy']    
        

        if rank == 0:
                wandb.log({"val/a1_init": a1_init, "semi_charts/epoch": 0})
                wandb.log({"val/abs_rel_init": abs_rel_init, "semi_charts/epoch": 0})
                wandb.log({"val/a1_nyu_init": a1_nyu_init, "semi_charts/epoch": 0})
                wandb.log({"val/abs_rel_nyu_init": abs_rel_nyu_init, "semi_charts/epoch": 0})
                wandb.log({"val/a1_sintel_init": a1_sintel_init, "semi_charts/epoch": 0})
                wandb.log({"val/abs_rel_sintel_init": abs_rel_sintel_init, "semi_charts/epoch": 0})
                wandb.log({"val/a1_DIODE_init": a1_DIODE_init, "semi_charts/epoch": 0})
                wandb.log({"val/abs_rel_DIODE_init": abs_rel_DIODE_init, "semi_charts/epoch": 0})
                wandb.log({"val/a1_ETH3D_init": a1_ETH3D_init, "semi_charts/epoch": 0})
                wandb.log({"val/abs_rel_ETH3D_init": abs_rel_ETH3D_init, "semi_charts/epoch": 0})


                wandb.log({"val/a1_robotcar_init": a1_robotcar_init, "semi_charts/epoch": 0})
                wandb.log({"val/abs_rel_robotcar_init": abs_rel_robotcar_init, "semi_charts/epoch": 0})
                wandb.log({"val/a1_nuscene_init": a1_nuscene_init, "semi_charts/epoch": 0})
                wandb.log({"val/abs_rel_nuscene_init": abs_rel_nuscene_init, "semi_charts/epoch": 0})
                wandb.log({"val/a1_foggy_init": a1_foggy_init, "semi_charts/epoch": 0})
                wandb.log({"val/abs_rel_foggy_init": abs_rel_foggy_init, "semi_charts/epoch": 0})
                wandb.log({"val/a1_cloudy_init": a1_cloudy_init, "semi_charts/epoch": 0})
                wandb.log({"val/abs_rel_cloudy_init": abs_rel_cloudy_init, "semi_charts/epoch": 0})
                wandb.log({"val/a1_rainy_init": a1_rainy_init, "semi_charts/epoch": 0})
                wandb.log({"val/abs_rel_rainy_init": abs_rel_rainy_init, "semi_charts/epoch": 0})

                wandb.log({"val/a1_kitti_c_fog_init": a1_kitti_c_fog_init, "semi_charts/epoch": 0})
                wandb.log({"val/abs_rel_kitti_c_fog_init": abs_rel_kitti_c_fog_init, "semi_charts/epoch": 0})
                wandb.log({"val/a1_kitti_c_snow_init": a1_kitti_c_snow_init, "semi_charts/epoch": 0})
                wandb.log({"val/abs_rel_kitti_c_snow_init": abs_rel_kitti_c_snow_init, "semi_charts/epoch": 0})
                wandb.log({"val/a1_kitti_c_dark_init": a1_kitti_c_dark_init, "semi_charts/epoch": 0})
                wandb.log({"val/abs_rel_kitti_c_dark_init": abs_rel_kitti_c_dark_init, "semi_charts/epoch": 0})
                wandb.log({"val/a1_kitti_c_motion_init": a1_kitti_c_motion_init, "semi_charts/epoch": 0})
                wandb.log({"val/abs_rel_kitti_c_motion_init": abs_rel_kitti_c_motion_init, "semi_charts/epoch": 0})
                wandb.log({"val/a1_kitti_c_gaussian_init": a1_kitti_c_gaussian_init, "semi_charts/epoch": 0})
                wandb.log({"val/abs_rel_kitti_c_gaussian_init": abs_rel_kitti_c_gaussian_init, "semi_charts/epoch": 0})

                wandb.log({"val/acc_DA2K_init": acc_DA2K_init, "semi_charts/epoch": 0})
                wandb.log({"val/acc_DA2K_dark_init": acc_DA2K_dark_init, "semi_charts/epoch": 0})
                wandb.log({"val/acc_DA2K_snow_init": acc_DA2K_snow_init, "semi_charts/epoch": 0})
                wandb.log({"val/acc_DA2K_fog_init": acc_DA2K_fog_init, "semi_charts/epoch": 0})
                wandb.log({"val/acc_DA2K_blur_init": acc_DA2K_blur_init, "semi_charts/epoch": 0})

                eval_mode = 'original'

                logger.info('***** Evaluation_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_init))
                logger.info('***** Evaluation_nyu_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_nyu_init))
                logger.info('***** Evaluation_sintel_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_sintel_init))
                logger.info('***** Evaluation_DIODE_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_DIODE_init))
                logger.info('***** Evaluation_ETH3D_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_ETH3D_init))
                logger.info('***** Evaluation_robotcar_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_robotcar_init))
                logger.info('***** Evaluation_nuscene_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_nuscene_init))
                logger.info('***** Evaluation_foggy_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_foggy_init))
                logger.info('***** Evaluation_cloudy_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_cloudy_init))
                logger.info('***** Evaluation_rainy_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_rainy_init))
                logger.info('***** Evaluation_kitti_c_fog_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_kitti_c_fog_init))
                logger.info('***** Evaluation_kitti_c_snow_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_kitti_c_snow_init))
                logger.info('***** Evaluation_kitti_c_dark_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_kitti_c_dark_init))
                logger.info('***** Evaluation_kitti_c_motion_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_kitti_c_motion_init))
                logger.info('***** Evaluation_kitti_c_gaussian_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_kitti_c_gaussian_init))
                logger.info('***** Evaluation_DA2K_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_DA2K_init))
                logger.info('***** Evaluation_DA2K_dark_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_DA2K_dark_init))
                logger.info('***** Evaluation_DA2K_snow_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_DA2K_snow_init))
                logger.info('***** Evaluation_DA2K_fog_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_DA2K_fog_init))
                logger.info('***** Evaluation_DA2K_blur_init {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_DA2K_blur_init))

    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.6f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss, total_loss_unlabeled, total_loss_kd_s,total_loss_consistency,total_loss_similarity,total_loss_prior= 0.0, 0.0, 0.0, 0.0, 0.0,0.0
        trainloader_u.sampler.set_epoch(epoch)

        loader =  trainloader_u
        model.train()

        if rank == 0:
            if use_tqdm:
                tbar = tqdm(total=len(trainloader_u))
        for i,  sample_u in enumerate(loader):
            mask_u = sample_u['mask'].cuda()
            img_u_w = sample_u['img_w'].cuda()
            img_u_s1 = sample_u['img_s1'].cuda()
            
            with torch.no_grad():
                teacher_model.eval()
                
                res_t_u_w = teacher_model(img_u_w)
                res_t_u_w = res_t_u_w['out'].detach()

            
            res_s_u_w_all = model(img_u_w,prior_mode=cfg['prior_mode'])
            res_s_u_w = res_s_u_w_all['out']
            
            res_s_u_s_all = model(img_u_s1,prior_mode=cfg['prior_mode'])
            res_s_u_s = res_s_u_s_all['out']

            if cfg['use_prior']:
                
                if cfg['prior_mode'] == 'teacher':
                    _,H,W=res_t_u_w.shape
                    geo_prior_teacher=geo_prior_generate(HW_tuple=(H//14,W//14),depth_map=res_t_u_w)
                    _,H,W=res_s_u_s.shape
                    geo_prior_student_s=geo_prior_generate(HW_tuple=(H//14,W//14),depth_map=res_s_u_s)
                   
                    if rank == 0 and cfg['vis'] and i % cfg['save_interval'] == 0:
                        os.makedirs(f"vis/epoch_{epoch}/iter_{i}/geo_prior", exist_ok=True)
                        
                        visualize_geo_prior(
                            img_u_w, 
                            geo_prior_teacher, 
                            f'vis/epoch_{epoch}/iter_{i}/geo_prior/teacher_center.png'
                        )
                        
                        visualize_geo_prior(
                            img_u_s1, 
                            geo_prior_student_s, 
                            f'vis/epoch_{epoch}/iter_{i}/geo_prior/student_center.png'
                        )

                        H_vis, W_vis = H//14, W//14
                        corner_points = [(0, 0), (0, W_vis-1), (H_vis-1, 0), (H_vis-1, W_vis-1)]
                        corner_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
                        
                        for point, name in zip(corner_points, corner_names):
                            visualize_geo_prior(
                                img_u_w, 
                                geo_prior_teacher, 
                                f'vis/epoch_{epoch}/iter_{i}/geo_prior/teacher_{name}.png',
                                point_coords=point
                            )
                            
                            visualize_geo_prior(
                                img_u_s1, 
                                geo_prior_student_s, 
                                f'vis/epoch_{epoch}/iter_{i}/geo_prior/student_{name}.png',
                                point_coords=point
                            )

            
            with torch.no_grad():
                if rank == 0 and cfg['vis']:
                    if i%cfg['save_interval']==0:
                        os.makedirs(f"vis/epoch_{epoch}/iter_{i}", exist_ok=True)
                        save_image(img_u_w[0], f'vis/epoch_{epoch}/iter_{i}/input_unlabeled_weak.png')
                        save_image(img_u_s1[0], f'vis/epoch_{epoch}/iter_{i}/input_unlabeled_strong.png')
                        mask_vis_u=mask_u[0]*255
                        mask_vis_u=mask_vis_u.detach().cpu().numpy().astype(np.uint8)
                        cv2.imwrite(f'vis/epoch_{epoch}/iter_{i}/mask_unlabeled.png', mask_vis_u)
                    
                        save_depth_visualization(res_s_u_w[0], f'vis/epoch_{epoch}/iter_{i}/depth_unlabeled_weak.png')
                        save_depth_visualization(res_s_u_s[0], f'vis/epoch_{epoch}/iter_{i}/depth_unlabeled_strong.png')
                        save_depth_visualization(res_t_u_w[0], f'vis/epoch_{epoch}/iter_{i}/depth_teacher_unlabeled.png')
                        
                        features_student_strong_vis = model.module.get_intermediate_features(img_u_s1[0:1])
                        features_student_weak_vis = model.module.get_intermediate_features(img_u_w[0:1])
                        features_teacher_vis = teacher_model.get_intermediate_features(img_u_w[0:1])


                        os.makedirs(f"vis/epoch_{epoch}/iter_{i}/features", exist_ok=True)
                        os.makedirs(f"vis/epoch_{epoch}/iter_{i}/features/encoder", exist_ok=True)
                        os.makedirs(f"vis/epoch_{epoch}/iter_{i}/features/decoder", exist_ok=True)
                        os.makedirs(f"vis/epoch_{epoch}/iter_{i}/features/decoder_path", exist_ok=True)
                        os.makedirs(f"vis/epoch_{epoch}/iter_{i}/features/cls_token", exist_ok=True)
                        
                        if 'encoder_features' in features_student_strong_vis:
                            for idx, feat in enumerate(features_student_strong_vis['encoder_features']):
                                save_feature_visualization(feat, f'vis/epoch_{epoch}/iter_{i}/features/encoder/encoder_student_strong_{idx}.png')
                        if 'encoder_features' in features_student_weak_vis:
                            for idx, feat in enumerate(features_student_weak_vis['encoder_features']):
                                save_feature_visualization(feat, f'vis/epoch_{epoch}/iter_{i}/features/encoder/encoder_student_weak_{idx}.png')

                        if 'encoder_features' in features_teacher_vis:
                            for idx, feat in enumerate(features_teacher_vis['encoder_features']):
                                save_feature_visualization(feat, f'vis/epoch_{epoch}/iter_{i}/features/encoder/encoder_teacher_{idx}.png')
                        
                        if 'decoder_features' in features_student_strong_vis:
                            for idx, feat in enumerate(features_student_strong_vis['decoder_features']):
                                save_feature_visualization(feat, f'vis/epoch_{epoch}/iter_{i}/features/decoder/decoder_student_strong_{idx}.png')

                            for idx, feat in enumerate(features_student_strong_vis['decoder_features_path']):
                                save_feature_visualization(feat, f'vis/epoch_{epoch}/iter_{i}/features/decoder_path/decoder_student_path_strong_{idx}.png')

                        if 'decoder_features' in features_student_weak_vis:
                            for idx, feat in enumerate(features_student_weak_vis['decoder_features']):
                                save_feature_visualization(feat, f'vis/epoch_{epoch}/iter_{i}/features/decoder/decoder_student_weak_{idx}.png')

                            for idx, feat in enumerate(features_student_weak_vis['decoder_features_path']):
                                save_feature_visualization(feat, f'vis/epoch_{epoch}/iter_{i}/features/decoder_path/decoder_student_path_weak_{idx}.png')

                        if 'decoder_features' in features_teacher_vis:
                            for idx, feat in enumerate(features_teacher_vis['decoder_features']):
                                save_feature_visualization(feat, f'vis/epoch_{epoch}/iter_{i}/features/decoder/decoder_teacher_{idx}.png')

                            for idx, feat in enumerate(features_teacher_vis['decoder_features_path']):
                                save_feature_visualization(feat, f'vis/epoch_{epoch}/iter_{i}/features/decoder_path/decoder_teacher_path_{idx}.png')
       
            if cfg['loss_mode'] == "prior":
                loss_unlabeled   = criterion_l(res_s_u_w, res_t_u_w, mask_u, uncertainty=None, need_inverse=False)
                loss_consistency = criterion_l(res_s_u_s, res_s_u_w, mask_u, uncertainty=None, need_inverse=False)
                if cfg['prior_mode'] == 'teacher':
                    loss_prior = F.mse_loss(geo_prior_student_s, geo_prior_teacher)

                loss = (loss_unlabeled + loss_consistency + loss_prior) / 3.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_unlabeled += loss_unlabeled.item()
            total_loss_consistency+=loss_consistency.item()
            total_loss_prior+=loss_prior.item()

            if epoch >= 0:
                iters = epoch  * len(trainloader_u) + i
                lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
            else:
                iters = epoch * len(trainloader_u) + i
                start_factor = 1e-6
                if iters == 1:
                    lr = cfg['lr'] * start_factor
                else:
                    lr = cfg['lr'] * start_factor + ((cfg['lr'] - cfg['lr'] * start_factor) / (len(trainloader_u))) * (
                                iters - 1)
                optimizer.param_groups[0]["lr"] = lr

            iters = epoch * len(trainloader_u) + i

            if rank == 0 :
                if i % 100 == 0:
                    wandb.log({"train/total_loss": (total_loss / (i + 1)), "semi_charts/iter": iters})
                    wandb.log({"train/unlabel unlabeled loss": (total_loss_unlabeled / (i + 1)), "semi_charts/iter": iters})
                    wandb.log({"train/unlabel consistency loss": (total_loss_consistency / (i + 1)), "semi_charts/iter": iters})
                    wandb.log({"train/unlabel prior loss": (total_loss_prior / (i + 1)), "semi_charts/iter": iters})
                if use_tqdm:
                    tbar.set_description(' Total loss: {:.3f}, Loss unlabel: {:.3f},Loss kd_s: {:.3f},Loss consistency: {:.3f},Loss similarity: {:.3f},Loss prior: {:.3f}'.format(
                        total_loss / (i + 1), total_loss_unlabeled / (i + 1),total_loss_kd_s / (i + 1),total_loss_consistency / (i + 1),total_loss_similarity / (i + 1),total_loss_prior / (i + 1)))
                    tbar.update(1)
        if rank == 0:
            if use_tqdm:
                tbar.close()

        eval_mode = 'original'
        torch.cuda.empty_cache()

        res_val_s = evaluate(model, valloader, cfg, cfg['dataset_val'], cfg['depth_min_val'], cfg['depth_cap_val'], use_tqdm)
        res_val_nyu_s = evaluate(model, valloader_nyu, cfg, cfg['dataset_val_nyu'], cfg['depth_min_val_nyu'], cfg['depth_cap_val_nyu'], use_tqdm)
        res_val_sintel_s = evaluate(model, valloader_sintel, cfg, cfg['dataset_val_sintel'], cfg['depth_min_val_sintel'], cfg['depth_cap_val_sintel'], use_tqdm)
        res_val_DIODE_s = evaluate(model, valloader_DIODE, cfg, cfg['dataset_val_DIODE'], cfg['depth_min_val_DIODE'], cfg['depth_cap_val_DIODE'], use_tqdm)
        res_val_ETH3D_s = evaluate(model, valloader_ETH3D, cfg, cfg['dataset_val_ETH3D'], cfg['depth_min_val_ETH3D'], cfg['depth_cap_val_ETH3D'], use_tqdm)
        res_val_robotcar_s = evaluate(model, valloader_robotcar, cfg, cfg['dataset_val_robotcar'], cfg['depth_min_val_robotcar'], cfg['depth_cap_val_robotcar'], use_tqdm)
        res_val_nuscene_s = evaluate(model, valloader_nuscene, cfg, cfg['dataset_val_nuscene'], cfg['depth_min_val_nuscene'], cfg['depth_cap_val_nuscene'], use_tqdm)
        res_val_foggy_s = evaluate(model, valloader_foggy, cfg, cfg['dataset_val_foggy'], cfg['depth_min_val_foggy'], cfg['depth_cap_val_foggy'], use_tqdm)
        res_val_cloudy_s = evaluate(model, valloader_cloudy, cfg, cfg['dataset_val_cloudy'], cfg['depth_min_val_cloudy'], cfg['depth_cap_val_cloudy'], use_tqdm)
        res_val_rainy_s = evaluate(model, valloader_rainy, cfg, cfg['dataset_val_rainy'], cfg['depth_min_val_rainy'], cfg['depth_cap_val_rainy'], use_tqdm)

        res_val_kitti_c_fog_s = evaluate(model, valloader_kitti_c_fog, cfg, cfg['dataset_val_kitti_c_fog'], cfg['depth_min_val_kitti_c_fog'], cfg['depth_cap_val_kitti_c_fog'], use_tqdm)
        res_val_kitti_c_snow_s = evaluate(model, valloader_kitti_c_snow, cfg, cfg['dataset_val_kitti_c_snow'], cfg['depth_min_val_kitti_c_snow'], cfg['depth_cap_val_kitti_c_snow'], use_tqdm)
        res_val_kitti_c_dark_s = evaluate(model, valloader_kitti_c_dark, cfg, cfg['dataset_val_kitti_c_dark'], cfg['depth_min_val_kitti_c_dark'], cfg['depth_cap_val_kitti_c_dark'], use_tqdm)
        res_val_kitti_c_motion_s = evaluate(model, valloader_kitti_c_motion, cfg, cfg['dataset_val_kitti_c_motion'], cfg['depth_min_val_kitti_c_motion'], cfg['depth_cap_val_kitti_c_motion'], use_tqdm)
        res_val_kitti_c_gaussian_s = evaluate(model, valloader_kitti_c_gaussian, cfg, cfg['dataset_val_kitti_c_gaussian'], cfg['depth_min_val_kitti_c_gaussian'], cfg['depth_cap_val_kitti_c_gaussian'], use_tqdm)

        res_val_DA2K_s = evaluate_DA2K(model, valloader_DA2K, cfg['depth_cap_val_DA2K'], use_tqdm)
        res_val_DA2K_dark_s = evaluate_DA2K(model, valloader_DA2K_dark, cfg['depth_cap_val_DA2K_dark'], use_tqdm)
        res_val_DA2K_snow_s = evaluate_DA2K(model, valloader_DA2K_snow, cfg['depth_cap_val_DA2K_snow'], use_tqdm)
        res_val_DA2K_fog_s = evaluate_DA2K(model, valloader_DA2K_fog, cfg['depth_cap_val_DA2K_fog'], use_tqdm)
        res_val_DA2K_blur_s = evaluate_DA2K(model, valloader_DA2K_blur, cfg['depth_cap_val_DA2K_blur'], use_tqdm)

        model.train()
        a1_s = res_val_s['a1']
        abs_rel_s = res_val_s['abs_rel']
        a1_nyu_s = res_val_nyu_s['a1']
        abs_rel_nyu_s = res_val_nyu_s['abs_rel']
        a1_sintel_s = res_val_sintel_s['a1']
        abs_rel_sintel_s = res_val_sintel_s['abs_rel']
        a1_DIODE_s = res_val_DIODE_s['a1']
        abs_rel_DIODE_s = res_val_DIODE_s['abs_rel']
        a1_ETH3D_s = res_val_ETH3D_s['a1']
        abs_rel_ETH3D_s = res_val_ETH3D_s['abs_rel']

        a1_robotcar_s = res_val_robotcar_s['a1']
        abs_rel_robotcar_s = res_val_robotcar_s['abs_rel']
        a1_nuscene_s = res_val_nuscene_s['a1']
        abs_rel_nuscene_s = res_val_nuscene_s['abs_rel']
        a1_foggy_s = res_val_foggy_s['a1']
        abs_rel_foggy_s = res_val_foggy_s['abs_rel']
        a1_cloudy_s = res_val_cloudy_s['a1']
        abs_rel_cloudy_s = res_val_cloudy_s['abs_rel']
        a1_rainy_s = res_val_rainy_s['a1']
        abs_rel_rainy_s = res_val_rainy_s['abs_rel']

        a1_kitti_c_fog_s = res_val_kitti_c_fog_s['a1']
        abs_rel_kitti_c_fog_s = res_val_kitti_c_fog_s['abs_rel']
        a1_kitti_c_snow_s = res_val_kitti_c_snow_s['a1']
        abs_rel_kitti_c_snow_s = res_val_kitti_c_snow_s['abs_rel']
        a1_kitti_c_dark_s = res_val_kitti_c_dark_s['a1']
        abs_rel_kitti_c_dark_s = res_val_kitti_c_dark_s['abs_rel']
        a1_kitti_c_motion_s = res_val_kitti_c_motion_s['a1']
        abs_rel_kitti_c_motion_s = res_val_kitti_c_motion_s['abs_rel']
        a1_kitti_c_gaussian_s = res_val_kitti_c_gaussian_s['a1']
        abs_rel_kitti_c_gaussian_s = res_val_kitti_c_gaussian_s['abs_rel']

        acc_DA2K_s = res_val_DA2K_s['Accuracy']
        acc_DA2K_dark_s = res_val_DA2K_dark_s['Accuracy']
        acc_DA2K_snow_s = res_val_DA2K_snow_s['Accuracy']
        acc_DA2K_fog_s = res_val_DA2K_fog_s['Accuracy']
        acc_DA2K_blur_s = res_val_DA2K_blur_s['Accuracy']

        torch.distributed.barrier()

        if rank == 0:
            wandb.log({"val/a1": a1_s, "semi_charts/epoch": epoch})
            wandb.log({"val/abs_rel": abs_rel_s, "semi_charts/epoch": epoch})
            wandb.log({"val/a1_nyu": a1_nyu_s, "semi_charts/epoch": epoch})
            wandb.log({"val/abs_rel_nyu": abs_rel_nyu_s, "semi_charts/epoch": epoch})
            wandb.log({"val/a1_sintel": a1_sintel_s, "semi_charts/epoch": epoch})
            wandb.log({"val/abs_rel_sintel": abs_rel_sintel_s, "semi_charts/epoch": epoch})
            wandb.log({"val/a1_DIODE": a1_DIODE_s, "semi_charts/epoch": epoch})
            wandb.log({"val/abs_rel_DIODE": abs_rel_DIODE_s, "semi_charts/epoch": epoch})
            wandb.log({"val/a1_ETH3D": a1_ETH3D_s, "semi_charts/epoch": epoch})
            wandb.log({"val/abs_rel_ETH3D": abs_rel_ETH3D_s, "semi_charts/epoch": epoch})
            wandb.log({"val/a1_robotcar": a1_robotcar_s, "semi_charts/epoch": epoch})
            wandb.log({"val/abs_rel_robotcar": abs_rel_robotcar_s, "semi_charts/epoch": epoch})
            wandb.log({"val/a1_nuscene": a1_nuscene_s, "semi_charts/epoch": epoch})
            wandb.log({"val/abs_rel_nuscene": abs_rel_nuscene_s, "semi_charts/epoch": epoch})
            wandb.log({"val/a1_foggy": a1_foggy_s, "semi_charts/epoch": epoch})
            wandb.log({"val/abs_rel_foggy": abs_rel_foggy_s, "semi_charts/epoch": epoch})
            wandb.log({"val/a1_cloudy": a1_cloudy_s, "semi_charts/epoch": epoch})
            wandb.log({"val/abs_rel_cloudy": abs_rel_cloudy_s, "semi_charts/epoch": epoch})
            wandb.log({"val/a1_rainy": a1_rainy_s, "semi_charts/epoch": epoch})
            wandb.log({"val/abs_rel_rainy": abs_rel_rainy_s, "semi_charts/epoch": epoch})
            wandb.log({"val/a1_kitti_c_fog": a1_kitti_c_fog_s, "semi_charts/epoch": epoch})
            wandb.log({"val/abs_rel_kitti_c_fog": abs_rel_kitti_c_fog_s, "semi_charts/epoch": epoch})
            wandb.log({"val/a1_kitti_c_snow": a1_kitti_c_snow_s, "semi_charts/epoch": epoch})
            wandb.log({"val/abs_rel_kitti_c_snow": abs_rel_kitti_c_snow_s, "semi_charts/epoch": epoch})
            wandb.log({"val/a1_kitti_c_dark": a1_kitti_c_dark_s, "semi_charts/epoch": epoch})
            wandb.log({"val/abs_rel_kitti_c_dark": abs_rel_kitti_c_dark_s, "semi_charts/epoch": epoch})
            wandb.log({"val/a1_kitti_c_motion": a1_kitti_c_motion_s, "semi_charts/epoch": epoch})
            wandb.log({"val/abs_rel_kitti_c_motion": abs_rel_kitti_c_motion_s, "semi_charts/epoch": epoch})
            wandb.log({"val/a1_kitti_c_gaussian": a1_kitti_c_gaussian_s, "semi_charts/epoch": epoch})
            wandb.log({"val/abs_rel_kitti_c_gaussian": abs_rel_kitti_c_gaussian_s, "semi_charts/epoch": epoch})
            wandb.log({"val/acc_DA2K": acc_DA2K_s, "semi_charts/epoch": epoch})
            wandb.log({"val/acc_DA2K_dark": acc_DA2K_dark_s, "semi_charts/epoch": epoch})
            wandb.log({"val/acc_DA2K_snow": acc_DA2K_snow_s, "semi_charts/epoch": epoch})
            wandb.log({"val/acc_DA2K_fog": acc_DA2K_fog_s, "semi_charts/epoch": epoch})
            wandb.log({"val/acc_DA2K_blur": acc_DA2K_blur_s, "semi_charts/epoch": epoch})

            logger.info('***** Evaluation Student {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_s))
            logger.info('***** Evaluation Student_nyu {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_nyu_s))
            logger.info('***** Evaluation Student_sintel {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_sintel_s))
            logger.info('***** Evaluation Student_DIODE {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_DIODE_s))
            logger.info('***** Evaluation Student_ETH3D {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_ETH3D_s))
            logger.info('***** Evaluation Student_robotcar {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_robotcar_s))
            logger.info('***** Evaluation Student_nuscene {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_nuscene_s))
            logger.info('***** Evaluation Student_foggy {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_foggy_s))
            logger.info('***** Evaluation Student_cloudy {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_cloudy_s))
            logger.info('***** Evaluation Student_rainy {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_rainy_s))
            logger.info('***** Evaluation Student_kitti_c_fog {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_kitti_c_fog_s))
            logger.info('***** Evaluation Student_kitti_c_snow {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_kitti_c_snow_s))
            logger.info('***** Evaluation Student_kitti_c_dark {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_kitti_c_dark_s))
            logger.info('***** Evaluation Student_kitti_c_motion {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_kitti_c_motion_s))
            logger.info('***** Evaluation Student_kitti_c_gaussian {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_kitti_c_gaussian_s))
            logger.info('***** Evaluation Student_DA2K {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_DA2K_s))
            logger.info('***** Evaluation Student_DA2K_dark {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_DA2K_dark_s))
            logger.info('***** Evaluation Student_DA2K_snow {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_DA2K_snow_s))
            logger.info('***** Evaluation Student_DA2K_fog {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_DA2K_fog_s))
            logger.info('***** Evaluation Student_DA2K_blur {} ***** >>>> Metrics: {} '.format(eval_mode, res_val_DA2K_blur_s))

        if rank == 0 and need_save:
            if previous_best < a1_s:
                if previous_best != 0:
                    os.remove(os.path.join(args.save_path, 'your_exp_name_%s_%.6f.pth' % (cfg['backbone'], previous_best)))
                previous_best = a1_s
                torch.save(model.module.state_dict(), os.path.join(args.save_path, 'your_exp_name_%s_%.6f.pth' % (cfg['backbone'], a1_s)))      
        torch.distributed.barrier()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
