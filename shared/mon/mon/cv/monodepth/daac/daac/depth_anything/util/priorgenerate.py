from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoPriorGen(nn.Module):

    def __init__(self,weight=[0.5,0.5]):
        super().__init__()
        self.weight = weight
        

    def generate_depth_decay(self, H: int, W: int, depth_grid):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        '''
        B,_,H,W = depth_grid.shape
        grid_d = depth_grid.reshape(B, H*W, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :] 
        mask_d = (mask_d.abs()).sum(dim=-1)
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None] 
        return mask_d
        
    def generate_pos_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H*W, 2) 
        mask = grid[:, None, :] - grid[None, :, :] 
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None] 
        return mask
    
    def generate_1d_depth_decay(self, H, W, depth_grid):
        '''
        generate 1d depth decay mask, the result is l*l
        '''
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None, None]
        assert mask.shape[2:] == (W,H,H)
        return mask
    
    
    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :] 
        mask = mask.abs()
        mask = mask * self.decay[:, None, None] 
        return mask
    
    def forward(self, HW_tuple: Tuple[int], depth_map, split_or_not=False):
        '''
        depth_map: depth patches  
        HW_tuple: (H, W)
        H * W == l
        '''
        depth_map = F.interpolate(depth_map, size=HW_tuple,mode='bilinear',align_corners=False)

        if split_or_not:
            index = torch.arange(HW_tuple[0]*HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]) 
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1) 
            cos = torch.cos(index[:, None] * self.angle[None, :]) 
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1) 

            mask_d_h = self.generate_1d_depth_decay(HW_tuple[0], HW_tuple[1], depth_map.transpose(-2,-1))
            mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)


            mask_h = self.generate_1d_decay(HW_tuple[0])
            mask_w = self.generate_1d_decay(HW_tuple[1])

            mask_h = self.weight[0]*mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1]*mask_d_h
            mask_w = self.weight[0]*mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1]*mask_d_w 
            

            geo_prior = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(HW_tuple[0]*HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]) 
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1) 
            cos = torch.cos(index[:, None] * self.angle[None, :]) 
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1) 
            mask = self.generate_pos_decay(HW_tuple[0], HW_tuple[1]) 

            mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map)
            mask = (self.weight[0]*mask+self.weight[1]*mask_d)

            geo_prior = ((sin, cos), mask)

        return geo_prior
