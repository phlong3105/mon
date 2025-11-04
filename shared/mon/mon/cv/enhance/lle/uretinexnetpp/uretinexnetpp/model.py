#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements URetinex-Net++ model for low-light image enhancement.

References:
    - Paper: "Interpretable Optimization-Inspired Unfolding Network for Low-light
      Image Enhancement," IEEE TPAMI 2025.
    - Code: https://github.com/AndersonYong/URetinex-Net-PLUS
"""

__all__ = [
    "URetinexNetPP",
]

import argparse
import time

import box
import torch
import torchvision.transforms as transforms
from PIL import Image

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, nn, Path, Task
from .network.Math_Module import P, Q
from .utils import (
    load_AdjustFusion,
    load_decom,
    load_unfolding,
    param_self_compute,
)

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


def get_params(decom_low, r, l, adjust, fusion):
    param  = 0
    param += param_self_compute(decom_low)
    param += param_self_compute(r)
    param += param_self_compute(l)
    param += param_self_compute(adjust)
    if fusion is not None:
        param += param_self_compute(fusion)
    return param


@MODELS.register(name="uretinexnet++", arch="uretinexnet++")
class URetinexNetPP(nn.Module, ModelMixin):
    """URetinex-Net++ model for low-light image enhancement.
    
    References:
        - Paper: "Interpretable Optimization-Inspired Unfolding Network for Low-light
          Image Enhancement," IEEE TPAMI 2025.
        - Code: https://github.com/AndersonYong/URetinex-Net-PLUS
    """
    
    arch     : str          = "uretinexnet++"
    name     : str          = "uretinexnet++"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, opts):
        super().__init__()
        self.opts = argparse.Namespace(**opts)
        # Loading R; old_model_opts; and L model
        self.unfolding_model_opts, self.model_R, self.model_L = load_unfolding(self.opts)
        # Loading adjustment model, fusion_model
        self.fusion_opts, self.adjust_model, self.fusion_model = load_AdjustFusion(self.opts)
        # Loading decomposition model
        self.model_Decom_low, self.model_Decom_high = load_decom(self.fusion_opts)
        
        self.P = P()
        self.Q = Q()
        transform = [
            transforms.ToTensor(),
        ]
        self.transform = transforms.Compose(transform)
        
        print("total parameters: ", get_params(self.model_Decom_low, self.model_R, self.model_L, self.adjust_model, self.fusion_model))
    
    def get_ratio(self, high_l, low_l):
        bs, c, w, h = low_l.shape
        ratio_maps = []
        for i in range(bs):
            ratio_mean = (high_l[i, :, :, :] / (low_l[i, :, :, :]+0.0001)).mean()
            if "min_ratio" in self.fusion_opts:
                ratio_mean = max(ratio_mean, self.fusion_opts.min_ratio)
                assert ratio_mean >= self.fusion_opts.min_ratio
            ratio_maps.append(torch.ones((1, c, w, h)).cuda() * ratio_mean)
        return torch.cat(ratio_maps, dim=0)
    
    def make_high_L(self, input_high_img):
        if self.model_Decom_high is not None:
            [R_high, Q_high] = self.model_Decom_high(input_high_img)
        else:
            Q_high, _ =  torch.max(input_high_img, dim=1)
            Q_high    = Q_high.unsqueeze(0)
            R_high    = input_high_img / (Q_high + 0.0001)
        return R_high, Q_high
    
    def ratio_maker(self, L, input_high_img):
        if input_high_img is None:
            ratio = torch.ones(L.shape).cuda() * self.opts.ratio
        else:
            _, Q_high = self.make_high_L(input_high_img)
            ratio     = self.get_ratio(high_l=Q_high, low_l=L)
        return ratio
    
    def unfolding(self, input_low_img, return_component=False):
        P_results, Q_results, R_results, L_results = [[], [], [], []]
        for t in range(self.unfolding_model_opts.round):
            if t == 0:  # init P, Q
                P, Q = self.model_Decom_low(input_low_img)
            else:  # update P, Q
                w_p  = (self.unfolding_model_opts.gamma + self.unfolding_model_opts.Roffset * t)
                w_q  = (self.unfolding_model_opts.lamda + self.unfolding_model_opts.Loffset * t)
                P    = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                Q    = self.Q(I=input_low_img, P=P, L=L, lamda=w_q)
            # update R, L
            # update R, L
            R = self.model_R(r=P, l=Q)
            L = self.model_L(l=Q)
            P_results.append(P)
            Q_results.append(Q)
            R_results.append(R)
            L_results.append(L)
        if not return_component:
            return R_results, L_results
        else:
            return P_results, Q_results, R_results, L_results
    
    def unfolding_inference(self, input_low_img):
        R_results, L_results = self.unfolding(input_low_img, return_component=False)
        results = []
        for t in range(len(R_results)):
            if (t+1) in self.fusion_opts.fusion_layers:
                # print("fusion layer : %d"%(t+1))
                results.append(R_results[t])
        assert len(results) == len(self.fusion_opts.fusion_layers)
        return results, L_results[-1]
    
    def adjust_and_fusion(self, inter_R_list, L, ratio):
        High_L    = self.adjust_model(l=L, alpha=ratio)
        R_enhance = None
        if self.fusion_model is not None:
            I_enhance, R_enhance = self.fusion_model(inter_R_list, High_L)
        else:
            assert len(inter_R_list) == 1
            I_enhance = inter_R_list[-1] * High_L
        return High_L, I_enhance, R_enhance
    
    def forward(self, input_low_img, input_high_img=None):
        if torch.cuda.is_available():
            input_low_img = input_low_img.cuda()
            if input_high_img is not None:
                input_high_img = input_high_img.cuda()
        with torch.no_grad():
            start = time.time()
            R_results, L = self.unfolding_inference(input_low_img=input_low_img)
            ratio  = self.ratio_maker(L, input_high_img)
            High_L, I_enhance, R_enhance = self.adjust_and_fusion(inter_R_list=R_results, L=L, ratio=ratio)
            p_time = (time.time() - start)
        return I_enhance, p_time
    
    def run(self, low_img_path):
        low_img           = self.transform(Image.open(str(low_img_path)).convert("RGB")).unsqueeze(0)
        enhance, run_time = self.forward(input_low_img=low_img)
        return enhance, run_time
