#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "UECNetwork",
]

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExNonlinearOp(nn.Module):
    
    def __init__(self, in_nc=3, out_nc=3,base_nf=64):
        super(ExNonlinearOp,self).__init__()
        self.base_nf = base_nf
        self.out_nc = out_nc
        self.encoder = nn.Conv2d(in_nc, base_nf, 1, 1)
        self.mid_conv = nn.Conv2d(base_nf, base_nf, 1, 1)
        self.decoder = nn.Conv2d(base_nf, out_nc, 1, 1)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self,x,val):
        x_code = self.encoder(x)
        x_code = self.act(x_code)
        x_code = self.act(self.mid_conv(x_code))
        y = self.decoder(x_code)
        return val*y + (1-val)*x


class ExCorrector(nn.Module):
    
    def __init__(self, in_nc=3, out_nc=3,base_nf=64):
        super(ExCorrector,self).__init__()
        self.in_nc = in_nc
        self.base_nf = base_nf
        self.out_nc = out_nc
        self.ex_block = ExNonlinearOp(in_nc,out_nc,base_nf)

    def forward(self,img,val):
        return self.ex_block(img,val)


class Encoder(nn.Module):
    
    def __init__(self, in_nc=3, encode_nf=32):
        super(Encoder, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, encode_nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(encode_nf, encode_nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.max = nn.AdaptiveMaxPool2d((1,1))

    def forward(self, x):
        b, _,_,_ = x.size()
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        std, mean = torch.std_mean(conv2_out, dim=[2, 3], keepdim=False)
        maxs = self.max(conv2_out).squeeze(2).squeeze(2)
        out = torch.cat([std, mean, maxs], dim=1)
        return out


class DiffPredictor(nn.Module):
    
    def __init__(self,fea_dim1=96,fea_dim2=8):
        super(DiffPredictor,self).__init__()
        self.fc3 = nn.Linear(fea_dim1,fea_dim2)
        self.tanh = nn.Tanh()
        self.fc4 = nn.Linear(fea_dim2*2,1)
        
    def forward(self,img_fea1, img_fea2):
        val1 = self.tanh(self.fc3(img_fea1))
        val2 = self.tanh(self.fc3(img_fea2))
        val = torch.cat([val1,val2],dim=1)
        val = self.fc4(val)
        return val


class UECNetwork(nn.Module):
    
    def __init__(
        self,
        in_channels    : int = 3,
        out_channels   : int = 3,
        base_features  : int = 64,
        encode_features: int = 32
    ):
        super().__init__()
        self.fea_dim        = encode_features * 3
        self.image_encoder  = Encoder(in_channels, encode_features)
        self.mExCorrector   = ExCorrector()
        self.mDiffPredictor =  DiffPredictor(self.fea_dim)
        self.renderers      = [self.mExCorrector]
        self.predict_heads  = [self.mDiffPredictor]

    def render(self, image: torch.Tensor, values: list[torch.Tensor]) -> list[torch.Tensor]:
        b, _, h, w = image.shape
        images     = []
        for render, scalar in zip(self.renderers, values):
            image      = render(image, scalar)
            output_img = torch.clamp(image, 0.0, 1.0)
            images.append(output_img)
        return images

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, return_vals: bool = True):
        b, _, h, w = image1.shape
        values     = []
        for render, predict_head in zip(self.renderers,self.predict_heads):
            image1_resized = F.interpolate(input=image1, size=(256, int(256 * w / h)), mode="bilinear", align_corners=False)
            image2_resized = F.interpolate(input=image2, size=(256, int(256 * w / h)), mode="bilinear", align_corners=False)
            feat1  = self.image_encoder(image1_resized)
            feat2  = self.image_encoder(image2_resized)
            scalar = predict_head(feat1, feat2)
            values.append(scalar)
            image  = render(image1, scalar)
        image = torch.clamp(image, 0, 1.0)
        
        if return_vals:
            return image, values
        else:
            return image
