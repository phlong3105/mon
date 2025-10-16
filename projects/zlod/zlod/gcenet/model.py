#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements GCENet model for low-light image enhancement."""

__all__ = [
    "GCENet",
    "GCENet_Baseline",
    "reparameterize_model",
]

from typing import Any

import box
import torch

from mon.constants import MODELS
from mon.core import image as I, MLType, ModelMixin, nn, Path, Task
from mon.core.nn import functional as F
from mon.vision.enhance.mef import mertens_cv2
from .network import *

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Baseline -----
@MODELS.register(name="gcenet_baseline", arch="gcenet")
class GCENet_Baseline(nn.Module, ModelMixin):
    """Reimplement the Zero-DCE network as the baseline."""
    
    arch     : str          = "gcenet"
    name     : str          = "gcenet_baseline"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(
        self,
        iters  : int  = 8,
        scale  : int  = 1,
        weights: Any  = None,
        *args, **kwargs
    ):
        super().__init__()
        self.iters    = iters
        self.scale    = scale
        in_channels   = 3
        hidden_dim    = 32
        hidden_dim_x2 = hidden_dim * 2
        out_channels  = iters * 3
        self.e_conv1  = nn.Conv2d(in_channels,   hidden_dim,   3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv4  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv5  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv6  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(hidden_dim_x2, out_channels, 3, 1, 1, bias=True)
        self.relu     = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale)
        self.apply(weights_init)
        
        # Load weights
        self.load_weights(weights)
        
    def forward(self, image: torch.Tensor, inference: bool = False) -> tuple[torch.Tensor, ...]:
        # Preprocess
        if self.scale == 1:
            x_lr = image
        else:
            x_lr = F.interpolate(image, scale_factor=1 / self.scale, mode="bilinear")
        
        # Forward
        r_lr = self.learn_curve(x_lr)
        if self.scale == 1:
            r = r_lr
        else:
            r = self.upsample(r_lr)
        
        # Enhancement
        rs = torch.split(r, 3, dim=1)
        y  = image
        outputs = [r, y]
        for i in range(0, self.iters):
            y = y + rs[i] * (torch.pow(y, 2) - y)
            outputs.append(y)
            
        return outputs
    
    def learn_curve(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r  =    F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        return r


# ----- Main Model -----
@MODELS.register(name="gcenet", arch="gcenet")
class GCENet(nn.Module, ModelMixin):
    
    arch     : str          = "gcenet"
    name     : str          = "gcenet"
    tasks    : list[Task]   = [Task.LLE, Task.MEF]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, iters: int = 8, scale: int = 1, weights: Any = None, *args, **kwargs):
        super().__init__()
        self.iters    = iters
        self.scale    = scale
        in_channels   = 6
        hidden_dim    = 32
        hidden_dim_x2 = hidden_dim * 2
        out_channels  = 3
        self.e_conv1  = nn.Conv2d(in_channels,   hidden_dim,   3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv4  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv5  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv6  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(hidden_dim_x2, out_channels, 3, 1, 1, bias=True)
        self.norm32   = nn.GroupNorm(1, hidden_dim)
        self.relu     = nn.ReLU(inplace=True)
        self.denoise  = DenoiseNet(in_channels=3)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale)
        self.apply(weights_init)
        
        # Load weights
        self.load_weights(weights)
        
    def forward(self, image: torch.Tensor, inference: bool = False) -> tuple[torch.Tensor, ...]:
        # Preprocess
        # Pre-denoise & estimate noise level (copied from CLODE paper).
        noise_map = self.loss_func(image)
        denoised  = image - self.denoise(image)
        
        # Combine underexposed and overexposed parts as input (copied from CLODE paper).
        x = torch.cat([denoised, 1 - denoised], 1)
        if self.scale == 1:
            x_lr = x
        else:
            x_lr = F.interpolate(x, scale_factor=1 / self.scale, mode="bilinear")
        
        # Forward
        r_lr = self.learn_curve(x_lr)
        if self.scale == 1:
            r = r_lr
        else:
            r = self.upsample(r_lr)
        
        # Enhancement
        y       = image
        outputs = [y]
        for i in range(0, self.iters):
            y = y + r * (torch.pow(y, 2) - y)
            outputs.append(y)
        
        noise_map = torch.ones_like(r) * noise_map
        if inference:
            fused = mertens_cv2(outputs, 0.8, 0.8, 0.5)
            fused = I.to_tensor(fused, normalize=True).to(image.device)
            return {
                "curve_map": normalize_minmax(r),
                "noise_map": self.denoise(fused),
                "all"      : [o for o in outputs],  # [torch.clamp(o - self.denoise(o), 0, 1) for o in outputs],
                "output"   : fused,  # torch.clamp(fused - self.denoise(fused), 0, 1),
            }
        else:
            return {
                "curve_map": r,
                "noise_map": noise_map,
                "all"      : outputs,
                "output"   : outputs[-1],
            }
    
    def learn_curve(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.norm32(self.e_conv1(x)))
        x2 = self.relu(self.norm32(self.e_conv2(x1)))
        x3 = self.relu(self.norm32(self.e_conv3(x2)))
        x4 = self.relu(self.norm32(self.e_conv4(x3)))
        x5 = self.relu(self.norm32(self.e_conv5(torch.cat([x3, x4], 1))))
        x6 = self.relu(self.norm32(self.e_conv6(torch.cat([x2, x5], 1))))
        r  =    F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        return r
    
    # ----- Utils -----
    def pair_downsampler(self, image: torch.Tensor) -> torch.Tensor:
        c       = image.shape[1]
        filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(image.device)
        filter1 = filter1.repeat(c, 1, 1, 1)
        filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(image.device)
        filter2 = filter2.repeat(c, 1, 1, 1)
        output1 = F.conv2d(image, filter1, stride=2, groups=c)
        output2 = F.conv2d(image, filter2, stride=2, groups=c)
        return output1, output2

    def mse(self, gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        loss = torch.nn.MSELoss()
        return loss(gt, pred)
    
    def loss_func(self, noisy_img: torch.Tensor) -> torch.Tensor:
        noisy1, noisy2       = self.pair_downsampler(noisy_img)
        pred1                = noisy1 - self.denoise(noisy1)
        pred2                = noisy2 - self.denoise(noisy2)
        loss_res             = 0.5 * (self.mse(noisy1, pred2) + self.mse(noisy2, pred1))
        noisy_denoised       = noisy_img - self.denoise(noisy_img)
        denoised1, denoised2 = self.pair_downsampler(noisy_denoised)
        loss_cons            = 0.5 * (self.mse(pred1, denoised1) + self.mse(pred2, denoised2))
        loss                 = loss_res + loss_cons
        return loss


# ----- Variants -----
#@MODELS.register(name="gcenet_pono", arch="gcenet")
class GCENet_PONO(nn.Module, ModelMixin):
    """GCE-Net with Positional Normalization (PONO) and Moment Shortcut (MS)."""
    
    arch     : str          = "gcenet"
    name     : str          = "gcenet_pono"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(
        self,
        iters  : int  = 8,
        scale  : int  = 1,
        weights: Any  = None,
        *args, **kwargs
    ):
        super().__init__()
        self.iters    = iters
        self.scale    = scale
        in_channels   = 3
        hidden_dim    = 32
        hidden_dim_x2 = hidden_dim * 2
        out_channels  = iters * 3
        self.e_conv1  = nn.Conv2d(in_channels,   hidden_dim,   3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv4  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv5  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv6  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(hidden_dim_x2, out_channels, 3, 1, 1, bias=True)
        self.pono     = nn.PositionalNorm()
        self.ms       = nn.MomentShortcut()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale)
        self.apply(weights_init)
        
        # Load weights
        self.load_weights(weights)
        
    def forward(self, image: torch.Tensor, depth: torch.Tensor = None) -> tuple[torch.Tensor, ...]:
        # Preprocess
        if self.scale == 1:
            x_lr = image
        else:
            x_lr = F.interpolate(image, scale_factor=1 / self.scale, mode="bilinear")
        
        # Forward
        r_lr = self.learn_curve(x_lr)
        if self.scale == 1:
            r = r_lr
        else:
            r = self.upsample(r_lr)
        
        # Enhancement
        rs = torch.split(r, 3, dim=1)
        y  = image
        for i in range(0, self.iters):
            y = y + rs[i] * (torch.pow(y, 2) - y)
        
        return r, y
    
    def learn_curve(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.e_conv1(x)
        x1, mean1, std1 = self.pono(x1)
        x1 = F.relu(x1)
        x2 = self.e_conv2(x1)
        x2, mean2, std2 = self.pono(x2)
        x2 = F.relu(x2)
        x3 = self.e_conv3(x2)
        x3, mean3, std3 = self.pono(x3)
        x3 = F.relu(x3)
        x4 = self.e_conv4(x3)
        x4, mean4, std4 = self.pono(x4)
        x4 = F.relu(x4)
        #
        x5 = self.e_conv5(torch.cat([x3, x4], 1))
        x5 = self.ms(x5, mean3, std3)
        x5 = F.relu(x5)
        x6 = self.e_conv6(torch.cat([x2, x5], 1))
        x6 = self.ms(x6, mean2, std2)
        x6 = F.relu(x6)
        r  = self.e_conv7(torch.cat([x1, x6], 1))
        r  = self.ms(r, mean1, std1)
        r  = F.tanh(r)
        return r


#@MODELS.register(name="gcenet_bam", arch="gcenet")
class GCENet_BAM(nn.Module, ModelMixin):
    """Reimplement the Zero-DCE network as the baseline."""
    
    arch     : str          = "gcenet"
    name     : str          = "gcenet_bam"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(
        self,
        iters  : int  = 8,
        scale  : int  = 1,
        weights: Any  = None,
        *args, **kwargs
    ):
        super().__init__()
        self.iters    = iters
        self.scale    = scale
        in_channels   = 4
        hidden_dim    = 32
        hidden_dim_x2 = hidden_dim * 2
        out_channels  = 8 * 3
        self.e_conv1  = nn.Conv2d(in_channels,   hidden_dim,   3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv4  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv5  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv6  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(hidden_dim_x2, out_channels, 3, 1, 1, bias=True)
        self.relu     = nn.ReLU(inplace=True)
        self.bam      = I.BrightnessAttentionMap(gamma=1.5)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale)
        self.apply(weights_init)
        
        # Load weights
        self.load_weights(weights)
        
    def forward(self, image: torch.Tensor, depth: torch.Tensor = None) -> tuple[torch.Tensor, ...]:
        # Preprocess
        if self.scale == 1:
            x_lr = image
        else:
            x_lr = F.interpolate(image, scale_factor=1 / self.scale, mode="bilinear")
            
        # Forward
        bam  = self.bam(x_lr)
        r_lr = self.learn_curve(torch.cat([x_lr, bam], 1))
        if self.scale == 1:
            r = r_lr
        else:
            r = self.upsample(r_lr)
        
        # Enhancement
        rs = torch.split(r, 3, dim=1)
        y  = image
        for i in range(0, self.iters):
            y = y + rs[i] * (torch.pow(y, 2) - y)
        
        return r, y
    
    def learn_curve(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r  =    F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        return r


#@MODELS.register(name="gcenet_pono_bam", arch="gcenet")
class GCENet_PONO_BAM(nn.Module, ModelMixin):
    """GCE-Net with Positional Normalization (PONO) and Moment Shortcut (MS)."""
    
    arch     : str          = "gcenet"
    name     : str          = "gcenet_pono_bam"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(
        self,
        iters  : int  = 8,
        scale  : int  = 1,
        weights: Any  = None,
        *args, **kwargs
    ):
        super().__init__()
        self.iters    = iters
        self.scale    = scale
        in_channels   = 4
        hidden_dim    = 32
        hidden_dim_x2 = hidden_dim * 2
        out_channels  = iters * 3
        self.e_conv1  = nn.Conv2d(in_channels,   hidden_dim,   3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv4  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv5  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv6  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(hidden_dim_x2, out_channels, 3, 1, 1, bias=True)
        self.pono     = nn.PositionalNorm()
        self.ms       = nn.MomentShortcut()
        self.bam      = I.BrightnessAttentionMap(gamma=1.5)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale)
        self.apply(weights_init)
        
        # Load weights
        self.load_weights(weights)
        
    def forward(self, image: torch.Tensor, depth: torch.Tensor = None) -> tuple[torch.Tensor, ...]:
        # Preprocess
        if self.scale == 1:
            x_lr = image
        else:
            x_lr = F.interpolate(image, scale_factor=1 / self.scale, mode="bilinear")
        
        # Forward
        bam  = self.bam(x_lr)
        r_lr = self.learn_curve(torch.cat([x_lr, bam], 1))
        if self.scale == 1:
            r = r_lr
        else:
            r = self.upsample(r_lr)
        
        # Enhancement
        rs = torch.split(r, 3, dim=1)
        y  = image
        for i in range(0, self.iters):
            y = y + rs[i] * (torch.pow(y, 2) - y)
        
        return r, y
    
    def learn_curve(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.e_conv1(x)
        x1, mean1, std1 = self.pono(x1)
        x1 = F.relu(x1)
        x2 = self.e_conv2(x1)
        x2, mean2, std2 = self.pono(x2)
        x2 = F.relu(x2)
        x3 = self.e_conv3(x2)
        x3, mean3, std3 = self.pono(x3)
        x3 = F.relu(x3)
        x4 = self.e_conv4(x3)
        x4, mean4, std4 = self.pono(x4)
        x4 = F.relu(x4)
        #
        x5 = self.e_conv5(torch.cat([x3, x4], 1))
        x5 = self.ms(x5, mean3, std3)
        x5 = F.relu(x5)
        x6 = self.e_conv6(torch.cat([x2, x5], 1))
        x6 = self.ms(x6, mean2, std2)
        x6 = F.relu(x6)
        r  = self.e_conv7(torch.cat([x1, x6], 1))
        r  = self.ms(r, mean1, std1)
        r  = F.tanh(r)
        return r


#@MODELS.register(name="gcenet_depth", arch="gcenet")
class GCENet_Depth(nn.Module, ModelMixin):
    """GCE-Net model for low-light image enhancement."""
    
    arch     : str          = "gcenet"
    name     : str          = "gcenet_depth"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(
        self,
        iters  : int  = 8,
        scale  : int  = 1,
        weights: Any  = None,
        *args, **kwargs
    ):
        super().__init__()
        self.iters    = iters
        self.scale    = scale
        in_channels   = 3
        in_channels_1 = 1
        hidden_dim    = 32
        hidden_dim_x2 = hidden_dim * 2
        out_channels  = iters * 3
        self.d_conv1  = nn.Conv2d(in_channels_1, hidden_dim,   3, 1, 1, bias=True)
        self.d_conv2  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.d_conv3  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv1  = nn.Conv2d(in_channels,   hidden_dim,   3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv4  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv5  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv6  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(hidden_dim_x2, out_channels, 3, 1, 1, bias=True)
        self.relu     = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale)
        self.apply(weights_init)
        
        # Load weights
        self.load_weights(weights)
        
    def forward(self, image: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Preprocess
        if self.scale == 1:
            x_lr = image
            d_lr = depth
        else:
            x_lr = F.interpolate(image, scale_factor=1 / self.scale, mode="bilinear")
            d_lr = F.interpolate(depth, scale_factor=1 / self.scale, mode="bilinear")
        
        # Forward
        r_lr = self.learn_curve(x_lr, d_lr)
        if self.scale == 1:
            r = r_lr
        else:
            r = self.upsample(r_lr)
            
        # Enhancement
        rs = torch.split(r, 3, dim=1)
        y  = image
        for i in range(0, self.iters):
            y = y + rs[i] * (torch.pow(y, 2) - y)
        
        return r, y
    
    def learn_curve(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # Branch for depth
        d1 = self.relu(self.d_conv1(d))
        d2 = self.relu(self.d_conv2(d1))
        d3 = self.relu(self.d_conv3(d2))
        # Branch for adjustment curve
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(torch.cat([x1, d1], 1)))
        x3 = self.relu(self.e_conv3(torch.cat([x2, d2], 1)))
        x4 = self.relu(self.e_conv4(torch.cat([x3, d3], 1)))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r  =    F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        return r
        
    # ----- Utils -----
    def interpolate_image(self, image: torch.Tensor, size: int) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        return F.interpolate(image, size=(size, size), mode="area")
    
    def filter_up(self, x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Applies the guided filter to upscale the predicted image. """
        y_hr = self.gf(x_lr, y_lr, x_hr)
        y_hr = torch.clip(y_hr, 0.0, 1.0)
        return y_hr
