#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements URetinex-Net model for low-light image enhancement.

References:
    - Paper: "URetinex-Net: Retinex-based Deep Unfolding Network for
      Low-light-Image-Enhancement," CVPR 2022.
    - Code: https://github.com/AndersonYong/URetinex-Net
"""

__all__ = [
    "URetinexNet",
]

import time

import box
import torch
import torchvision.transforms as transforms
from PIL import Image

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, nn, Path, Task
from .network.decom import Decom
from .network.Math_Module import P, Q
from .utils import load_adjustment, load_initialize, load_unfolding

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


@MODELS.register(name="uretinexnet", arch="uretinexnet")
class URetinexNet(nn.Module, ModelMixin):
    """URetinex-Net model for low-light image enhancement.
    
    References:
        - Paper: "URetinex-Net: Retinex-based Deep Unfolding Network for
          Low-light-Image-Enhancement," CVPR 2022.
        - Code: https://github.com/AndersonYong/URetinex-Net
    """
    
    arch     : str          = "uretinexnet"
    name     : str          = "uretinexnet"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        # Loading decomposition model
        self.model_Decom_low = Decom()
        self.model_Decom_low = load_initialize(self.model_Decom_low, self.opts["decom_model_low_weights"])
        # Loading R; old_model_opts; and L model
        self.unfolding_opts, self.model_R, self.model_L = load_unfolding(self.opts["unfolding_model_weights"])
        # Loading adjustment model
        self.adjust_model    = load_adjustment(self.opts["adjust_model_weights"])
        self.P = P()
        self.Q = Q()
        transform = [
            transforms.ToTensor(),
            # transforms.Resize(1280),
        ]
        self.transform = transforms.Compose(transform)
        # mon.log(self.model_Decom_low)
        # mon.log(self.model_R)
        # mon.log(self.model_L)
        # mon.log(self.adjust_model)
        # time.sleep(8)

    def unfolding(self, input_low_img):
        for t in range(self.unfolding_opts.round):
            if t == 0:  # Initialize R0, L0
                P, Q = self.model_Decom_low(input_low_img)
            else:  # Update P and Q
                w_p = (self.unfolding_opts.gamma + self.unfolding_opts.Roffset * t)
                w_q = (self.unfolding_opts.lamda + self.unfolding_opts.Loffset * t)
                P   = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                Q   = self.Q(I=input_low_img, P=P, L=L, lamda=w_q)
            R = self.model_R(r=P, l=Q)
            L = self.model_L(l=Q)
        return R, L
    
    def illumination_adjust(self, L, ratio):
        ratio = torch.ones(L.shape).cuda() * ratio
        return self.adjust_model(l=L, alpha=ratio)
    
    def forward(self, input_low_img):
        if torch.cuda.is_available():
            input_low_img = input_low_img.cuda()
        with torch.no_grad():
            start_time = time.time()
            R, L       = self.unfolding(input_low_img)
            High_L     = self.illumination_adjust(L, self.opts["ratio"])
            I_enhance  = High_L * R
            run_time   = (time.time() - start_time)
        return I_enhance, run_time

    def run(self, low_img_path):
        low_img           = self.transform(Image.open(str(low_img_path)).convert("RGB")).unsqueeze(0)
        enhance, run_time = self.forward(input_low_img=low_img)
        """
        file_name = os.path.basename(self.opts.img_path)
        name      = file_name.split('.')[0]
        if not os.path.exists(self.opts.output):
            os.makedirs(self.opts.output)
        save_path = os.path.join(self.opts.output, file_name.replace(name, "%s_%d_URetinexNet"%(name, self.opts.ratio)))
        np_save_TensorImg(enhance, save_path)
        mon.log("================================= time for %s: %f============================"%(file_name, p_time))
        """
        return enhance, run_time
