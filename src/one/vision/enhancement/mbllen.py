#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from torchvision.models import vgg19
from torchvision.models import VGG19_Weights

from one.nn import *

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Loss -------------------------------------------------------------------

def ssim(
    input      : Tensor,
    target     : Tensor,
    cs_map     : bool  = False,
    mean_metric: bool  = True,
    depth      : int   = 1,
    size       : int   = 11,
    sigma      : float = 1.5,
) -> Optional[Tensor]:
    """
    Calculate the SSIM (Structural Similarity Index) score between 2 4D-/3D-
    channel-first- images.
    """
    input  = input.to(torch.float64)
    target = target.to(torch.float64)

    # Window shape [size, size]
    window = fspecial_gauss(size=size, sigma=sigma)
    window = window.to(input.get_device())
    
    # Depth of image (255 in case the image has a different scale)
    l      = depth
    c1     = (0.01 * l) ** 2
    c2     = (0.03 * l) ** 2

    mu1       = F.conv2d(input=input,  weight=window, stride=1)
    mu2       = F.conv2d(input=target, weight=window, stride=1)
    mu1_sq    = mu1 * mu1
    mu2_sq    = mu2 * mu2
    mu1_mu2   = mu1 * mu2
    sigma1_sq = F.conv2d(input=input  * input,  weight=window, stride=1) - mu1_sq
    sigma2_sq = F.conv2d(input=target * target, weight=window, stride=1) - mu2_sq
    sigma12   = F.conv2d(input=input  * target, weight=window, stride=1) - mu1_mu2

    if cs_map:
        score = (
            ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) /
            ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)),
            (2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
        )
    else:
        score = (
            ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2))
            / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        )

    if mean_metric:
        score = torch.mean(score)
    score = score.detach()
    return score


def mae(input: Tensor, target: Tensor) -> Tensor:
    """
    Calculate MAE (Absolute Error) score between 2 4D-/3D- channel-first-
    images.
    """
    input  = input.to(torch.float64)
    target = target.to(torch.float64)
    score  = torch.mean(torch.abs(input - target) ** 2)
    return score


def fspecial_gauss(size: int, sigma: float) -> Tensor:
    """
    Function to mimic the `special` gaussian MATLAB function.

    Args:
        size (int): Size of gaussian's window. Defaults to 11.
        sigma (float): Sigma value of gaussian's window. Defaults to 1.5.
    """
    x_data, y_data = np.mgrid[-size // 2 + 1: size // 2 + 1,
                              -size // 2 + 1: size // 2 + 1]
    x_data = np.expand_dims(x_data, axis=0)
    x_data = np.expand_dims(x_data, axis=0)
    y_data = np.expand_dims(y_data, axis=0)
    y_data = np.expand_dims(y_data, axis=0)
    x      = torch.from_numpy(x_data)
    y      = torch.from_numpy(y_data)
    x      = x.type(torch.float64)
    y      = y.type(torch.float64)
    z      = -((x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    g      = torch.exp(z)
    return g / torch.sum(g)


def range_scale(input: Tensor) -> Tensor:
    return input * 2.0 - 1.0


# noinspection PyMethodMayBeStatic
@LOSSES.register(name="mbllen_loss")
class MBLLENLoss(Module):
    """
    Implementation of loss function defined in the paper "MBLLEN: Low-light
    Image/Video Enhancement Using CNNs".
    """
    
    def __init__(self):
        super().__init__()
        self.name = "mbllen_loss"
        self.vgg  = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # self.vgg = VGG19(out_indexes=26, pretrained=True)
        # self.vgg.freeze()
        
    def forward(self, input: Tensor, target: Tensor, **_) -> Tensor:
        """
        Mbllen_loss = (structure_loss + (context_loss / 3.0) + 3 +
        region_loss)

        Args:
            input (Tensor): Enhanced images.
            target (Tensor): Normal-light images.
        
        Returns:
            loss (Tensor): Loss.
        """
        loss = (
            self.structure_loss(input, target) +
            self.context_loss(input, target) / 3.0 +
            3 + self.region_loss(input, target)
		)
        return loss
    
    def context_loss(self, input: Tensor, target: Tensor) -> Tensor:
        b, c, h, w     = [int(x) for x in target.shape]
        y_hat_scale    = range_scale(input)
        y_hat_features = self.vgg.forward_features(y_hat_scale)
        y_hat_features = torch.reshape(y_hat_features, shape=(-1, 16, h, w))
    
        y_scale    = range_scale(target)
        y_features = self.vgg.forward_features(y_scale)
        y_features = torch.reshape(y_features, shape=(-1, 16, h, w))
    
        loss = torch.mean(
            torch.abs(y_hat_features[:, :16, :, :] - y_features[:, :16, :, :])
        )
        return loss
    
    def region_loss(
        self, input: Tensor, target: Tensor, dark_pixel_percent: float = 0.4
    ) -> Tensor:
        """
        Implementation of region loss function defined in the paper
        "MBLLEN: Low-light Image/Video Enhancement Using CNNs".
        
        Args:
            input (Tensor): Enhanced images.
            target (Tensor): Normal-light images.
            dark_pixel_percent (float): Defaults to 0.4.
                
        Returns:
            loss (Tensor): Region loss.
        """
        index     = int(256 * 256 * dark_pixel_percent - 1)
        gray1     = (0.39 * input[:, 0, :, :] + 0.5 * input[:, 1, :, :] +
                     0.11 * input[:, 2, :, :])
        gray      = torch.reshape(gray1, [-1, 256 * 256])
        gray_sort = torch.topk(-gray, k=256 * 256)[0]
        yu        = gray_sort[:, index]
        yu        = torch.unsqueeze(input=torch.unsqueeze(input=yu, dim=-1),
                                    dim=-1)
        mask      = (gray1 <= yu).type(torch.float64)
        mask1     = torch.unsqueeze(input=mask, dim=1)
        mask      = torch.cat(tensors=[mask1, mask1, mask1], dim=1)
    
        low_fake_clean  = torch.mul(mask, input[:, :3, :, :])
        high_fake_clean = torch.mul(1 - mask, input[:, :3, :, :])
        low_clean       = torch.mul(mask, target[:, : , :, :])
        high_clean      = torch.mul(1 - mask, target[:, : , :, :])
        loss            = torch.mean(torch.abs(low_fake_clean - low_clean) * 4 +
                                     torch.abs(high_fake_clean - high_clean))
        
        return loss
    
    def structure_loss(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Implementation of structure loss function defined in the paper
        "MBLLEN: Low-light Image/Video Enhancement Using CNNs".
        
        Args:
            input (Tensor): Enhanced images.
            target (Tensor): Normal-light images.
    
        Returns:
            loss (Tensor): Structure loss.
        """
        mae_loss  = mae(input[:, :3, :, :], target)
        ssim_loss = (
            ssim(torch.unsqueeze(input[:, 0, :, :], dim=1),
                 torch.unsqueeze(target[:, 0, :, :], dim=1))
            + ssim(torch.unsqueeze(input[:, 1, :, :], dim=1),
                   torch.unsqueeze(target[:, 1, :, :], dim=1))
            + ssim(torch.unsqueeze(input[:, 2, :, :], dim=1),
                   torch.unsqueeze(target[:, 2, :, :], dim=1))
        )
        loss = mae_loss - ssim_loss
        return loss


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "mbllen": {
        "channels": 3,
        "backbone": [
            # [from,    number, module,     args(out_channels, ...)]
            [-1,        1,      ConvReLU2d, [32, 3, 1, 1, 1, 1, True, "replicate"]],  # 0  (fem1)
            [-1,        1,      ConvReLU2d, [32, 3, 1, 1, 1, 1, True, "replicate"]],  # 1  (fem2)
            [-1,        1,      ConvReLU2d, [32, 3, 1, 1, 1, 1, True, "replicate"]],  # 2  (fem3)
            [-1,        1,      ConvReLU2d, [32, 3, 1, 1, 1, 1, True, "replicate"]],  # 3  (fem4)
            [-1,        1,      ConvReLU2d, [32, 3, 1, 1, 1, 1, True, "replicate"]],  # 4  (fem5)
            [-1,        1,      ConvReLU2d, [32, 3, 1, 1, 1, 1, True, "replicate"]],  # 5  (fem6)
            [-1,        1,      ConvReLU2d, [32, 3, 1, 1, 1, 1, True, "replicate"]],  # 6  (fem7)
            [-1,        1,      ConvReLU2d, [32, 3, 1, 1, 1, 1, True, "replicate"]],  # 7  (fem8)
            [-1,        1,      ConvReLU2d, [32, 3, 1, 1, 1, 1, True, "replicate"]],  # 8  (fem9)
            [-1,        1,      ConvReLU2d, [32, 3, 1, 1, 1, 1, True, "replicate"]],  # 9  (fem10)
            [ 0,        1,      EM,         [8, 3, 5]],                               # 10 (em1)
            [ 1,        1,      EM,         [8, 3, 5]],                               # 11 (em2)
            [ 2,        1,      EM,         [8, 3, 5]],                               # 12 (em3)
            [ 3,        1,      EM,         [8, 3, 5]],                               # 13 (em4)
            [ 4,        1,      EM,         [8, 3, 5]],                               # 14 (em5)
            [ 5,        1,      EM,         [8, 3, 5]],                               # 15 (em6)
            [ 6,        1,      EM,         [8, 3, 5]],                               # 16 (em7)
            [ 7,        1,      EM,         [8, 3, 5]],                               # 17 (em8)
            [ 8,        1,      EM,         [8, 3, 5]],                               # 18 (em9)
            [ 9,        1,      EM,         [8, 3, 5]],                               # 19 (em10)
            [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 1, Concat, []],                # 20
        ],
        "head": [
            [20,        1,      ConvReLU2d, [32, 3, 1, 1, 1, 1, True, "replicate"]],  # 21
        ]
    },
}


@MODELS.register(name="mbllen")
class MBLLEN(ImageEnhancementModel):
    """
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE
    """
    
    def __init__(
        self,
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "mbllen",
        fullname   : str  | None         = "mbllen",
        cfg        : dict | Path_ | None = "mbllen.yaml",
        channels   : int                 = 3,
        num_classes: int  | None 		 = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = MBLLENLoss(),
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "mbllen"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg

        super().__init__(
            root        = root,
            name        = name,
            fullname    = fullname,
            cfg         = cfg,
            channels    = channels,
            num_classes = num_classes,
            pretrained  = pretrained,
            phase       = phase,
            loss        = loss or MBLLENLoss(),
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
   
    def init_weights(self, m: Module):
        pass
