#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HVI-CIDNet.

This module implements the paper: "You Only Need One Color Space: An Efficient
Network for Low-light Image Enhancement"

References:
    https://github.com/Fediory/HVI-CIDNet
"""

from __future__ import annotations

__all__ = [
    "HVICIDNet_RE"
]

import os
from collections import OrderedDict
from typing import Any, Literal

import torch
from einops import rearrange
from torchvision.models import vgg as vgg, VGG19_Weights

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision.enhance import base

console = core.console


# region Loss

_vgg_pretrain_path = "experiments/pretrained_models/vgg19-dcbb9e9d.pth"
_layer_names       = {
    "vgg11": [
        "conv1_1", "relu1_1", "pool1", "conv2_1", "relu2_1", "pool2", "conv3_1", "relu3_1", "conv3_2", "relu3_2",
        "pool3", "conv4_1", "relu4_1", "conv4_2", "relu4_2", "pool4", "conv5_1", "relu5_1", "conv5_2", "relu5_2",
        "pool5"
    ],
    "vgg13": [
        "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1", "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
        "conv3_1", "relu3_1", "conv3_2", "relu3_2", "pool3", "conv4_1", "relu4_1", "conv4_2", "relu4_2", "pool4",
        "conv5_1", "relu5_1", "conv5_2", "relu5_2", "pool5"
    ],
    "vgg16": [
        "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1", "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
        "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "pool3", "conv4_1", "relu4_1", "conv4_2",
        "relu4_2", "conv4_3", "relu4_3", "pool4", "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3",
        "pool5"
    ],
    "vgg19": [
        "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1", "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
        "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "conv3_4", "relu3_4", "pool3", "conv4_1",
        "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "conv4_4", "relu4_4", "pool4", "conv5_1", "relu5_1",
        "conv5_2", "relu5_2", "conv5_3", "relu5_3", "conv5_4", "relu5_4", "pool5"
    ]
}


def insert_bn(names: list[str]) -> list[str]:
    """Insert BN layer after each Conv.

    Args:
        names: The list of layer names.

    Returns:
        The list of layer names with BN layers.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if "conv" in name:
            position = name.replace("conv", "")
            names_bn.append("bn" + position)
    return names_bn


class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.

    In this implementation, we allow users to choose whether we use
    normalization in the input feature, and the type of vgg network. Note that
    the pretrained path must fit the vgg type.

    Args:
        layer_name_list: Forward function returns the corresponding features
            according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type: Set the type of vgg network. Default: ``'vgg19'``.
        use_input_norm: If ``True``, normalize the input image. Importantly, the
            input feature must in the range ``[0.0, 1.0]``. Default: ``True``.
        range_norm: If ``True``, norm images with range `[-1, 1]` to
            ``[0.0, 1.0]``. Default: ``False``.
        requires_grad: If ``true``, the parameters of VGG network will be
            optimized. Default: ``False``.
        remove_pooling: If ``true``, the max pooling operations in VGG net will
            be removed. Default: False.
        pooling_stride: The stride of max pooling operation. Default: ``2``.
    """

    def __init__(
        self,
        layer_name_list,
        vgg_type        = "vgg19",
        use_input_norm  = True,
        range_norm      = False,
        requires_grad   = False,
        remove_pooling  = False,
        pooling_stride  = 2
    ):
        super().__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm  = use_input_norm
        self.range_norm      = range_norm

        self.names = _layer_names[vgg_type.replace("_bn", "")]
        if "bn" in vgg_type:
            self.names = insert_bn(self.names)

        # only borrow layers that will be used to avoid unused params
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        if os.path.exists(_vgg_pretrain_path):
            vgg_net    = getattr(vgg, vgg_type)()
            state_dict = torch.load(_vgg_pretrain_path, map_location=lambda storage, loc: storage)
            vgg_net.load_state_dict(state_dict)
        else:
            vgg_net = getattr(vgg, vgg_type)(weights=VGG19_Weights.IMAGENET1K_V1)

        features = vgg_net.features[:max_idx + 1]

        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if "pool" in k:
                # if remove_pooling is true, pooling operation will be removed
                if remove_pooling:
                    continue
                else:
                    # in some cases, we may want to change the default stride
                    modified_net[k] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net).cuda()

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda())
            # the std is for image with range [0, 1]
            self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}

        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights: The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type: The type of vgg network used as feature extractor.
            Default: ``'vgg19'``.
        use_input_norm: If ``True``, normalize the input image in vgg.
            Default: True.
        range_norm: If ``True``, norm images with range ``[-1, 1]`` to
            ``[0, 1]``. Default: ``False``.
        perceptual_weight: If `perceptual_weight > 0`, the perceptual loss will
            be calculated and the loss will multiplied by the weight.
            Default: ``1.0``.
        style_weight: If `style_weight > 0`, the style loss will be calculated
            and the loss will multiplied by the weight. Default: ``0``.
        criterion: Criterion used for perceptual loss. Default: ``'l1'``.
    """
    
    def __init__(
        self,
        layer_weights,
        vgg_type          : str   = "vgg19",
        use_input_norm    : bool  = True,
        range_norm        : bool  = True,
        perceptual_weight : float = 1.0,
        style_weight      : float = 0.0,
        criterion         : str   = "l1"
    ):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight      = style_weight
        self.layer_weights     = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list = list(layer_weights.keys()),
            vgg_type        = vgg_type,
            use_input_norm  = use_input_norm,
            range_norm      = range_norm
        )
        
        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion  = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion  = nn.L2Loss()
        elif self.criterion_type == "mse":
            self.criterion  = nn.MSELoss(reduction="mean")
        elif self.criterion_type == "fro":
            self.criterion  = None
        else:
            raise NotImplementedError(f"{criterion} criterion has not been "
                                      f"supported.")
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Extract vgg features
        input_features  = self.vgg(input)
        target_features = self.vgg(target.detach())
        
        # Calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in input_features.keys():
                if self.criterion_type == "fro":
                    percep_loss += torch.norm(input_features[k] - target_features[k], p="fro") * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(input_features[k], target_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss  = None
        
        # Calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in input_features.keys():
                if self.criterion_type == "fro":
                    style_loss += torch.norm(self._gram_mat(input_features[k]) - self._gram_mat(target_features[k]), p="fro") * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(input_features[k]), self._gram_mat(target_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None
        
        return percep_loss, style_loss


class Loss(nn.Loss):
    
    def __init__(
        self,
        l1_weight    : float = 1.0,
        detail_weight: float = 0.5,
        edge_weight  : float = 50.0,
        per_weight   : float = 0.01,
        reduction    : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs, reduction=reduction)
        self.l1_weight     = l1_weight
        self.detail_weight = detail_weight
        self.edge_weight   = edge_weight
        self.per_weight    = per_weight
        
        self.l1_loss       = nn.L1Loss(loss_weight=l1_weight, reduction=reduction)
        self.detail_loss   = nn.SSIMLoss(loss_weight=detail_weight, reduction=reduction)
        self.edge_loss     = nn.EdgeLoss(loss_weight=edge_weight, reduction=reduction)
        self.per_loss      = PerceptualLoss(
            layer_weights     = {"conv1_2": 1, "conv2_2": 1, "conv3_4": 1, "conv4_4": 1},
            perceptual_weight = 1.0,
            criterion         = "mse",
        )
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_loss     = self.l1_loss(input, target)
        detail_loss = self.detail_loss(input, target)
        edge_loss   = self.edge_loss(input, target)
        per_loss    = self.per_loss(input, target)[0]
        loss        = l1_loss + detail_loss + edge_loss + self.per_weight * per_loss
        return loss
    
# endregion


# region Module

class DownsampleNorm(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        scale       : float = 0.5,
        use_norm    : bool  = False,
    ):
        super().__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.LayerNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.down  = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale)
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.down(x)
        x = self.prelu(x)
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x


class UpsampleNorm(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        scale       : float = 2.0,
        use_norm    : bool  = False
    ):
        super().__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.LayerNorm2d(out_channels)
        self.prelu    = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale)
        )
        self.up = nn.Conv2d(out_channels * 2, out_channels, 1, 1, 0, bias=False)
    
    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor
    ) -> torch.Tensor:
        input1 = self.up_scale(input1)
        input1 = torch.cat([input1, input2], dim=1)
        input1 = self.up(input1)
        input1 = self.prelu(input1)
        if self.use_norm:
            return self.norm(input1)
        else:
            return input1


class CAB(nn.Module):
    """Cross Attention Block."""
    
    def __init__(self, dim: int, num_heads: int, bias: bool):
        super().__init__()
        self.num_heads   = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q           = nn.Conv2d(dim,     dim,     1, bias=bias)
        self.q_dwconv    = nn.Conv2d(dim,     dim,     3, 1, 1, groups=dim, bias=bias)
        self.kv          = nn.Conv2d(dim,     dim * 2, 1, bias=bias)
        self.kv_dwconv   = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim,     dim,     1, bias=bias)
    
    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        x = input1
        y = input2
        b, c, h, w = x.shape
        
        q    = self.q_dwconv(self.q(x))
        kv   = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out  = (attn @ v)
        out  = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out  = self.project_out(out)
        return out


class IEL(nn.Module):
    """Intensity Enhancement Layer."""
    
    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66, bias: bool = False):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, 1, bias=bias)
        
        self.dwconv  = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, 1, 1, groups=hidden_dim * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_dim,     hidden_dim,     3, 1, 1, groups=hidden_dim,     bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_dim,     hidden_dim,     3, 1, 1, groups=hidden_dim,     bias=bias)
        
        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=bias)
        self.Tanh        = nn.Tanh()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x      = input
        x      = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1     = self.Tanh(self.dwconv1(x1)) + x1
        x2     = self.Tanh(self.dwconv2(x2)) + x2
        x      = x1 * x2
        x      = self.project_out(x)
        return x


class HV_LCA(nn.Module):
    
    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.gdfn = IEL(dim)  # IEL and CDL have same structure
        self.norm = nn.LayerNorm2d(dim)
        self.ffn  = CAB(dim, num_heads, bias)
    
    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1 = input1 + self.ffn(self.norm(input1), self.norm(input2))
        input1 = self.gdfn(self.norm(input1))
        return input1


class I_LCA(nn.Module):
    
    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm2d(dim)
        self.gdfn = IEL(dim)
        self.ffn  = CAB(dim, num_heads, bias=bias)
    
    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1 = input1 + self.ffn(self.norm(input1), self.norm(input2))
        input1 = input1 + self.gdfn(self.norm(input1))
        return input1
    
# endregion


# region Model

@MODELS.register(name="hvi_cidnet_re", arch="hvi_cidnet")
class HVICIDNet_RE(base.ImageEnhancementModel):
    """You Only Need One Color Space: An Efficient Network for Low-light Image
    Enhancement.
    
    Notes:
        - Using batch_size = 1 or 2.
        
    References:
        https://github.com/Fediory/HVI-CIDNet
    """
    
    arch   : str  = "hvi_cidnet"
    tasks  : list[Task]   = [Task.LLIE]
    schemes: list[Scheme] = [Scheme.SUPERVISED]
    zoo    : dict = {}
    
    def __init__(
        self,
        in_channels : int         = 3,
        channels    : list[int]   = [36, 36, 72, 144],
        heads       : list[int]   = [1, 2, 4, 8],
        norm        : bool        = False,
        hvi_weight  : float       = 1.0,
        loss_weights: list[float] = [1.0, 0.5, 50.0, 0.01],
        weights     : Any         = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = "hvi_cidnet_re",
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels  = self.weights.get("in_channels" , in_channels)
            channels     = self.weights.get("channels"    , channels)
            heads        = self.weights.get("heads"       , heads)
            norm         = self.weights.get("norm"        , norm)
            hvi_weight   = self.weights.get("hvi_weight"  , hvi_weight)
            loss_weights = self.weights.get("loss_weights", loss_weights)
        self.in_channels  = in_channels
        self.channels     = channels
        self.heads        = heads
        self.norm         = norm
        self.hvi_weight   = hvi_weight
        self.loss_weights = loss_weights
        
        # Construct model
        [ch1, ch2, ch3, ch4]         = self.channels
        [head1, head2, head3, head4] = self.heads
        
        # HW Branch
        self.hve_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.hve_block1 = DownsampleNorm(ch1, ch2, use_norm=norm)
        self.hve_block2 = DownsampleNorm(ch2, ch3, use_norm=norm)
        self.hve_block3 = DownsampleNorm(ch3, ch4, use_norm=norm)
        
        self.hvd_block3 = UpsampleNorm(ch4, ch3, use_norm=norm)
        self.hvd_block2 = UpsampleNorm(ch3, ch2, use_norm=norm)
        self.hvd_block1 = UpsampleNorm(ch2, ch1, use_norm=norm)
        self.hvd_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )
        
        # I Branch
        self.ie_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.ie_block1 = DownsampleNorm(ch1, ch2, use_norm=norm)
        self.ie_block2 = DownsampleNorm(ch2, ch3, use_norm=norm)
        self.ie_block3 = DownsampleNorm(ch3, ch4, use_norm=norm)
        
        self.id_block3 = UpsampleNorm(ch4, ch3, use_norm=norm)
        self.id_block2 = UpsampleNorm(ch3, ch2, use_norm=norm)
        self.id_block1 = UpsampleNorm(ch2, ch1, use_norm=norm)
        self.id_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )
        
        self.hv_lca1 = HV_LCA(ch2, head2)
        self.hv_lca2 = HV_LCA(ch3, head3)
        self.hv_lca3 = HV_LCA(ch4, head4)
        self.hv_lca4 = HV_LCA(ch4, head4)
        self.hv_lca5 = HV_LCA(ch3, head3)
        self.hv_lca6 = HV_LCA(ch2, head2)
        
        self.i_lca1  = I_LCA(ch2, head2)
        self.i_lca2  = I_LCA(ch3, head3)
        self.i_lca3  = I_LCA(ch4, head4)
        self.i_lca4  = I_LCA(ch4, head4)
        self.i_lca5  = I_LCA(ch3, head3)
        self.i_lca6  = I_LCA(ch2, head2)
        
        self.trans   = core.RGBToHVI()
        
        # Loss
        self.loss = Loss(*self.loss_weights, reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        pass
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        pred_rgb   = outputs.get("enhanced")
        pred_hvi   = self.rgb_to_hvi(pred_rgb)
        target_rgb = datapoint.get("hq_image")
        target_hvi = self.rgb_to_hvi(target_rgb)
        loss_rgb   = self.loss(pred_rgb, target_rgb)
        loss_hvi   = self.loss(pred_hvi, target_hvi)
        loss       = loss_rgb + self.hvi_weight * loss_hvi
        # Return
        return {
            "enhanced": pred_rgb,
            "hvi_k"   : float(self.trans.density_k.item()),
            "loss"    : loss,
        }
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x        = datapoint.get("image")
        dtypes   = x.dtype
        hvi      = self.trans.rgb_to_hvi(x)
        i        = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
        
        i_enc0   = self.ie_block0(i)
        i_enc1   = self.ie_block1(i_enc0)
        hv_0     = self.hve_block0(hvi)
        hv_1     = self.hve_block1(hv_0)
        i_jump0  = i_enc0
        hv_jump0 = hv_0
        
        i_enc2   = self.i_lca1(i_enc1, hv_1)
        hv_2     = self.hv_lca1(hv_1, i_enc1)
        v_jump1  = i_enc2
        hv_jump1 = hv_2
        i_enc2   = self.ie_block2(i_enc2)
        hv_2     = self.hve_block2(hv_2)
        
        i_enc3   = self.i_lca2(i_enc2, hv_2)
        hv_3     = self.hv_lca2(hv_2, i_enc2)
        v_jump2  = i_enc3
        hv_jump2 = hv_3
        i_enc3   = self.ie_block3(i_enc2)
        hv_3     = self.hve_block3(hv_2)
        
        i_enc4   = self.i_lca3(i_enc3, hv_3)
        hv_4     = self.hv_lca3(hv_3, i_enc3)
        
        i_dec4   = self.i_lca4(i_enc4, hv_4)
        hv_4     = self.hv_lca4(hv_4, i_enc4)
        
        hv_3     = self.hvd_block3(hv_4, hv_jump2)
        i_dec3   = self.id_block3(i_dec4, v_jump2)
        i_dec2   = self.i_lca5(i_dec3, hv_3)
        hv_2     = self.hv_lca5(hv_3, i_dec3)
        
        hv_2     = self.hvd_block2(hv_2, hv_jump1)
        i_dec2   = self.id_block2(i_dec3, v_jump1)
        
        i_dec1   = self.i_lca6(i_dec2, hv_2)
        hv_1     = self.hv_lca6(hv_2, i_dec2)
        
        i_dec1   = self.id_block1(i_dec1, i_jump0)
        i_dec0   = self.id_block0(i_dec1)
        hv_1     = self.hvd_block1(hv_1, hv_jump0)
        hv_0     = self.hvd_block0(hv_1)
        
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.hvi_to_rgb(output_hvi)
        
        return {"enhanced": output_rgb}
    
    def rgb_to_hvi(self, input: torch.Tensor) -> torch.Tensor:
        return self.trans.rgb_to_hvi(input)
        
# endregion
