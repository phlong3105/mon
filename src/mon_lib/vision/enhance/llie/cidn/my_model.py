#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements CIDN (Cross-Image Disentanglement for Low-Light
Enhancement in Real World) models.
"""

from __future__ import annotations

__all__ = [

]

import functools

import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler

from mon import core, nn
from mon.core import _size_2_t
from mon.nn import functional as F

console = core.console


# region Module

# region Basic Functions

def get_scheduler(optimizer, opts, cur_ep: int = -1):
    if opts.lr_policy == "lambda":
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    else:
        return NotImplementedError("No such learning rate policy")
    return scheduler


def mean_pool_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    sequence  = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def conv_mean_pool(in_channels: int, out_channels: int) -> nn.Sequential:
    sequence  = []
    sequence += conv3x3(in_channels, out_channels)
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


def get_norm_layer(layer_type: str = "instance") -> nn.Module:
    if layer_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("Normalization layer [%s] is not found" % layer_type)
    return norm_layer


def get_non_linearity(layer_type: str = "relu") -> nn.Module:
    if layer_type == "relu":
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == "lrelu":
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)
    elif layer_type == "elu":
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError("Non-linearity activation [%s] is not found" % layer_type)
    return nl_layer


def conv3x3(in_channels: int, out_channels: int) -> list:
    return [
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3, 1, 0, bias=True),
    ]


def gaussian_weights_init(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname.find("Conv") == 0:
        m.weight.data.normal_(0.0, 0.02)
        
# endregion


# region Basic Blocks

class LayerNorm(nn.Module):
    
    def __init__(self, n_out: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.n_out  = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias   = nn.Parameter(torch.zeros(n_out, 1, 1))
        return
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_shape = input.size()[1:]
        if self.affine:
            return F.layer_norm(input, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(input, normalized_shape)


class BasicBlock(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        norm_layer  = None,
        nl_layer    = None,
    ):
        super().__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(in_channels)]
        layers += [nl_layer()]
        layers += conv3x3(in_channels, in_channels)
        if norm_layer is not None:
            layers += [norm_layer(in_channels)]
        layers += [nl_layer()]
        layers += [conv_mean_pool(in_channels, out_channels)]
        self.conv     = nn.Sequential(*layers)
        self.shortcut = mean_pool_conv(in_channels, out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv(input) + self.shortcut(input)
        return output


class LeakyReLUConv2d(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t,
        padding     : _size_2_t | str = 0,
        norm        = "None",
        sn          : bool = False,
    ):
        super().__init__()
        model  = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if "norm" == "Instance":
            model += [nn.InstanceNorm2d(out_channels, affine=False)]
        model     += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)


class ReLUINSConv2d(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t,
        padding     : _size_2_t | str = 0,
    ):
        super().__init__()
        model      = []
        model     += [nn.ReflectionPad2d(padding)]
        model     += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        model     += [nn.InstanceNorm2d(out_channels, affine=False)]
        model     += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)


class INSResBlock(nn.Module):
    
    @staticmethod
    def conv3x3(in_channels: int, out_channels: int, stride: _size_2_t = 1) -> list:
        return [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        ]

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : _size_2_t = 1,
        dropout     : float     = 0.0,
    ):
        super().__init__()
        model  = []
        model += self.conv3x3(in_channels, out_channels, stride)
        model += [nn.InstanceNorm2d(out_channels)]
        model += [nn.ReLU(inplace=True)]
        model += self.conv3x3(out_channels, out_channels)
        model += [nn.InstanceNorm2d(out_channels)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        output   = self.model(input)
        output  += residual
        return output


class MisINSResBlock(nn.Module):
    
    @staticmethod
    def conv3x3(in_channels: int, out_channels: int, stride: _size_2_t = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        )

    @staticmethod
    def conv1x1(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def __init__(
        self,
        dim      : int,
        dim_extra: int,
        stride   : _size_2_t = 1,
        dropout  : float     = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim)
        )
        self.conv2 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim)
        )
        self.blk1 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False)
        )
        self.blk2 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False)
        )
        model = []
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.blk1.apply(gaussian_weights_init)
        self.blk2.apply(gaussian_weights_init)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        residual  = x
        z_expand  = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        o1        = self.conv1(x)
        o2        = self.blk1(torch.cat([o1, z_expand], dim=1))
        o3        = self.conv2(o2)
        output    = self.blk2(torch.cat([o3, z_expand], dim=1))
        output   += residual
        return output


class GaussianNoiseLayer(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return input
        noise = Variable(torch.randn(input.size()).cuda(input.get_device()))
        return input + noise


class ReLUINSConvTranspose2d(nn.Module):
    
    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        kernel_size   : _size_2_t,
        stride        : _size_2_t,
        padding       : _size_2_t | str,
        output_padding: _size_2_t,
    ):
        super().__init__()
        model  = []
        model += [
            nn.ConvTranspose2d(
                in_channels    = in_channels,
                out_channels   = out_channels,
                kernel_size    = kernel_size,
                stride         = stride,
                padding        = padding,
                output_padding = output_padding,
                bias           = True,
            )
        ]
        model += [LayerNorm(out_channels)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)
    
# endregion


# region Spectral Normalization

class SpectralNorm:
    
    def __init__(
        self,
        name              : str   = "weight",
        n_power_iterations: int   = 1,
        dim               : int   = 0,
        eps               : float = 1e-12,
    ):
        self.name = name
        self.dim  = dim
        if n_power_iterations <= 0:
            raise ValueError("Expected n_power_iterations to be positive, but got n_power_iterations={}".format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps  = eps

    def compute_weight(self, module: nn.Module):
        weight = getattr(module, self.name + "_orig")
        u      = getattr(module, self.name + "_u")
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim])
        height     = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
        sigma  = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u

    def remove(self, module: nn.Module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + "_u")
        delattr(module, self.name + "_orig")
        module.register_parameter(self.name, torch.nn.Parameter(weight))

    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + "_u", u)
        else:
            r_g = getattr(module, self.name + "_orig").requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(
        module            : nn.Module,
        name              : str,
        n_power_iterations: int,
        dim               : int,
        eps               : float,
    ):
        fn     = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)
        u      = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(
    module            : nn.Module,
    name              : str        = "weight",
    n_power_iterations: int        = 1,
    eps               : float      = 1e-12,
    dim               : int | None = None
) -> nn.Module:
    if dim is None:
        if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module: nn.Module, name: str = "weight"):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module
    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))

# endregion


# region Discriminators

class DiscriminatorContent(nn.Module):
    
    def __init__(self):
        super().__init__()
        model  = []
        model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm="Instance")]
        model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm="Instance")]
        model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm="Instance")]
        model += [LeakyReLUConv2d(256, 256, kernel_size=4, stride=1, padding=0)]
        model += [nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)
    
    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        output  = self.model(input)
        output  = output.view(-1)
        outputs = [output]
        return outputs


class MultiScaleDiscriminator(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        num_scales : int  = 3,
        num_layers : int  = 4,
        norm       : str  = "None",
        sn         : bool = False,
    ):
        super().__init__()
        channels        = 64
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.diss       = nn.ModuleList()
        for _ in range(num_scales):
            self.diss.append(self._make_net(channels, in_channels, num_layers, norm, sn))

    @staticmethod
    def _make_net(channels: int, in_channels, num_layers: int, norm: str, sn: bool) -> nn.Sequential:
        model  = []
        model += [LeakyReLUConv2d(in_channels, channels, 4, 2, 1, norm, sn)]
        tch    = channels
        for _ in range(1, num_layers):
            model += [LeakyReLUConv2d(tch, tch * 2, 4, 2, 1, norm, sn)]
            tch   *= 2
        if sn:
            model += [spectral_norm(nn.Conv2d(tch, 1, 1, 1, 0))]
        else:
            model += [nn.Conv2d(tch, 1, 1, 1, 0)]
        return nn.Sequential(*model)

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for dis in self.diss:
            outputs.append(dis(input))
            input = self.downsample(input)
        return outputs


class Discriminator(nn.Module):
    
    def __init__(self, in_channels: int, norm: str = "None", sn: bool = False):
        super().__init__()
        channels   = 64
        n_layers   = 6
        self.model = self._make_net(channels, in_channels, n_layers, norm, sn)

    @staticmethod
    def _make_net(channels: int, in_channels: int, n_layers: int, norm: str, sn: bool) -> nn.Sequential:
        model  = []
        model += [LeakyReLUConv2d(in_channels, channels, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]  # 16
        tch    = channels
        for i in range(1, n_layers - 1):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]  # 8
            tch   *= 2
        model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)]  # 2
        tch   *= 2
        if sn:
            model += [spectral_norm(nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0))]  # 1
        else:
            model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1
        return nn.Sequential(*model)

    def cuda(self, gpu: int):
        self.model.cuda(gpu)

    def forward(self, x_a: torch.Tensor) -> list[torch.Tensor]:
        out_a  = self.model(x_a)
        out_a  = out_a.view(-1)
        outs_a = [out_a]
        return outs_a

# endregion


# region Encoder

class EncoderContent(nn.Module):
    
    def __init__(self, in_channels_a: int, out_channels_b: int):
        super().__init__()
        enc_a_c   = []
        channels  = 64
        enc_a_c  += [LeakyReLUConv2d(in_channels_a, channels, kernel_size=7, stride=1, padding=3)]
        for i in range(1, 3):
            enc_a_c  += [ReLUINSConv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1)]
            channels *= 2
        for i in range(0, 3):
            enc_a_c  += [INSResBlock(channels, channels)]

        enc_b_c   = []
        channels  = 64
        enc_b_c  += [LeakyReLUConv2d(out_channels_b, channels, kernel_size=7, stride=1, padding=3)]
        for i in range(1, 3):
            enc_b_c  += [ReLUINSConv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1)]
            channels *= 2
        for i in range(0, 3):
            enc_b_c  += [INSResBlock(channels, channels)]

        enc_share = []
        for i in range(0, 1):
            enc_share += [INSResBlock(channels, channels)]
            enc_share += [GaussianNoiseLayer()]
            self.conv_share = nn.Sequential(*enc_share)
        
        self.conv_a = nn.Sequential(*enc_a_c)
        self.conv_b = nn.Sequential(*enc_b_c)

    def forward(
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_a = self.conv_a(input_a)
        output_b = self.conv_b(input_b)
        output_a = self.conv_share(output_a)
        output_b = self.conv_share(output_b)
        return output_a, output_b

    def forward_a(self, input_a: torch.Tensor) -> torch.Tensor:
        output_a = self.conv_a(input_a)
        output_a = self.conv_share(output_a)
        return output_a

    def forward_b(self, input_b: torch.Tensor) -> torch.Tensor:
        output_b = self.conv_b(input_b)
        output_b = self.conv_share(output_b)
        return output_b


class EncoderContentShare(nn.Module):
    
    def __init__(self, in_channels_a: int, in_channels_b: int):
        super().__init__()
        enc_share  = []
        channels   = 64
        enc_share += [LeakyReLUConv2d(in_channels_a, channels, kernel_size=7, stride=1, padding=3)]
        for i in range(1, 3):
            enc_share += [ReLUINSConv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1)]
            channels  *= 2
        for i in range(0, 3):
            enc_share += [INSResBlock(channels, channels)]

        # enc_share1 = []
        for i in range(0, 1):
            enc_share += [INSResBlock(channels, channels)]
            enc_share += [GaussianNoiseLayer()]
            self.conv_share = nn.Sequential(*enc_share)

        self.conv_a = nn.Sequential(*enc_share)
        self.conv_b = nn.Sequential(*enc_share)

    def forward(
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_a = self.conv_a(input_a)
        output_b = self.conv_b(input_b)
        # output_a = self.conv_share(output_a)
        # output_b = self.conv_share(output_b)
        return output_a, output_b

    def forward_a(self, input_a: torch.Tensor) -> torch.Tensor:
        output_a = self.conv_a(input_a)
        # output_a = self.conv_share(output_a)
        return output_a

    def forward_b(self, input_b: torch.Tensor) -> torch.Tensor:
        output_b = self.conv_b(input_b)
        # output_b = self.conv_share(output_b)
        return output_b


class EncoderAttribute(nn.Module):
    
    def __init__(self, in_channels_a: int, in_channels_b: int, out_channels: int = 8):
        super().__init__()
        channels = 64
        self.model_a = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels_a, channels, 7, 1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels * 2, 4, 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels * 2, channels * 4, 4, 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels * 4, channels * 4, 4, 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels * 4, channels * 4, 4, 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 4, out_channels, 1, 1, 0))
        self.model_b = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels_b, channels, 7, 1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels * 2, 4, 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels * 2, channels * 4, 4, 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels * 4, channels * 4, 4, 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels * 4, channels * 4, 4, 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 4, out_channels, 1, 1, 0))

    def forward(
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_a  = self.model_a(input_a)
        input_b  = self.model_b(input_b)
        output_a = input_a.view(input_a.size(0), -1)
        output_b = input_b.view(input_b.size(0), -1)
        return output_a, output_b

    def forward_a(self, input_a: torch.Tensor) -> torch.Tensor:
        input_a  = self.model_a(input_a)
        output_a = input_a.view(input_a.size(0), -1)
        return output_a

    def forward_b(self, input_b: torch.Tensor) -> torch.Tensor:
        input_b  = self.model_b(input_b)
        output_b = input_b.view(input_b.size(0), -1)
        return output_b


class EncoderAttributeConcat(nn.Module):
    
    def __init__(
        self,
        in_channels_a: int,
        in_channels_b: int,
        out_channels : int = 8,
        norm_layer   = None,
        nl_layer     = None
    ):
        super().__init__()
        ndf      = 64
        n_blocks = 4
        max_ndf  = 4
        # E_A
        conv_layers_a  = [nn.ReflectionPad2d(1)]
        conv_layers_a += [nn.Conv2d(in_channels_a, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
        for n in range(1, n_blocks):
            input_ndf      = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf     = ndf * min(max_ndf, n + 1)  # 2**n
            conv_layers_a += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        conv_layers_a += [nl_layer(), nn.AdaptiveAvgPool2d(1)]  # AvgPool2d(13)
        self.fc_a      = nn.Sequential(*[nn.Linear(output_ndf, out_channels)])
        self.fc_var_a  = nn.Sequential(*[nn.Linear(output_ndf, out_channels)])
        self.conv_a    = nn.Sequential(*conv_layers_a)
        
        # E_B
        conv_layers_b  = [nn.ReflectionPad2d(1)]
        conv_layers_b += [nn.Conv2d(in_channels_b, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
        for n in range(1, n_blocks):
            input_ndf      = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf     = ndf * min(max_ndf, n + 1)  # 2**n
            conv_layers_b += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        conv_layers_b += [nl_layer(), nn.AdaptiveAvgPool2d(1)]  # AvgPool2d(13)
        self.fc_b      = nn.Sequential(*[nn.Linear(output_ndf, out_channels)])
        self.fc_var_b  = nn.Sequential(*[nn.Linear(output_ndf, out_channels)])
        self.conv_b    = nn.Sequential(*conv_layers_b)

    def forward(
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_conv_a     = self.conv_a(input_a)
        conv_flat_a  = x_conv_a.view(input_a.size(0), -1)
        output_a     = self.fc_a(conv_flat_a)
        output_var_a = self.fc_var_a(conv_flat_a)
        x_conv_b     = self.conv_b(input_b)
        conv_flat_b  = x_conv_b.view(input_b.size(0), -1)
        output_b     = self.fc_b(conv_flat_b)
        output_var_b = self.fc_var_b(conv_flat_b)
        return output_a, output_var_a, output_b, output_var_b

    def forward_a(self, input_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_conv_a     = self.conv_a(input_a)
        conv_flat_a  = x_conv_a.view(input_a.size(0), -1)
        output_a     = self.fc_a(conv_flat_a)
        output_var_a = self.fc_var_a(conv_flat_a)
        return output_a, output_var_a

    def forward_b(self, input_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_conv_b    = self.conv_b(input_b)
        conv_flat_b   = x_conv_b.view(input_b.size(0), -1)
        output_b     = self.fc_b(conv_flat_b)
        output_var_b = self.fc_var_b(conv_flat_b)
        return output_b, output_var_b


class EncoderAttributeConcatShare(nn.Module):
    
    def __init__(
        self,
        in_channels_a: int,
        in_channels_b: int,
        out_channels : int = 8,
        norm_layer         = None,
        nl_layer           = None
    ):
        super().__init__()
        ndf      = 64
        n_blocks = 4
        max_ndf  = 4
        
        conv_layers_a  = [nn.ReflectionPad2d(1)]
        conv_layers_a += [nn.Conv2d(in_channels_a, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
        for n in range(1, n_blocks):
            input_ndf      = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf     = ndf * min(max_ndf, n + 1)  # 2**n
            conv_layers_a += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        conv_layers_a += [nl_layer(), nn.AdaptiveAvgPool2d(1)]  # AvgPool2d(13)
        self.fc_a     = nn.Sequential(*[nn.Linear(output_ndf, out_channels)])
        self.fc_var_a = nn.Sequential(*[nn.Linear(output_ndf, out_channels)])
        self.conv_a   = nn.Sequential(*conv_layers_a)

        # conv_layers_B = [nn.ReflectionPad2d(1)]
        # conv_layers_B += [nn.Conv2d(input_dim_b, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
        # for n in range(1, n_blocks):
        #     input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
        #     output_ndf = ndf * min(max_ndf, n + 1)  # 2**n
        #     conv_layers_B += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        # conv_layers_B += [nl_layer(), nn.AdaptiveAvgPool2d(1)]  # AvgPool2d(13)
        # self.fc_B = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        # self.fcVar_B = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        # self.conv_B = nn.Sequential(*conv_layers_B)
    
    def forward(
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_conv_a     = self.conv_a(input_a)
        conv_flat_a  = x_conv_a.view(input_a.size(0), -1)
        output_a     = self.fc_a(conv_flat_a)
        output_var_a = self.fc_var_a(conv_flat_a)
        x_conv_b     = self.conv_a(input_b)
        conv_flat_b  = x_conv_b.view(input_b.size(0), -1)
        output_b     = self.fc_a(conv_flat_b)
        output_var_b = self.fc_var_a(conv_flat_b)
        return output_a, output_var_a, output_b, output_var_b
    
    def forward_a(self, input_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_conv_a     = self.conv_a(input_a)
        conv_flat_a  = x_conv_a.view(input_a.size(0), -1)
        output_a     = self.fc_a(conv_flat_a)
        output_var_a = self.fc_var_a(conv_flat_a)
        return output_a, output_var_a
    
    def forward_b(self, input_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_conv_b     = self.conv_a(input_b)
        conv_flat_b  = x_conv_b.view(input_b.size(0), -1)
        output_b     = self.fc_a(conv_flat_b)
        output_var_b = self.fc_var_a(conv_flat_b)
        return output_b, output_var_b
    
# endregion


# region Generator

class Generator(nn.Module):
    
    def __init__(self, out_channels_a: int, out_channels_b: int, nz):
        super().__init__()
        self.nz = nz
        ini_tch = 256
        tch_add = ini_tch
        tch     = ini_tch
        self.tch_add = tch_add
        self.decA1   = MisINSResBlock(tch, tch_add)
        self.decA2   = MisINSResBlock(tch, tch_add)
        self.decA3   = MisINSResBlock(tch, tch_add)
        self.decA4   = MisINSResBlock(tch, tch_add)

        decA5       = []
        decA5      += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch         = tch // 2
        decA5      += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch         = tch // 2
        decA5      += [nn.ConvTranspose2d(tch, out_channels_a, kernel_size=1, stride=1, padding=0)]
        decA5      += [nn.Tanh()]
        self.decA5  = nn.Sequential(*decA5)

        tch         = ini_tch
        self.decB1  = MisINSResBlock(tch, tch_add)
        self.decB2  = MisINSResBlock(tch, tch_add)
        self.decB3  = MisINSResBlock(tch, tch_add)
        self.decB4  = MisINSResBlock(tch, tch_add)
        decB5       = []
        decB5      += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch         = tch // 2
        decB5      += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch         = tch // 2
        decB5      += [nn.ConvTranspose2d(tch, out_channels_b, kernel_size=1, stride=1, padding=0)]
        decB5      += [nn.Tanh()]
        self.decB5  = nn.Sequential(*decB5)

        self.mlpA = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, tch_add * 4)
        )
        self.mlpB = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, tch_add * 4)
        )

    def forward_a(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = self.mlpA(z)
        z1, z2, z3, z4 = torch.split(z, self.tch_add, dim=1)
        z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
        out1 = self.decA1(x, z1)
        out2 = self.decA2(out1, z2)
        out3 = self.decA3(out2, z3)
        out4 = self.decA4(out3, z4)
        out  = self.decA5(out4)
        return out

    def forward_b(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = self.mlpB(z)
        z1, z2, z3, z4 = torch.split(z, self.tch_add, dim=1)
        z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
        out1 = self.decB1(x, z1)
        out2 = self.decB2(out1, z2)
        out3 = self.decB3(out2, z3)
        out4 = self.decB4(out3, z4)
        out  = self.decB5(out4)
        return out


class GeneratorConcat(nn.Module):
    
    def __init__(self, out_channels_a: int, out_channels_b: int, nz: int):
        super().__init__()
        self.nz    = nz
        tch        = 256
        dec_share  = []
        dec_share += [INSResBlock(tch, tch)]
        self.dec_share = nn.Sequential(*dec_share)
        tch   = 256 + self.nz
        decA1 = []
        for i in range(0, 3):
            decA1 += [INSResBlock(tch, tch)]
        tch   = tch + self.nz
        decA2 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        tch   = tch // 2
        tch   = tch + self.nz
        decA3 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        tch   = tch // 2
        tch   = tch + self.nz
        decA4 = [nn.ConvTranspose2d(tch, out_channels_a, kernel_size=1, stride=1, padding=0)] + [nn.Tanh()]
        self.decA1 = nn.Sequential(*decA1)
        self.decA2 = nn.Sequential(*[decA2])
        self.decA3 = nn.Sequential(*[decA3])
        self.decA4 = nn.Sequential(*decA4)

        tch   = 256 + self.nz
        decB1 = []
        for i in range(0, 3):
            decB1 += [INSResBlock(tch, tch)]
        tch   = tch + self.nz
        decB2 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        tch   = tch // 2
        tch   = tch + self.nz
        decB3 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        tch   = tch // 2
        tch   = tch + self.nz
        decB4 = [nn.ConvTranspose2d(tch, out_channels_b, kernel_size=1, stride=1, padding=0)] + [nn.Tanh()]
        self.decB1 = nn.Sequential(*decB1)
        self.decB2 = nn.Sequential(*[decB2])
        self.decB3 = nn.Sequential(*[decB3])
        self.decB4 = nn.Sequential(*decB4)

    def forward_a(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out0     = self.dec_share(x)
        z_img    = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_and_z  = torch.cat([out0, z_img], 1)
        out1     = self.decA1(x_and_z)
        z_img2   = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
        x_and_z2 = torch.cat([out1, z_img2], 1)
        out2     = self.decA2(x_and_z2)
        z_img3   = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
        x_and_z3 = torch.cat([out2, z_img3], 1)
        out3     = self.decA3(x_and_z3)
        z_img4   = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
        x_and_z4 = torch.cat([out3, z_img4], 1)
        out4     = self.decA4(x_and_z4)
        return out4
    
    def forward_a(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out0     = self.dec_share(x)
        z_img    = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_and_z  = torch.cat([out0, z_img], 1)
        out1     = self.decB1(x_and_z)
        z_img2   = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
        x_and_z2 = torch.cat([out1, z_img2], 1)
        out2     = self.decB2(x_and_z2)
        z_img3   = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
        x_and_z3 = torch.cat([out2, z_img3], 1)
        out3     = self.decB3(x_and_z3)
        z_img4   = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
        x_and_z4 = torch.cat([out3, z_img4], 1)
        out4     = self.decB4(x_and_z4)
        return out4
    
# endregion

# endregion


# region Model

class CIDN(nn.Module):
    
    def __init__(self, opts):
        super().__init__()
        
        # parameters
        lr = 0.0001
        lr_d_content = lr / 2.5
        self.nz = 8
        self.concat = opts.concat
        self.no_ms = opts.no_ms
        self.content_share   = opts.content_share
        self.attribute_share = opts.attribute_share
        self.vgg = opts.vgg
        self.gpu = opts.gpu
        
        # perceptual
        if opts.vgg > 0:
            self.vgg_loss = PerceptualLoss(opts)
            # if self.opt.IN_vgg:
            #   self.vgg_patch_loss = PerceptualLoss(opts)
            #   self.vgg_patch_loss.cuda()
            self.vgg_loss.cuda()
            self.vgg_net = load_vgg16("../model", self.gpu)
            self.vgg_net.eval()
            for param in self.vgg_net.parameters():
                param.requires_grad = False
        
        # discriminators
        if opts.dis_scale > 1:
            self.d_a  = MultiScaleDiscriminator(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.d_b  = MultiScaleDiscriminator(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.d_a2 = MultiScaleDiscriminator(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.d_b2 = MultiScaleDiscriminator(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
        else:
            self.d_a  = Discriminator(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.d_b  = Discriminator(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.d_a2 = Discriminator(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.d_b2 = Discriminator(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
        self.d_content = DiscriminatorContent()
        
        # encoders
        if self.content_share:
            self.e_content = EncoderContentShare(opts.input_dim_a, opts.input_dim_b)
        else:
            self.e_content = EncoderContent(opts.input_dim_a, opts.input_dim_b)
        
        if self.concat:
            if self.attribute_share:
                self.e_attribute = EncoderAttributeConcatShare(opts.input_dim_a, opts.input_dim_b, self.nz, norm_layer=None, nl_layer=get_non_linearity(layer_type='lrelu'))
            else:
                self.e_attribute = EncoderAttributeConcat(opts.input_dim_a, opts.input_dim_b, self.nz, norm_layer=None, nl_layer=get_non_linearity(layer_type='lrelu'))
        else:
            self.e_attribute = EncoderAttribute(opts.input_dim_a, opts.input_dim_b, self.nz)
        
        # Generator
        if self.concat:
            self.g = GeneratorConcat(opts.input_dim_a, opts.input_dim_b, nz=self.nz)
        else:
            self.g = Generator(opts.input_dim_a, opts.input_dim_b, nz=self.nz)
        
        # Optimizers
        self.d_a_optim         = torch.optim.Adam(self.d_a.parameters(),         lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.d_b_optim         = torch.optim.Adam(self.d_b.parameters(),         lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.d_a2_optim        = torch.optim.Adam(self.d_a2.parameters(),        lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.d_b2_optim        = torch.optim.Adam(self.d_b2.parameters(),        lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.d_content_optim   = torch.optim.Adam(self.d_content.parameters(),   lr=lr_d_content, betas=(0.5, 0.999), weight_decay=0.0001)
        self.e_content_optim   = torch.optim.Adam(self.e_content.parameters(),   lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.e_attribute_optim = torch.optim.Adam(self.e_attribute.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.g_optim           = torch.optim.Adam(self.g.parameters(),           lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        
        # Setup the loss function for training
        self.criterionL1 = torch.nn.L1Loss()
    
    def initialize(self):
        self.d_a.apply(gaussian_weights_init)
        self.d_b.apply(gaussian_weights_init)
        self.d_a2.apply(gaussian_weights_init)
        self.d_b2.apply(gaussian_weights_init)
        self.d_content.apply(gaussian_weights_init)
        self.g.apply(gaussian_weights_init)
        self.e_content.apply(gaussian_weights_init)
        self.e_attribute.apply(gaussian_weights_init)
    
    def set_scheduler(self, opts, last_ep=0):
        self.d_a_sch         = get_scheduler(self.d_a_optim        , opts, last_ep)
        self.d_b_sch         = get_scheduler(self.d_b_optim        , opts, last_ep)
        self.d_a2_sch        = get_scheduler(self.d_a2_optim       , opts, last_ep)
        self.d_b2_sch        = get_scheduler(self.d_b2_optim       , opts, last_ep)
        self.d_content_sch   = get_scheduler(self.d_content_optim  , opts, last_ep)
        self.e_content_sch   = get_scheduler(self.e_content_optim  , opts, last_ep)
        self.e_attribute_sch = get_scheduler(self.e_attribute_optim, opts, last_ep)
        self.g_sch           = get_scheduler(self.g_optim          , opts, last_ep)
    
    def setgpu(self, gpu):
        self.gpu = gpu
        self.d_a.cuda(self.gpu)
        self.d_b.cuda(self.gpu)
        self.d_a2.cuda(self.gpu)
        self.d_b2.cuda(self.gpu)
        self.d_content.cuda(self.gpu)
        self.e_content.cuda(self.gpu)
        self.e_attribute.cuda(self.gpu)
        self.g.cuda(self.gpu)
    
    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z = torch.randn(batchSize, nz).cuda(self.gpu)
        return z
    
    def test_forward(self, image, a2b=True):
        self.z_random = self.get_z_random(image.size(0), self.nz, 'gauss')
        if a2b:
            self.z_content = self.e_content.forward_a(image)
            output = self.g.forward_b(self.z_content, self.z_random)
        else:
            self.z_content = self.e_content.forward_b(image)
            output = self.g.forward_a(self.z_content, self.z_random)
        return output
    
    def test_forward_transfer(self, image_a, image_b, a2b=True):
        self.z_content_a, self.z_content_b = self.e_content.forward(image_a, image_b)
        if self.concat:
            self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.e_attribute.forward(image_a, image_b)
            std_a = self.logvar_a.mul(0.5).exp_()
            eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
            self.z_attr_a = eps.mul(std_a).add_(self.mu_a)
            std_b = self.logvar_b.mul(0.5).exp_()
            eps = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
            self.z_attr_b = eps.mul(std_b).add_(self.mu_b)
        else:
            self.z_attr_a, self.z_attr_b = self.e_attribute.forward(image_a, image_b)
        if a2b:
            output = self.g.forward_b(self.z_content_a, self.z_attr_b)
        else:
            output = self.g.forward_a(self.z_content_b, self.z_attr_a)
        return output
    
    def forward(self):
        # input images
        half_size = 1
        real_A = self.input_A
        real_B = self.input_B
        self.real_A_encoded = real_A[0:half_size]
        self.real_A_random = real_A[half_size:]
        self.real_B_encoded = real_B[0:half_size]
        self.real_B_random = real_B[half_size:]
        
        # get encoded z_c
        self.z_content_a, self.z_content_b = self.e_content.forward(self.real_A_encoded, self.real_B_encoded)
        # fea_a = self.real_A_encoded.mean(1, keepdim=True)
        # fea_a = fea_a.cpu().detach().numpy()
        # fea_a = np.squeeze(fea_a, axis=0)
        # fea_a = np.transpose(fea_a, (1, 2, 0))
        # # plt.imshow(fea_a, cmap=plt.get_cmap('gray'))
        # # io.show()
        # fea_b = self.real_B_encoded.mean(1, keepdim=True)
        # fea_b = fea_b.cpu().detach().numpy()
        # fea_b = np.squeeze(fea_b, axis=0)
        # fea_b = np.transpose(fea_b, (1, 2, 0))
        # # plt.imshow(fea_b, cmap=plt.get_cmap('gray'))
        # # io.show()
        # fea_a = self.z_content_a.mean(1, keepdim=True)
        # fea_a = fea_a.cpu().detach().numpy()
        # fea_a = np.squeeze(fea_a, axis=0)
        # fea_a = np.transpose(fea_a, (1, 2, 0))
        # # plt.imshow(fea_a, cmap=plt.get_cmap('gray'))
        # io.show()
        # fea_b = self.z_content_b.mean(1, keepdim=True)
        # fea_b = fea_b.cpu().detach().numpy()
        # fea_b = np.squeeze(fea_b, axis=0)
        # fea_b = np.transpose(fea_b, (1, 2, 0))
        # plt.imshow(fea_b, cmap=plt.get_cmap('gray'))
        # io.show()
        # # plt.imshow(c_b, cmap=plt.get_cmap('gray'))
        # #     # io.show()
        # get encoded z_a
        if self.concat:
            self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.e_attribute.forward(
                self.real_A_encoded, self.real_B_encoded
                )
            std_a = self.logvar_a.mul(0.5).exp_()
            eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
            self.z_attr_a = eps_a.mul(std_a).add_(self.mu_a)
            std_b = self.logvar_b.mul(0.5).exp_()
            eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
            self.z_attr_b = eps_b.mul(std_b).add_(self.mu_b)
        else:
            self.z_attr_a, self.z_attr_b = self.e_attribute.forward(self.real_A_encoded, self.real_B_encoded)
        
        # get random z_a
        self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')
        if not self.no_ms:
            self.z_random2 = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')
        
        if not self.no_ms:
            input_content_forB = torch.cat((self.z_content_b, self.z_content_b, self.z_content_b), 0)
            input_content_forA = torch.cat((self.z_content_a, self.z_content_a, self.z_content_a), 0)
            input_attr_forA = torch.cat((self.z_attr_a, self.z_random, self.z_random2), 0)
            input_attr_forB = torch.cat((self.z_attr_b, self.z_random, self.z_random2), 0)
            output_fakeA = self.g.forward_a(input_content_forA, input_attr_forA)
            output_fakeB = self.g.forward_b(input_content_forB, input_attr_forB)
            self.fake_AA_encoded, self.fake_AA_random, self.fake_AA_random2 = torch.split(
                output_fakeA, self.z_content_a.size(0), dim=0
                )
            self.fake_BB_encoded, self.fake_BB_random, self.fake_BB_random2 = torch.split(
                output_fakeB, self.z_content_a.size(0), dim=0
                )
        else:
            input_content_forB = torch.cat((self.z_content_b, self.z_content_b), 0)
            input_content_forA = torch.cat((self.z_content_a, self.z_content_a), 0)
            input_attr_forA = torch.cat((self.z_attr_a, self.z_random), 0)
            input_attr_forB = torch.cat((self.z_attr_b, self.z_random), 0)
            output_fakeA = self.g.forward_a(input_content_forA, input_attr_forA)
            output_fakeB = self.g.forward_b(input_content_forB, input_attr_forB)
            self.fake_AA_encoded, self.fake_AA_random = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
            self.fake_BB_encoded, self.fake_BB_random = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)
        
        # first cross translation
        if not self.no_ms:
            input_content_forA = torch.cat((self.z_content_b, self.z_content_b, self.z_content_b), 0)
            input_content_forB = torch.cat((self.z_content_a, self.z_content_a, self.z_content_a), 0)
            input_attr_forA = torch.cat((self.z_attr_a, self.z_random, self.z_random2), 0)
            input_attr_forB = torch.cat((self.z_attr_b, self.z_random, self.z_random2), 0)
            output_fakeA = self.g.forward_a(input_content_forA, input_attr_forA)
            output_fakeB = self.g.forward_b(input_content_forB, input_attr_forB)
            self.fake_A_encoded, self.fake_A_random, self.fake_A_random2 = torch.split(
                output_fakeA, self.z_content_a.size(0), dim=0
                )
            self.fake_B_encoded, self.fake_B_random, self.fake_B_random2 = torch.split(
                output_fakeB, self.z_content_a.size(0), dim=0
                )
        else:
            input_content_forA = torch.cat((self.z_content_b, self.z_content_b), 0)
            input_content_forB = torch.cat((self.z_content_a, self.z_content_a), 0)
            input_attr_forA = torch.cat((self.z_attr_a, self.z_random), 0)
            input_attr_forB = torch.cat((self.z_attr_b, self.z_random), 0)
            output_fakeA = self.g.forward_a(input_content_forA, input_attr_forA)
            output_fakeB = self.g.forward_b(input_content_forB, input_attr_forB)
            self.fake_A_encoded, self.fake_A_random = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
            self.fake_B_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)
        
        # # get reconstructed encoded z_c
        # self.z_content_recon_b, self.z_content_recon_a = self.enc_c.forward(self.fake_A_encoded, self.fake_B_encoded)
        
        # get reconstructed encoded z_a
        # if self.concat:
        #   self.mu_recon_a, self.logvar_recon_a, self.mu_recon_b, self.logvar_recon_b = self.enc_a.forward(self.fake_A_encoded, self.fake_B_encoded)
        #   std_a = self.logvar_recon_a.mul(0.5).exp_()
        #   eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
        #   self.z_attr_recon_a = eps_a.mul(std_a).add_(self.mu_recon_a)
        #   std_b = self.logvar_recon_b.mul(0.5).exp_()
        #   eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
        #   self.z_attr_recon_b = eps_b.mul(std_b).add_(self.mu_recon_b)
        # else:
        #   self.z_attr_recon_a, self.z_attr_recon_b = self.enc_a.forward(self.fake_A_encoded, self.fake_B_encoded)
        #
        # # second cross translation
        # self.fake_A_recon = self.gen.forward_a(self.z_content_recon_a, self.z_attr_recon_a)
        # self.fake_B_recon = self.gen.forward_b(self.z_content_recon_b, self.z_attr_recon_b)
        
        # for display
        self.image_display = torch.cat(
            (self.real_A_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(), \
             self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(), \
             self.real_B_encoded[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(), \
             self.fake_A_random[0:1].detach().cpu(), self.fake_BB_encoded[0:1].detach().cpu()), dim=0
            )
        
        # # for latent regression
        # if self.concat:
        #   self.mu2_a, _, self.mu2_b, _ = self.enc_a.forward(self.fake_A_random, self.fake_B_random)
        # else:
        #   self.z_attr_random_a, self.z_attr_random_b = self.enc_a.forward(self.fake_A_random, self.fake_B_random)
    
    def forward_content(self):
        half_size = 1
        self.real_A_encoded = self.input_A[0:half_size]
        self.real_B_encoded = self.input_B[0:half_size]
        # get encoded z_c
        self.z_content_a, self.z_content_b = self.e_content.forward(self.real_A_encoded, self.real_B_encoded)
    
    def update_D_content(self, image_a, image_b):
        self.input_A = image_a
        self.input_B = image_b
        self.forward_content()
        self.d_content_optim.zero_grad()
        loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm_(self.d_content.parameters(), 5)
        self.d_content_optim.step()
    
    def update_D(self, image_a, image_b):
        self.input_A = image_a
        self.input_B = image_b
        self.forward()
        
        # update disA
        self.d_a_optim.zero_grad()
        loss_D1_A = self.backward_D(self.d_a, self.real_A_encoded, self.fake_A_encoded)
        self.disA_loss = loss_D1_A.item()
        self.d_a_optim.step()
        
        # update disA2
        self.d_a2_optim.zero_grad()
        loss_D2_A = self.backward_D(self.d_a2, self.real_A_random, self.fake_A_random)
        self.disA2_loss = loss_D2_A.item()
        if not self.no_ms:
            loss_D2_A2 = self.backward_D(self.d_a2, self.real_A_random, self.fake_A_random2)
            self.disA2_loss += loss_D2_A2.item()
        self.d_a2_optim.step()
        
        # update disB
        self.d_b_optim.zero_grad()
        loss_D1_B = self.backward_D(self.d_b, self.real_B_encoded, self.fake_B_encoded)
        self.disB_loss = loss_D1_B.item()
        self.d_b_optim.step()
        
        # update disB2
        self.d_b2_optim.zero_grad()
        loss_D2_B = self.backward_D(self.d_b2, self.real_B_random, self.fake_B_random)
        self.disB2_loss = loss_D2_B.item()
        if not self.no_ms:
            loss_D2_B2 = self.backward_D(self.d_b2, self.real_B_random, self.fake_B_random2)
            self.disB2_loss += loss_D2_B2.item()
        self.d_b2_optim.step()
        
        # update disContent
        self.d_content_optim.zero_grad()
        loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm_(self.d_content.parameters(), 5)
        self.d_content_optim.step()
    
    def backward_D(self, netD, real, fake):
        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)
        loss_D = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = nn.functional.sigmoid(out_a)
            out_real = nn.functional.sigmoid(out_b)
            all0 = torch.zeros_like(out_fake).cuda(self.gpu)
            all1 = torch.ones_like(out_real).cuda(self.gpu)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss_D += ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D
    
    def backward_contentD(self, imageA, imageB):
        pred_fake = self.d_content.forward(imageA.detach())
        pred_real = self.d_content.forward(imageB.detach())
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = nn.functional.sigmoid(out_a)
            out_real = nn.functional.sigmoid(out_b)
            all1 = torch.ones((out_real.size(0))).cuda(self.gpu)
            all0 = torch.zeros((out_fake.size(0))).cuda(self.gpu)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
        loss_D = ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D
    
    def update_EG(self):
        # update G, Ec, Ea
        self.e_content_optim.zero_grad()
        self.e_attribute_optim.zero_grad()
        self.g_optim.zero_grad()
        self.backward_EG()
        self.e_content_optim.step()
        self.e_attribute_optim.step()
        self.g_optim.step()
        
        # update G, Ec
        self.e_content_optim.zero_grad()
        self.g_optim.zero_grad()
        self.backward_G_alone()
        self.e_content_optim.step()
        self.g_optim.step()
    
    def backward_EG(self):
        # content Ladv for generator
        loss_G_GAN_Acontent = self.backward_G_GAN_content(self.z_content_a)
        loss_G_GAN_Bcontent = self.backward_G_GAN_content(self.z_content_b)
        
        # Ladv for generator
        loss_G_GAN_A = self.backward_G_GAN(self.fake_A_encoded, self.d_a)
        loss_G_GAN_B = self.backward_G_GAN(self.fake_B_encoded, self.d_b)
        
        # KL loss - z_a
        if self.concat:
            kl_element_a = self.mu_a.pow(2).add_(self.logvar_a.exp()).mul_(-1).add_(1).add_(self.logvar_a)
            loss_kl_za_a = torch.sum(kl_element_a).mul_(-0.5) * 0.01
            kl_element_b = self.mu_b.pow(2).add_(self.logvar_b.exp()).mul_(-1).add_(1).add_(self.logvar_b)
            loss_kl_za_b = torch.sum(kl_element_b).mul_(-0.5) * 0.01
        else:
            loss_kl_za_a = self._l2_regularize(self.z_attr_a) * 0.01
            loss_kl_za_b = self._l2_regularize(self.z_attr_b) * 0.01
        
        # KL loss - z_c
        loss_kl_zc_a = self._l2_regularize(self.z_content_a) * 0.01
        loss_kl_zc_b = self._l2_regularize(self.z_content_b) * 0.01
        
        # cross cycle consistency loss
        # loss_G_L1_A = self.criterionL1(self.fake_A_recon, self.real_A_encoded) * 10
        # loss_G_L1_B = self.criterionL1(self.fake_B_recon, self.real_B_encoded) * 10
        loss_content_ab = self.criterionL1(self.z_content_a, self.z_content_b) * 10
        loss_G_L1_AA = self.criterionL1(self.fake_AA_encoded, self.real_A_encoded) * 10
        loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded) * 10
        
        # perceptual loss
        if self.vgg > 0:
            loss_vgg_a = self.vgg_loss.compute_vgg_loss(self.vgg_net, self.fake_A_encoded, self.input_A)
            loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg_net, self.fake_B_encoded, self.input_B)
            loss_vgg_random_a = self.vgg_loss.compute_vgg_loss(self.vgg_net, self.fake_A_random, self.input_A)
            loss_vgg_random_b = self.vgg_loss.compute_vgg_loss(self.vgg_net, self.fake_B_random, self.input_B)
            loss_G = loss_G_GAN_A + loss_G_GAN_B + \
                     loss_G_GAN_Acontent + loss_G_GAN_Bcontent + \
                     loss_vgg_random_a + loss_vgg_random_b + \
                     loss_vgg_a + loss_vgg_b + \
                     loss_G_L1_AA + loss_G_L1_BB + loss_content_ab + \
                     loss_kl_zc_a + loss_kl_zc_b + \
                     loss_kl_za_a + loss_kl_za_b
            
            loss_G.backward(retain_graph=True)
            
            self.loss_vgg_a = loss_vgg_a.item()
            self.loss_vgg_b = loss_vgg_b.item()
        else:
            loss_G = loss_G_GAN_A + loss_G_GAN_B + \
                     loss_G_GAN_Acontent + loss_G_GAN_Bcontent + \
                     loss_G_L1_AA + loss_G_L1_BB + loss_content_ab + \
                     loss_kl_zc_a + loss_kl_zc_b + \
                     loss_kl_za_a + loss_kl_za_b
            
            loss_G.backward(retain_graph=True)
        
        self.content_loss_ab = loss_content_ab.item()
        self.gan_loss_a = loss_G_GAN_A.item()
        self.gan_loss_b = loss_G_GAN_B.item()
        self.gan_loss_acontent = loss_G_GAN_Acontent.item()
        self.gan_loss_bcontent = loss_G_GAN_Bcontent.item()
        self.kl_loss_za_a = loss_kl_za_a.item()
        self.kl_loss_za_b = loss_kl_za_b.item()
        self.kl_loss_zc_a = loss_kl_zc_a.item()
        self.kl_loss_zc_b = loss_kl_zc_b.item()
        self.l1_recon_AA_loss = loss_G_L1_AA.item()
        self.l1_recon_BB_loss = loss_G_L1_BB.item()
        self.G_loss = loss_G.item()
    
    def backward_G_GAN_content(self, data):
        outs = self.d_content.forward(data)
        for out in outs:
            outputs_fake = nn.functional.sigmoid(out)
            all_half = 0.5 * torch.ones((outputs_fake.size(0))).cuda(self.gpu)
            ad_loss = nn.functional.binary_cross_entropy(outputs_fake, all_half)
        return ad_loss
    
    def backward_G_GAN(self, fake, netD=None):
        outs_fake = netD.forward(fake)
        loss_G = 0
        for out_a in outs_fake:
            outputs_fake = nn.functional.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
            loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
        return loss_G
    
    def backward_G_alone(self):
        # Ladv for generator
        loss_G_GAN2_A = self.backward_G_GAN(self.fake_A_random, self.d_a2)
        loss_G_GAN2_B = self.backward_G_GAN(self.fake_B_random, self.d_b2)
        if not self.no_ms:
            loss_G_GAN2_A2 = self.backward_G_GAN(self.fake_A_random2, self.d_a2)
            loss_G_GAN2_B2 = self.backward_G_GAN(self.fake_B_random2, self.d_b2)
        
        # mode seeking loss for A-->B and B-->A
        if not self.no_ms:
            lz_AB = torch.mean(torch.abs(self.fake_B_random2 - self.fake_B_random)) / torch.mean(
                torch.abs(self.z_random2 - self.z_random)
                )
            lz_BA = torch.mean(torch.abs(self.fake_A_random2 - self.fake_A_random)) / torch.mean(
                torch.abs(self.z_random2 - self.z_random)
                )
            eps        = 1 * 1e-5
            loss_lz_AB = 1 / (lz_AB + eps)
            loss_lz_BA = 1 / (lz_BA + eps)
        # # latent regression loss
        # if self.concat:
        #   loss_z_L1_a = torch.mean(torch.abs(self.mu2_a - self.z_random)) * 10
        #   loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - self.z_random)) * 10
        # else:
        #   loss_z_L1_a = torch.mean(torch.abs(self.z_attr_random_a - self.z_random)) * 10
        #   loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - self.z_random)) * 10
        
        loss_z_L1 = loss_G_GAN2_A + loss_G_GAN2_B
        if not self.no_ms:
            loss_z_L1 += (loss_G_GAN2_A2 + loss_G_GAN2_B2)
            loss_z_L1 += (loss_lz_AB + loss_lz_BA)
        
        # perceptual loss
        # if self.vgg > 0:
        #   loss_vgg_a = self.vgg_loss.compute_vgg_loss(self.vgg_net, self.fake_A_encoded, self.input_A)
        #   loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg_net, self.fake_B_encoded, self.input_B)
        #   loss_z_L1 += loss_vgg_a + loss_vgg_b
        
        loss_z_L1.backward()
        if not self.no_ms:
            self.gan2_loss_a = loss_G_GAN2_A.item() + loss_G_GAN2_A2.item()
            self.gan2_loss_b = loss_G_GAN2_B.item() + loss_G_GAN2_B2.item()
            self.lz_AB = loss_lz_AB.item()
            self.lz_BA = loss_lz_BA.item()
        else:
            self.gan2_loss_a = loss_G_GAN2_A.item()
            self.gan2_loss_b = loss_G_GAN2_B.item()
        # if self.vgg > 0:
        #   self.loss_vgg_a = loss_vgg_a.item()
        #   self.loss_vgg_b = loss_vgg_b.item()
    
    def update_lr(self):
        self.d_a_sch.step()
        self.d_b_sch.step()
        self.d_a2_sch.step()
        self.d_b2_sch.step()
        self.d_content_sch.step()
        self.e_content_sch.step()
        self.e_attribute_sch.step()
        self.g_sch.step()
    
    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss
    
    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # weight
        if train:
            self.d_a.load_state_dict(checkpoint['disA'])
            self.d_a2.load_state_dict(checkpoint['disA2'])
            self.d_b.load_state_dict(checkpoint['disB'])
            self.d_b2.load_state_dict(checkpoint['disB2'])
            self.d_content.load_state_dict(checkpoint['disContent'])
        self.e_content.load_state_dict(checkpoint['enc_c'])
        self.e_attribute.load_state_dict(checkpoint['enc_a'])
        self.g.load_state_dict(checkpoint['gen'])
        # optimizer
        if train:
            self.d_a_optim.load_state_dict(checkpoint['disA_opt'])
            self.d_a2_optim.load_state_dict(checkpoint['disA2_opt'])
            self.d_b_optim.load_state_dict(checkpoint['disB_opt'])
            self.d_b2_optim.load_state_dict(checkpoint['disB2_opt'])
            self.d_content_optim.load_state_dict(checkpoint['disContent_opt'])
            self.e_content_optim.load_state_dict(checkpoint['enc_c_opt'])
            self.e_attribute_optim.load_state_dict(checkpoint['enc_a_opt'])
            self.g_optim.load_state_dict(checkpoint['gen_opt'])
        return checkpoint['ep'], checkpoint['total_it']
    
    def save(self, filename, ep, total_it):
        state = {
            'disA'          : self.d_a.state_dict(),
            'disA2'         : self.d_a2.state_dict(),
            'disB'          : self.d_b.state_dict(),
            'disB2'         : self.d_b2.state_dict(),
            'disContent'    : self.d_content.state_dict(),
            'enc_c'         : self.e_content.state_dict(),
            'enc_a'         : self.e_attribute.state_dict(),
            'gen'           : self.g.state_dict(),
            'disA_opt'      : self.d_a_optim.state_dict(),
            'disA2_opt'     : self.d_a2_optim.state_dict(),
            'disB_opt'      : self.d_b_optim.state_dict(),
            'disB2_opt'     : self.d_b2_optim.state_dict(),
            'disContent_opt': self.d_content_optim.state_dict(),
            'enc_c_opt'     : self.e_content_optim.state_dict(),
            'enc_a_opt'     : self.e_attribute_optim.state_dict(),
            'gen_opt'       : self.g_optim.state_dict(),
            'ep'            : ep,
            'total_it'      : total_it
        }
        torch.save(state, filename)
        return
    
    def assemble_outputs(self):
        images_a  = self.normalize_image(self.real_A_encoded).detach()
        images_b  = self.normalize_image(self.real_B_encoded).detach()
        images_a1 = self.normalize_image(self.fake_A_encoded).detach()
        # images_a2 = self.normalize_image(self.fake_A_random).detach()
        # images_a3 = self.normalize_image(self.fake_A_recon).detach()
        images_a4 = self.normalize_image(self.fake_AA_encoded).detach()
        images_b1 = self.normalize_image(self.fake_B_encoded).detach()
        # images_b2 = self.normalize_image(self.fake_B_random).detach()
        # images_b3 = self.normalize_image(self.fake_B_recon).detach()
        images_b4 = self.normalize_image(self.fake_BB_encoded).detach()
        row1      = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_a4[0:1, ::]), 3)
        row2      = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_b4[0:1, ::]), 3)
        return torch.cat((row1, row2), 2)
    
    def normalize_image(self, x):
        return x[:, 0:3, :, :]

# endregion
