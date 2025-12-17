import torch
import torch.nn as nn
import torch.nn.functional as F


class MBRConv5(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, rep_scale: int = 4):
        super().__init__()
        self.in_channels     = in_channels
        self.out_channels    = out_channels
        self.conv            = nn.Conv2d(in_channels, out_channels * rep_scale, 5, 1, 2)
        self.conv_bn         = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv1           = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn        = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv2           = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
        self.conv2_bn        = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_crossh    = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_crossv    = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_out       = nn.Conv2d(out_channels * rep_scale * 10, out_channels, 1)
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(inp)
        x2 = self.conv1(inp)
        x3 = self.conv2(inp)
        x4 = self.conv_crossh(inp)
        x5 = self.conv_crossv(inp)
        x = torch.cat(
            [x1, x2, x3, x4, x5,
             self.conv_bn(x1),
             self.conv1_bn(x2),
             self.conv2_bn(x3),
             self.conv_crossh_bn(x4),
             self.conv_crossv_bn(x5)],
            1
        )
        out = self.conv_out(x)
        return out 

    def slim(self):
        conv_weight  = self.conv.weight
        conv_bias    = self.conv.bias

        conv1_weight = self.conv1.weight
        conv1_bias   = self.conv1.bias
        conv1_weight = nn.functional.pad(conv1_weight, (2, 2, 2, 2))

        conv2_weight = self.conv2.weight
        conv2_weight = nn.functional.pad(conv2_weight, (1, 1, 1, 1))
        conv2_bias   = self.conv2.bias

        conv_crossv_weight = self.conv_crossv.weight
        conv_crossv_weight = nn.functional.pad(conv_crossv_weight, (1, 1, 2, 2))
        conv_crossv_bias   = self.conv_crossv.bias

        conv_crossh_weight = self.conv_crossh.weight
        conv_crossh_weight = nn.functional.pad(conv_crossh_weight, (2, 2, 1, 1))
        conv_crossh_bias   = self.conv_crossh.bias

        conv1_bn_weight = self.conv1.weight
        conv1_bn_weight = nn.functional.pad(conv1_bn_weight, (2, 2, 2, 2))

        conv2_bn_weight = self.conv2.weight
        conv2_bn_weight = nn.functional.pad(conv2_bn_weight, (1, 1, 1, 1))

        conv_crossv_bn_weight = self.conv_crossv.weight
        conv_crossv_bn_weight = nn.functional.pad(conv_crossv_bn_weight, (1, 1, 2, 2))

        conv_crossh_bn_weight = self.conv_crossh.weight
        conv_crossh_bn_weight = nn.functional.pad(conv_crossh_bn_weight, (2, 2, 1, 1))

        bn = self.conv_bn[0]
        k  = 1 / (bn.running_var + bn.eps) ** .5
        b  = - bn.running_mean / (bn.running_var + bn.eps) ** .5

        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias   = self.conv.bias * k + b
        conv_bn_bias   = conv_bn_bias * bn.weight + bn.bias

        bn = self.conv1_bn[0]
        k  = 1 / (bn.running_var + bn.eps) ** .5
        b  = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv1_bn_weight = conv1_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_bias   = self.conv1.bias * k + b
        conv1_bn_bias   = conv1_bn_bias * bn.weight + bn.bias

        bn = self.conv2_bn[0]
        k  = 1 / (bn.running_var + bn.eps) ** .5
        b  = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv2_bn_weight = conv2_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv2_bn_weight = conv2_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv2_bn_bias   = self.conv2.bias * k + b
        conv2_bn_bias   = conv2_bn_bias * bn.weight + bn.bias

        bn = self.conv_crossv_bn[0]
        k  = 1 / (bn.running_var + bn.eps) ** .5
        b  = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_crossv_bn_weight = conv_crossv_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_bias   = self.conv_crossv.bias * k + b
        conv_crossv_bn_bias   = conv_crossv_bn_bias * bn.weight + bn.bias

        bn = self.conv_crossh_bn[0]
        k  = 1 / (bn.running_var + bn.eps) ** .5
        b  = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_crossh_bn_weight = conv_crossh_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_bias   = self.conv_crossh.bias * k + b
        conv_crossh_bn_bias   = conv_crossh_bn_bias * bn.weight + bn.bias

        weight = torch.cat(
            [conv_weight, conv1_weight, conv2_weight,
             conv_crossh_weight, conv_crossv_weight,
             conv_bn_weight, conv1_bn_weight, conv2_bn_weight,
             conv_crossh_bn_weight, conv_crossv_bn_weight],
            0
        )
        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])
        bias_  = torch.cat(
            [conv_bias, conv1_bias, conv2_bias,
             conv_crossh_bias, conv_crossv_bias,
             conv_bn_bias, conv1_bn_bias, conv2_bn_bias,
             conv_crossh_bn_bias, conv_crossv_bn_bias],
            0
        )
        bias = torch.matmul(weight_compress, bias_)
        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias
        return weight, bias


##############################################################################################################
class MBRConv3(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, rep_scale: int = 4):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.rep_scale    = rep_scale
        
        self.conv           = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
        self.conv_bn        = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv1          = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn       = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_crossh    = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_crossv    = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_out       = nn.Conv2d(out_channels * rep_scale * 8, out_channels, 1)

    def forward(self, inp):    
        x0 = self.conv(inp)
        x1 = self.conv1(inp)
        x2 = self.conv_crossh(inp)
        x3 = self.conv_crossv(inp)
        x  = torch.cat(
        [x0, x1, x2, x3,
             self.conv_bn(x0),
             self.conv1_bn(x1),
             self.conv_crossh_bn(x2),
             self.conv_crossv_bn(x3)],
            1
        )    
        out = self.conv_out(x)
        return out

    def slim(self):
        conv_weight = self.conv.weight
        conv_bias   = self.conv.bias

        conv1_weight = self.conv1.weight
        conv1_bias   = self.conv1.bias
        conv1_weight = F.pad(conv1_weight, (1, 1, 1, 1))

        conv_crossh_weight = self.conv_crossh.weight
        conv_crossh_bias   = self.conv_crossh.bias
        conv_crossh_weight = F.pad(conv_crossh_weight, (1, 1, 0, 0))

        conv_crossv_weight = self.conv_crossv.weight
        conv_crossv_bias   = self.conv_crossv.bias
        conv_crossv_weight = F.pad(conv_crossv_weight, (0, 0, 1, 1))

        # conv_bn
        bn = self.conv_bn[0]
        k  = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias   = self.conv.bias * k + (-bn.running_mean * k)
        conv_bn_bias   = conv_bn_bias * bn.weight + bn.bias

        # conv1_bn
        bn = self.conv1_bn[0]
        k  = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv1_bn_weight = self.conv1.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = F.pad(conv1_bn_weight, (1, 1, 1, 1))
        conv1_bn_bias   = self.conv1.bias * k + (-bn.running_mean * k)
        conv1_bn_bias   = conv1_bn_bias * bn.weight + bn.bias

        # conv_crossh_bn
        bn = self.conv_crossh_bn[0]
        k  = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_crossh_bn_weight = self.conv_crossh.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = F.pad(conv_crossh_bn_weight, (1, 1, 0, 0))
        conv_crossh_bn_bias   = self.conv_crossh.bias * k + (-bn.running_mean * k)
        conv_crossh_bn_bias   = conv_crossh_bn_bias * bn.weight + bn.bias

        # conv_crossv_bn
        bn = self.conv_crossv_bn[0]
        k  = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_crossv_bn_weight = self.conv_crossv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = F.pad(conv_crossv_bn_weight, (0, 0, 1, 1))
        conv_crossv_bn_bias   = self.conv_crossv.bias * k + (-bn.running_mean * k)
        conv_crossv_bn_bias   = conv_crossv_bn_bias * bn.weight + bn.bias

        weight = torch.cat([
            conv_weight,
            conv1_weight,
            conv_crossh_weight,
            conv_crossv_weight,
            conv_bn_weight,
            conv1_bn_weight,
            conv_crossh_bn_weight,
            conv_crossv_bn_weight
        ], dim=0)

        bias = torch.cat([
            conv_bias,
            conv1_bias,
            conv_crossh_bias,
            conv_crossv_bias,
            conv_bn_bias,
            conv1_bn_bias,
            conv_crossh_bn_bias,
            conv_crossv_bn_bias
        ], dim=0)

        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.view(weight.size(0), -1))
        weight = weight.view(self.conv_out.out_channels, self.in_channels, 3, 3)

        bias = torch.matmul(weight_compress, bias.unsqueeze(-1)).squeeze(-1)
        if self.conv_out.bias is not None:
            bias += self.conv_out.bias

        return weight, bias
    
    
######################################################################################################
class MBRConv1(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, rep_scale: int = 4):
        super().__init__()
        
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.rep_scale    = rep_scale
        
        self.conv     = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv_bn  = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x0  = self.conv(inp)
        x   = torch.cat([x0, self.conv_bn(x0)], 1)
        out = self.conv_out(x)
        return out 

    def slim(self):
        conv_weight = self.conv.weight
        conv_bias   = self.conv.bias

        bn = self.conv_bn[0]
        k  = 1 / (bn.running_var + bn.eps) ** .5
        b  = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias   = self.conv.bias * k + b
        conv_bn_bias   = conv_bn_bias * bn.weight + bn.bias

        weight          = torch.cat([conv_weight, conv_bn_weight], 0)
        weight_compress = self.conv_out.weight.squeeze()
        weight          = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])

        bias = torch.cat([conv_bias, conv_bn_bias], 0)
        bias = torch.matmul(weight_compress, bias)

        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias
        return weight, bias
    
    
class FST(nn.Module):
    
    def __init__(self, block1: nn.Module, channels: int):
        super().__init__()
        self.block1  = block1
        self.weight1 = nn.Parameter(torch.randn(1)) 
        self.weight2 = nn.Parameter(torch.randn(1)) 
        self.bias    = nn.Parameter(torch.randn((1, channels, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)
        weighted_block1 = self.weight1 * x1
        weighted_block2 = self.weight2 * x1
        return weighted_block1 * weighted_block2 + self.bias
        
        
class FSTS(nn.Module):
    
    def __init__(self, block1: nn.Module, channels: int):
        super().__init__()
        self.block1  = block1
        self.weight1 = nn.Parameter(torch.randn(1)) 
        self.weight2 = nn.Parameter(torch.randn(1)) 
        self.bias    = nn.Parameter(torch.randn((1, channels, 1, 1)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)
        weighted_block1 = self.weight1 * x1
        weighted_block2 = self.weight2 * x1
        return weighted_block1 * weighted_block2 + self.bias
    
    
##################################################################################
class DropBlock(nn.Module):
    
    def __init__(self, block_size: int, p: float = 0.5):
        super().__init__()
        self.block_size = block_size
        self.p          = p / block_size / block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = 1 - (torch.rand_like(x[:, :1]) >= self.p).float()
        mask = nn.functional.max_pool2d(mask, self.block_size, 1, self.block_size // 2)
        return x * (1 - mask)
