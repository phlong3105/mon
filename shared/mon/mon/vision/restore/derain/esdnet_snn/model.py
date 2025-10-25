import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import functional, layer, neuron


class Feature_Refinement_Block(nn.Module):
    
    def __init__(self, channel: int, reduction: int):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.Conv2d(channel, channel // 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.ca(x)
        t = self.sa(x)
        s = torch.mul((1 - t), a) + torch.mul(t, x)
        return s


class OverlapPatchEmbed(nn.Module):
    
    def __init__(
        self,
        in_channels   : int  = 3,
        embed_dim     : int  = 32,
        spike_mode    : str  = "lif",
        layernorm_type: str  = "WithBias",
        bias          : bool = False
    ):
        super().__init__()
        functional.set_step_mode(self, step_mode="m")
        self.proj = layer.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


class Spiking_Residual_Block(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        v_th  = 0.15
        alpha = 1 / (2 ** 0.5)
        functional.set_step_mode(self, step_mode="m")
        self.residual = nn.Sequential(
            neuron.LIFNode(v_threshold=v_th, backend="cupy", step_mode="m", decay_input=False),
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"),
            layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True),
            
            neuron.LIFNode(v_threshold=v_th, backend="cupy", step_mode="m", decay_input=False),
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"),
            layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th * 0.2, affine=True),
        )
        self.shortcut = nn.Sequential(
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"),
            layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True),
        )
        self.attn = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = torch.clone(x)
        out      = self.residual(x) + self.shortcut(x)
        out      = self.attn(out) + shortcut
        return out


class DownSampling(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        v_th  = 0.15
        alpha = 1 / (2 ** 0.5)
        functional.set_step_mode(self, step_mode="m")
        self.maxpool_conv = nn.Sequential(
            neuron.LIFNode(v_threshold=v_th, backend="cupy", step_mode="m", decay_input=False),
            layer.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, step_mode="m", bias=False),
            layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th, num_features=dim * 2, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpSampling(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        v_th  = 0.15
        alpha = 1 / (2 ** 0.5)
        self.scale_factor = 2
        self.up = nn.Sequential(
            neuron.LIFNode(v_threshold=v_th, backend="cupy", step_mode="m", decay_input=False),
            layer.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, step_mode="m", bias=False),
            layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th, num_features=dim // 2, affine=True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        temp = torch.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3] * self.scale_factor, input.shape[4] * self.scale_factor)).cuda()
        output = []
        for i in range(input.shape[0]):
            temp[i] = F.interpolate(input[i], scale_factor=self.scale_factor, mode="bilinear")
            output.append(temp[i])
        out = torch.stack(output, dim=0)
        return self.up(out)


class ESDNet(nn.Module):
    
    def __init__(
        self,
        in_channels  : int       = 3,
        out_channels : int       = 3,
        dim          : int       = 24,
        en_num_blocks: list[int] = [4, 4, 6, 6],
        de_num_blocks: list[int] = [4, 4, 6, 6],
        bias         : bool      = False,
        T            : int       = 4
    ):
        super().__init__()
        v_th  = 0.15
        alpha = 1 / (2 ** 0.5)
        functional.set_backend(self,   backend="cupy")
        functional.set_step_mode(self, step_mode="m")

        self.T = T
        self.patch_embed    = OverlapPatchEmbed(in_channels=in_channels, embed_dim=dim)
        self.encoder_level1 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 1)) for i in range(en_num_blocks[0])
        ])
        
        self.down1_2        = DownSampling(dim)  # From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 1)) for i in range(en_num_blocks[1])
        ])
        
        self.down2_3        = DownSampling(int(dim * 2 ** 1))  # From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 2)) for i in range(en_num_blocks[2])
        ])

        self.decoder_level3 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 2)) for i in range(de_num_blocks[2])
        ])
        
        self.up3_2 = UpSampling(int(dim * 2 ** 2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Sequential(
            neuron.LIFNode(v_threshold=v_th, backend="cupy", step_mode="m", decay_input=False),
            layer.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias, step_mode="m"),
            layer.ThresholdDependentBatchNorm2d(num_features=int(dim * 2 ** 1), alpha=alpha, v_th=v_th),
        )
        self.decoder_level2 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 1)) for i in range(de_num_blocks[1])
        ])
        
        self.up2_1          = UpSampling(int(dim * 2 ** 1))  # From Level 2 to Level 1 (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 1)) for i in range(de_num_blocks[0])
        ])

        self.refinement = Feature_Refinement_Block(channel=int(dim * 2 ** 1), reduction=8)
        self.output     = nn.Sequential(nn.Conv2d(in_channels=int(dim * 2 ** 1), out_channels=out_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        short = input.clone()
        # Repeat Feature
        if len(input.shape) < 5:
            input = (input.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        
        inp_enc_level1 = self.patch_embed(input)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        
        out_dec_level3 = self.decoder_level3(out_enc_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=2)
        
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=2)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # Image Reconstruction
        out_dec_level1 = self.refinement(out_dec_level1.mean(0))
        out_dec_level1 = (self.output(out_dec_level1)) + short
        return out_dec_level1


model = ESDNet(dim=48, en_num_blocks=[4, 4, 8, 8], de_num_blocks=[2, 2, 2, 2], T=4)

# Debug
# x = torch.rand(1, 3, 256, 256).cuda()
# functional.set_step_mode(model, step_mode='m')
# functional.set_backend(model, backend='cupy')
# # print(model(x).shape)
# flops, params = profile(model, inputs=(x,))
# print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
# print('Params = ' + str(params / 1000 ** 2) + 'M')
