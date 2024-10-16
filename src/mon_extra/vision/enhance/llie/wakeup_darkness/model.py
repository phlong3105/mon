import torch
import torch.nn as nn
from loss import LossFunction
from fuse_block import TransformerBlock_1


class GatedResidualBlock(nn.Module):
    
    def __init__(self, channels):
        super(GatedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.Mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        gate = self.gate(x)
        x = gate * x + (1 - gate) * residual  # Sigmoid门控的残差连接
        return x


class EnhanceNetwork(nn.Module):
    
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.Mish()
        )
        
        self.fusion = TransformerBlock_1(channels, channels, channels, num_heads=3)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.Mish()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.block = GatedResidualBlock(channels)
        
    def forward(self, input, sem, depth):
        fea = self.in_conv(input)
        
        fea = fea + self.fusion(fea, sem,depth)
        for conv in self.blocks:
            fea = fea + conv(fea)
            fea = self.block(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)

        return illu


class ColorCorrectionModule(nn.Module):
    
    def __init__(self, correction_matrix=None, use_lut=False, lut_size=33):
        super(ColorCorrectionModule, self).__init__()
        
        # 如果提供了色彩校正矩阵，则直接使用
        if correction_matrix is not None:
            assert correction_matrix.shape == (3, 3), "Correction matrix should be a 3x3 matrix"
            self.correction_matrix = nn.Parameter(torch.tensor(correction_matrix).float(), requires_grad=True)
            self.use_lut = False
        else:
            self.correction_matrix = None
            self.use_lut = use_lut

            # 初始化查找表（如果use_lut为True）
            if self.use_lut:
                self.lut = nn.Parameter(torch.randn(lut_size, 3).float(), requires_grad=True)

    def forward(self, input):
        if self.correction_matrix is not None:
            # 使用色彩校正矩阵的方式
            corrected_input = input @ self.correction_matrix
        elif self.use_lut:
            # 使用查找表的方式
            normalized_input = input / 255.0
            indices = torch.floor(normalized_input * (self.lut.size(0) - 1)).long()
            corrected_input = F.embedding(indices.unsqueeze(0).unsqueeze(0), self.lut).squeeze()
        else:
            # 若没有提供校正方式，默认输出原输入
            corrected_input = input

        return corrected_input  

    
class Network_woCalibrate(nn.Module):

    def __init__(self, use_lut=False):
        super().__init__()
        self.enhance = EnhanceNetwork(layers=2, channels=3)
        self.color_correction = ColorCorrectionModule(use_lut=use_lut)
        self._criterion = LossFunction()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input, sem,depth):
        i = self.enhance(input, sem,depth)
        r = input / i
        r = torch.clamp(r, 0, 1)
        
        # 应用颜色校正
        corrected_r = self.color_correction(r)
        
        return i, corrected_r, depth

    def _loss(self, input, sem, depth):
        i, r,d = self(input, sem,depth)
        loss_semantic = self._criterion(input, i)
        loss_depth =self._criterion(input, d)
        loss=loss_semantic+loss_depth
        return loss
    
    
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# net = Network_woCalibrate()
# total_params = count_parameters(net)
# print(f"Total trainable parameters: {total_params}")

# import torch
# from thop import profile

# input_tensor = torch.randn(1, 3, 480, 600)  # Replace height and width with your input size
# model = Network_woCalibrate()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # Move the model and input tensor to the same device (e.g., CPU or GPU)
# model.to(device)
# input_tensor = input_tensor.to(device)

# flops, params = profile(model, inputs=(input_tensor,input_tensor,input_tensor))
# print(f"FLOPs: {flops / 1e9} G FLOPs")  # Convert FLOPs to Giga FLOPs
