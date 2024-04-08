import torch
from torch import nn


class UNetConvBlock(nn.Module):
    
    def __init__(self, in_size, out_size, relu_slope=0.2):     
        super(UNetConvBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_size, in_size, kernel_size=3, padding=1, bias=True)
        self.norm1  = nn.InstanceNorm2d(in_size, affine=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
    
        self.conv_2 = nn.Conv2d(in_size*2, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_3 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_3 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_1_1_1 = nn.Conv2d(in_size, in_size, 1, 1, 0)
        self.conv_1_1   = nn.Conv2d(in_size * 2, out_size, 1, 1, 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x   = input
        out = self.conv_1(x)
        out = self.norm1(out)
        out = self.relu_1(out)
        
        out1 = self.conv_1_1_1(x)
        out2 = torch.cat([out, out1], dim = 1)
        
        out = self.conv_2(out2)
        out = self.relu_2(out)
        out = self.conv_3(out)
        out = self.relu_3(out)

        res  = self.conv_1_1(out2)
        out += res

        return out


class NestedUNet(nn.Module):
    
    def __init__(self, input_channels=3, out_channels=3):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = UNetConvBlock(input_channels, nb_filter[0])
        self.conv1_0 = UNetConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = UNetConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = UNetConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = UNetConvBlock(nb_filter[3], nb_filter[4])
        
        self.conv0_1 = UNetConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = UNetConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = UNetConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = UNetConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])
        
        self.conv0_2 = UNetConvBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = UNetConvBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = UNetConvBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])
        
        self.conv0_3 = UNetConvBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = UNetConvBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])
        
        self.conv0_4 = UNetConvBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
    
        self.final   = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
            
        return torch.clamp(output, 0, 1)
