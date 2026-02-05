import torch
import torch.nn as nn

from .utils import DropBlock, FST, FSTS, MBRConv1, MBRConv3, MBRConv5


# Training Model
class MobileIELLENet(nn.Module):
    
    def __init__(self, channels: int, rep_scale: int = 4):
        super().__init__()
        self.channels = channels
        self.head     = FST(
            nn.Sequential(
                MBRConv5(3, channels, rep_scale=rep_scale),
                nn.PReLU(channels),
                MBRConv3(channels, channels, rep_scale=rep_scale)
            ),
            channels
        )
        self.body = FST(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            channels
        )
        self.att = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(channels, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        self.att1 = nn.Sequential(
            MBRConv1(1, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        self.tail      = MBRConv3(channels, 3, rep_scale=rep_scale)
        self.tail_warm = MBRConv3(channels, 3, rep_scale=rep_scale)
        self.drop      = DropBlock(3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.head(x)
        x1 = self.body(x0)      
        x2 = self.att(x1)
        max_out, _ = torch.max(x2 * x1 , dim=1, keepdim=True)   
        x3 = self.att1(max_out)
        x4 = torch.mul(x2, x3) * x1
        return self.tail(x4)

    def forward_warm(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(x)
        x = self.head(x)
        x = self.body(x)
        return self.tail(x), self.tail_warm(x)

    def slim(self):
        net_slim    = MobileIELLENetS(self.channels)
        weight_slim = net_slim.state_dict()
        for name, mod in self.named_modules():
            if isinstance(mod, MBRConv3) or isinstance(mod, MBRConv5) or isinstance(mod, MBRConv1):
               if "%s.weight" % name in weight_slim:
                    w, b = mod.slim()
                    weight_slim["%s.weight" % name] = w 
                    weight_slim["%s.bias"   % name] = b
            elif isinstance(mod, FST):
                weight_slim["%s.bias"    % name] = mod.bias
                weight_slim["%s.weight1" % name] = mod.weight1
                weight_slim["%s.weight2" % name] = mod.weight2
            elif isinstance(mod, nn.PReLU):
                weight_slim["%s.weight"  % name] = mod.weight
        net_slim.load_state_dict(weight_slim)
        return net_slim


# Inference Model
class MobileIELLENetS(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.head = FSTS(
            nn.Sequential(
                nn.Conv2d(3, channels, 5, 1, 2),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1)
            ),
            channels
        )
        self.body = FSTS(
            nn.Conv2d(channels, channels, 3, 1, 1),
            channels
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.att1 = nn.Sequential( 
            nn.Conv2d(1, channels, 1, 1),
            nn.Sigmoid()
        )
        self.tail = nn.Conv2d(channels, 3, 3, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)   
        x3 = self.att1(max_out)
        x4 = torch.mul(x3, x2) * x1
        return self.tail(x4)
