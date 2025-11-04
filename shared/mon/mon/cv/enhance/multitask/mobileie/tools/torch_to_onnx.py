import torch
import torch.nn as nn


class MobileIENetS(nn.Module):
    
    def __init__(self, channels):
        super(MobileIENetS, self).__init__()
        self.head = FST(
            nn.Sequential(
                nn.Conv2d(3, channels, 5, 1, 2),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1)
            ),
            channels
        )
        self.body = FST(
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
        
    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)   
        x3 = self.att1(max_out)
        x4 = torch.mul(x3, x2) * x1
        return self.tail(x4)


class FST(nn.Module):
    
    def __init__(self, block1, channels):
        super(FST, self).__init__()
        self.block1 = block1
        self.weight1 = nn.Parameter(torch.randn(1)) 
        self.weight2 = nn.Parameter(torch.randn(1)) 
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))
        
    def forward(self, x):
        x1 = self.block1(x)
        weighted_block1 = self.weight1 * x1
        weighted_block2 = self.weight2 * x1
        return weighted_block1 * weighted_block2 + self.bias


def export_onnx(pretrained_model_path):
    model = MobileIENetS(12)  
    
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint)
    model.eval()  
    
    dummy_input = torch.randn(1, 3, 400, 600)
    
    torch.onnx.export(
        model,                          
        dummy_input,                   
        "LLE.onnx",        
        opset_version=12,              
        export_params=True,            
        do_constant_folding=True,       
        input_names=['input'],          
        output_names=['output'],     
        dynamic_axes=None
    )
    print("ONNX Success.")


if __name__ == "__main__":
    pretrained_model_path = r'./pretrain/lolv1_best_slim.pkl'
    export_onnx(pretrained_model_path)
