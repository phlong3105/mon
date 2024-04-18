import torch
import torch.nn as nn
import mon
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torchvision import transforms
from torchvision.models import vgg


class Loss(_Loss):
    
    def __init__(
        self,
        alpha1: float = 0.35,
        alpha2: float = 0.10,
        alpha3: float = 0.25,
        alpha4: float = 0.30,
    ):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4
        
        self.msssim     = mon.CustomMSSSIM(data_range=1.0)
        self.ssim       = mon.CustomSSIM(data_range=1.0, non_negative_ssim=True)
        self.perceptual = PerceptualLoss()
        self.tvloss     = TVLoss()

    def region(self, pred, label):
        gray     = 0.30 * label[:, 0, :, :] + 0.59 * label[:, 1, :, :] + 0.11 * label[:, 2, :, :]
        gray     = gray.view(-1)
        value    = -torch.topk(-gray, int(gray.shape[0] * 0.4))[0][0]
        weight   = 1 * (label > value) + 4 * (label <= value)
        abs_diff = torch.abs(pred - label)
        return torch.mean(weight * abs_diff)
    
    def forward(self, x, y):
        str_loss    = 2 - self.msssim(x, y) - self.ssim(x, y)
        vgg_loss    = self.perceptual(x, y)
        region_loss = self.region(x, y)
        tv_loss     = self.tvloss(x)
        loss        = (
              self.alpha1 * str_loss
            + self.alpha2 * tv_loss
            + self.alpha3 * region_loss
            + self.alpha4 * vgg_loss
        )
        return loss


class PerceptualLoss(_Loss):
    
    def __init__(self,):
        super().__init__()
        self.vgg = vgg.vgg19(weights=vgg.VGG19_Weights.IMAGENET1K_V1).features
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.vgg.eval()

    def vgg_forward(self, x):
        output = []
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name == "26":
                return x
    
    def preprocess(self, tensor):
        trsfrm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        res    = trsfrm(tensor)
        return res       

    def forward(self, output, label):
        output = self.preprocess(output)
        label  = self.preprocess(label)
        feat_a = self.vgg_forward(output)
        feat_b = self.vgg_forward(label)
        return F.l1_loss(feat_a, feat_b)


class TVLoss(nn.Module):
    
    def __init__(self, TVLoss_weight=1):
        super().__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x     = x.size()[2]
        w_x     = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv    = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv    = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
