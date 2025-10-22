import torch
import torch.nn as nn

from mon.core import nn


class CharbonnierLoss(nn.BaseLoss):
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, inp, target):
        return ((nn.functional.mse_loss(inp, target, reduction="none") + self.eps2) ** 0.5).mean()
    
    
#####################################################################################################
class OutlierAwareLoss(nn.BaseLoss):
    
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta  = input - target
        var    = delta.std((2, 3), keepdims=True) / (2 ** 0.5)
        avg    = delta.mean((2, 3), True)
        weight = torch.tanh((delta - avg).abs() / (var + 1e-6)).detach()
        loss   = (delta.abs() * weight)
        loss   = self.reduce(loss=loss)
        return loss
    
    
#####################################################################################################
class WarmupLoss(nn.BaseLoss):
    
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.loss_cb = nn.CharbonnierLoss(1e-8, reduction=reduction)
        self.loss_cs = nn.CosineSimilarity(reduction=reduction)

    def forward(self, input, target, warmup1, warmup2):
        loss = (self.loss_cb(warmup2, input) +
                (self.loss_cb(warmup1, target)
                 + (1 - self.loss_cs(warmup1.clip(0, 1), target))))
        loss = self.reduce(loss=loss)
        return loss 


class LLELoss(nn.BaseLoss):
    
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.loss_cs = nn.CosineSimilarity(reduction=reduction)
        self.loss_oa = OutlierAwareLoss(reduction=reduction)
        self.psnr    = nn.PSNRLoss(reduction=reduction)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = ((self.loss_oa(input, target)
                + (1 - self.loss_cs(input.clip(0, 1), target)))
                + 2 * self.psnr(input, target))
        loss = self.reduce(loss=loss)
        return loss
        
        
class ISPLoss(nn.BaseLoss):
    
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.loss_cs = nn.CosineSimilarity(reduction=reduction)
        self.loss_oa = OutlierAwareLoss(reduction=reduction)
        self.psnr    = nn.PSNRLoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = ((self.loss_oa(input, target)
                + (1 - self.loss_cs(input.clip(0, 1), target)))
                + 2 * self.psnr(input, target))
        loss = self.reduce(loss=loss)
        return loss


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight: float = 1.0, reduction: str = "mean", toY: bool = False):
        super().__init__()
        assert reduction == "mean"
        self.loss_weight = loss_weight
        self.toY   = toY
        self.coef  = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef  = self.coef.to(pred.device)
                self.first = False

            pred   = (pred   * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            pred   =   pred / 255.0
            target = target / 255.0
            pass
        assert len(pred.size()) == 4
        imdff = pred - target
        rmse  = ((imdff ** 2).mean(dim=(1, 2, 3)) + 1e-8).sqrt()
        loss  = 20 * torch.log10(1 / rmse).mean()
        loss  = (50.0 - loss) / 100.0
        return loss


def import_loss(training_task):
    if training_task == 'isp':
        return ISPLoss()
    elif training_task == 'lle':
        return LLELoss()
    elif training_task == 'warmup':
        return WarmupLoss()
    else:
        raise ValueError('unknown training task, please choose from [isp, lle, warmup].')
