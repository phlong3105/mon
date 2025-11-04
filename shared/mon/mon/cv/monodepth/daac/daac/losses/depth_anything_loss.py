import torch
from einops import rearrange
from torch import nn


class AffineInvariantLossV2(nn.Module):

    def __init__(self):
        super(AffineInvariantLossV2, self).__init__()

    def forward(
        self,
        predicted_depth,
        target_depth,
        mask,
        eps          = 1e-6,
        need_inverse = True,
        need_norm    = True,
        uncertainty  = None
    ):
        target_disparity_list = []
        pred_disparity_list   = []
        all_empty = True
        for i in range(mask.shape[0]):
            mask_i = mask[i]
            if uncertainty is not None:
                uncertainty_i = uncertainty[i]
            if torch.sum(mask_i) == 0:
                continue
            all_empty = False

            # target
            if need_inverse:
                target_disparity_i = 1 / (target_depth[i] + eps)
            else:
                target_disparity_i = target_depth[i].clone()
            if need_norm:
                target_disparity_i[mask_i] = (target_disparity_i[mask_i] - target_disparity_i[mask_i].min()) / (target_disparity_i[mask_i].max()-target_disparity_i[mask_i].min()+eps)
            else:
                target_disparity_i[mask_i] = target_disparity_i[mask_i]

            target_median_i = torch.mean(target_disparity_i[mask_i])
            target_scale_i  = torch.mean(torch.abs(target_disparity_i[mask_i] - target_median_i))
            scaled_target_disparity_i = (target_disparity_i - target_median_i) / (target_scale_i + eps)
            scaled_target_disparity_i = rearrange(scaled_target_disparity_i, 'h w -> 1 h w')
            # w/o norm
            if uncertainty is not None:
                scaled_target_disparity_i = scaled_target_disparity_i * mask_i * uncertainty_i
            else:
                scaled_target_disparity_i = scaled_target_disparity_i * mask_i
            target_disparity_list.append(scaled_target_disparity_i)
            
            # predict
            predicted_disparity_i        = predicted_depth[i]
            predicted_disparity_i        = (predicted_disparity_i - predicted_disparity_i.min()) / (predicted_disparity_i.max()-predicted_disparity_i.min()+eps)
            predicted_median_i           = torch.mean(predicted_disparity_i[mask_i])
            predicted_scale_i            = torch.mean(torch.abs(predicted_disparity_i[mask_i] - predicted_median_i))
            scaled_predicted_disparity_i = (predicted_disparity_i - predicted_median_i) / (predicted_scale_i + eps)
            scaled_predicted_disparity_i = rearrange(scaled_predicted_disparity_i, 'h w -> 1 h w')
            if uncertainty is not None:
                scaled_predicted_disparity_i = scaled_predicted_disparity_i * mask_i * uncertainty_i
            else:
                scaled_predicted_disparity_i = scaled_predicted_disparity_i * mask_i
            
            pred_disparity_list.append(scaled_predicted_disparity_i)

        if all_empty:
            return 0 
        pred_disparity_norm   = torch.cat(pred_disparity_list)
        target_disparity_norm = torch.cat(target_disparity_list)
        loss = torch.mean(torch.abs(pred_disparity_norm - target_disparity_norm))
        return loss
