"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..core import register

__all__ = ["PostProcessor"]


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessorOriginal(nn.Module):
    
    __share__ = [
        "num_classes",
        "use_focal_loss",
        "num_top_queries",
        "remap_mscoco_category"
    ]

    def __init__(
        self,
        num_classes            = 80,
        use_focal_loss         = True,
        num_top_queries        = 300,   # max_dets
        remap_mscoco_category  = False,
        deploy_num_top_queries = 300,
        deploy_out_fmt         = "xyxy"
    ):
        super().__init__()
        self.use_focal_loss         = use_focal_loss
        self.num_top_queries        = num_top_queries
        self.num_classes            = int(num_classes)
        self.remap_mscoco_category  = remap_mscoco_category
        self.deploy_num_top_queries = deploy_num_top_queries
        self.deploy_out_fmt         = deploy_out_fmt
        self.deploy_mode            = False

    def extra_repr(self) -> str:
        return (
            f"use_focal_loss={self.use_focal_loss}, "
            f"num_classes={self.num_classes}, "
            f"num_top_queries={self.num_top_queries}, "
            f"deploy_num_top_queries={self.deploy_num_top_queries}, "
            f"deploy_out_fmt={self.deploy_out_fmt}"
        )

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        # My Modification
        if self.deploy_mode:
            bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt=self.deploy_out_fmt)
        else:
            bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.deploy_mode:
            num_top_queries = self.deploy_num_top_queries
        else:
            num_top_queries = self.num_top_queries

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index  = index // self.num_classes
            boxes  = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > num_top_queries:
                scores, index = torch.topk(scores, num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes  = torch.gather(boxes,  dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()]).to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self


@register()
class PostProcessor(nn.Module):

    __share__ = [
        "num_classes",
        "use_focal_loss",
        "num_top_queries",
        "remap_mscoco_category"
    ]

    def __init__(
        self,
        num_classes            = 80,
        use_focal_loss         = True,
        num_top_queries        = 300,   # max_dets
        remap_mscoco_category  = False,
        deploy_num_top_queries = 300,
        deploy_out_fmt         = "xyxy"
    ):
        super().__init__()
        self.use_focal_loss         = use_focal_loss
        self.num_top_queries        = num_top_queries
        self.num_classes            = int(num_classes)
        self.remap_mscoco_category  = remap_mscoco_category
        self.deploy_num_top_queries = deploy_num_top_queries
        self.deploy_out_fmt         = deploy_out_fmt
        self.deploy_mode            = False

    def extra_repr(self) -> str:
        return (
            f"use_focal_loss={self.use_focal_loss}, "
            f"num_classes={self.num_classes}, "
            f"num_top_queries={self.num_top_queries}, "
            f"deploy_num_top_queries={self.deploy_num_top_queries}, "
            f"deploy_out_fmt={self.deploy_out_fmt}"
        )

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits = outputs["pred_logits"]
        boxes  = outputs["pred_boxes"]
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        # My Modification
        scale = orig_target_sizes.repeat(1, 2).unsqueeze(1)  # (B, 1, 4)
        if self.deploy_mode:
            bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt=self.deploy_out_fmt) * scale
        else:
            bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy") * scale

        if self.deploy_mode:
            num_top_queries = self.deploy_num_top_queries
        else:
            num_top_queries = self.num_top_queries

        if self.use_focal_loss:
            scores         = F.sigmoid(logits)
            scores_flat    = scores.flatten(1)
            scores, index  = torch.topk(scores_flat, num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels         = torch.remainder(index, self.num_classes)
            index          = index // self.num_classes
            index_expanded = index.unsqueeze(-1).expand(-1, -1, 4)
            boxes          = bbox_pred.gather(1, index_expanded)
        else:
            scores         = F.softmax(logits, dim=-1)[..., :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > num_top_queries:
                scores, index  = torch.topk(scores, num_top_queries, dim=-1)
                labels         = torch.gather(labels, dim=1, index=index)
                index_expanded = index.unsqueeze(-1).expand(-1, -1, 4)
                boxes          = bbox_pred.gather(1, index_expanded)
            else:
                boxes = bbox_pred[:, :num_top_queries]

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()]).to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
