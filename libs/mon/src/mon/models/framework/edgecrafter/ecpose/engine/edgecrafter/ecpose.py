
# EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
# Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
# (https://github.com/SebastianJanampa/DETRPose)
# ------------------------------------------------------------------------
# Modified from Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

from torch import nn

from ..core import register

__all__ = ['ECPose', ]

@register()
class ECPose(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder',]

    def __init__(
        self, 
        backbone, 
        encoder, 
        decoder
        ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self

    def forward(self, samples, targets=None):
        feats = self.backbone(samples)
        feats = self.encoder(feats)
        out = self.decoder(feats, targets, samples if self.training else None)
        return out

