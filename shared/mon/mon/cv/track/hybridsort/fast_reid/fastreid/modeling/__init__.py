# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from . import losses
from .backbones import (
	BACKBONE_REGISTRY, build_backbone, build_resnet_backbone,
)
from .heads import (build_heads, EmbeddingHead, REID_HEADS_REGISTRY)
from .meta_arch import (
	build_model,
	META_ARCH_REGISTRY,
)

__all__ = [k for k in globals().keys() if not k.startswith("_")]
