# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import build_heads, REID_HEADS_REGISTRY
from .clas_head import ClasHead
# import all the meta_arch, so they will be registered
from .embedding_head import EmbeddingHead
