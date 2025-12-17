# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# import all the meta_arch, so they will be registered
from .baseline import Baseline
from .build import build_model, META_ARCH_REGISTRY
from .distiller import Distiller
from .mgn import MGN
from .moco import MoCo
