"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Dict

from ._solver import BaseSolver
from .ec_solver import ECSolver

TASKS :Dict[str, BaseSolver] = {
    'detection': ECSolver,
    'segmentation': ECSolver
}
