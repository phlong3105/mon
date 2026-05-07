"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T

from ...core import GLOBAL_CONFIG, register
from ._transforms import EmptyTransform

torchvision.disable_beta_transforms_warning()



@register()
class Compose(T.Compose):
    def __init__(self, ops, policy=None, mosaic_prob=-0.1) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    transform = getattr(GLOBAL_CONFIG[name]['_pymodule'], GLOBAL_CONFIG[name]['_name'])(**op)
                    transforms.append(transform)
                    op['type'] = name

                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError('')
        else:
            transforms =[EmptyTransform(), ]

        super().__init__(transforms=transforms)

        self.mosaic_prob = mosaic_prob
        if policy is None:
            policy = {'name': 'default'}
        self.global_samples = 0
        self.policy = policy
        self.warning_mosaic_start = True

    def __call__(self, image, target, dataset=None):
        return self.get_forward(self.policy['name'])(image, target, dataset)

    def get_forward(self, name):
        forwards = {
            'default': self.default_forward,
            'stop_epoch': self.stop_epoch_forward,
        }
        return forwards[name]

    def default_forward(self, image, target, dataset=None):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

    def stop_epoch_forward(self, image, target, dataset=None):
        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_epoch = self.policy['epoch']

        if isinstance(policy_epoch, list) and len(policy_epoch) == 3:
            if policy_epoch[0] <= cur_epoch < policy_epoch[1]:
                with_mosaic = random.random() <= self.mosaic_prob
            else:
                with_mosaic = False

            for transform in self.transforms:
                if (type(transform).__name__ in policy_ops and cur_epoch < policy_epoch[0]):  
                    pass
                elif (type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch[-1]):   
                    pass
                elif (type(transform).__name__ == 'MixupCopypasteMosaic' and cur_epoch >= policy_epoch[1]):
                    pass
                else:
                    if (type(transform).__name__ == 'PoseMosaic' and not with_mosaic):    
                        pass
                    elif (type(transform).__name__ == 'RandomZoomOut' or type(transform).__name__ == 'RandomCrop') and with_mosaic:      
                        pass
                    else:
                        if type(transform).__name__ == 'PoseMosaic':
                            if self.warning_mosaic_start:
                                print(f'     ### Mosaic is being used @ epoch {cur_epoch}...')
                                self.warning_mosaic_start = False
                            image, target = transform(image, target, dataset)
                        else:
                            image, target = transform(image, target)
        else:
            for transform in self.transforms:
                image, target = transform(image, target)

        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
