#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from one.nn import *

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "vgg11": {
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64,  3, 1, 1]],       # 0
            [-1,     1,      ReLU,              [True]],               # 1
            [-1,     1,      MaxPool2d,         [2, 2]],               # 2
            [-1,     1,      Conv2d,            [128, 3, 1, 1]],       # 3
            [-1,     1,      ReLU,              [True]],               # 4
            [-1,     1,      MaxPool2d,         [2, 2]],               # 5
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 6
            [-1,     1,      ReLU,              [True]],               # 7
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 8
            [-1,     1,      ReLU,              [True]],               # 9
            [-1,     1,      MaxPool2d,         [2, 2]],               # 10
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 11
            [-1,     1,      ReLU,              [True]],               # 12
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 13
            [-1,     1,      ReLU,              [True]],               # 14
            [-1,     1,      MaxPool2d,         [2, 2]],               # 15
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 16
            [-1,     1,      ReLU,              [True]],               # 17
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 18
            [-1,     1,      ReLU,              [True]],               # 19
            [-1,     1,      MaxPool2d,         [2, 2]],               # 20
            [-1,     1,      AdaptiveAvgPool2d, [7]],                  # 21
        ],
        "head": [
            [-1,     1,      VGGClassifier,     [512]],                # 22
        ]
    },
    "vgg11-bn": {
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64,  3, 1, 1]],       # 0
            [-1,     1,      BatchNorm2d,       [64]],                 # 1
            [-1,     1,      ReLU,              [True]],               # 2
            [-1,     1,      MaxPool2d,         [2, 2]],               # 3
            [-1,     1,      Conv2d,            [128, 3, 1, 1]],       # 4
            [-1,     1,      BatchNorm2d,       [128]],                # 5
            [-1,     1,      ReLU,              [True]],               # 6
            [-1,     1,      MaxPool2d,         [2, 2]],               # 7
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 8
            [-1,     1,      BatchNorm2d,       [256]],                # 9
            [-1,     1,      ReLU,              [True]],               # 10
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 11
            [-1,     1,      BatchNorm2d,       [256]],                # 12
            [-1,     1,      ReLU,              [True]],               # 13
            [-1,     1,      MaxPool2d,         [2, 2]],               # 14
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 15
            [-1,     1,      BatchNorm2d,       [512]],                # 16
            [-1,     1,      ReLU,              [True]],               # 17
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 18
            [-1,     1,      BatchNorm2d,       [512]],                # 19
            [-1,     1,      ReLU,              [True]],               # 20
            [-1,     1,      MaxPool2d,         [2, 2]],               # 21
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 22
            [-1,     1,      BatchNorm2d,       [512]],                # 23
            [-1,     1,      ReLU,              [True]],               # 24
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 25
            [-1,     1,      BatchNorm2d,       [512]],                # 26
            [-1,     1,      ReLU,              [True]],               # 27
            [-1,     1,      MaxPool2d,         [2, 2]],               # 28
            [-1,     1,      AdaptiveAvgPool2d, [7]],                  # 29
        ],
        "head": [
            [-1,     1,      VGGClassifier,     [512]],                # 30
        ]
    },
    "vgg13": {
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64,  3, 1, 1]],       # 0
            [-1,     1,      ReLU,              [True]],               # 1
            [-1,     1,      Conv2d,            [64,  3, 1, 1]],       # 2
            [-1,     1,      ReLU,              [True]],               # 3
            [-1,     1,      MaxPool2d,         [2, 2]],               # 4
            [-1,     1,      Conv2d,            [128, 3, 1, 1]],       # 5
            [-1,     1,      ReLU,              [True]],               # 6
            [-1,     1,      Conv2d,            [128, 3, 1, 1]],       # 7
            [-1,     1,      ReLU,              [True]],               # 8
            [-1,     1,      MaxPool2d,         [2, 2]],               # 9
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 10
            [-1,     1,      ReLU,              [True]],               # 11
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 12
            [-1,     1,      ReLU,              [True]],               # 13
            [-1,     1,      MaxPool2d,         [2, 2]],               # 14
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 15
            [-1,     1,      ReLU,              [True]],               # 16
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 17
            [-1,     1,      ReLU,              [True]],               # 18
            [-1,     1,      MaxPool2d,         [2, 2]],               # 19
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 20
            [-1,     1,      ReLU,              [True]],               # 21
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 22
            [-1,     1,      ReLU,              [True]],               # 23
            [-1,     1,      MaxPool2d,         [2, 2]],               # 24
            [-1,     1,      AdaptiveAvgPool2d, [7]],                  # 25
        ],
        "head": [
            [-1,     1,      VGGClassifier,     [512]],                # 26
        ]
    },
    "vgg13-bn": {
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64,  3, 1, 1]],       # 0
            [-1,     1,      BatchNorm2d,       [64]],                 # 1
            [-1,     1,      ReLU,              [True]],               # 2
            [-1,     1,      Conv2d,            [64,  3, 1, 1]],       # 3
            [-1,     1,      BatchNorm2d,       [64]],                 # 4
            [-1,     1,      ReLU,              [True]],               # 5
            [-1,     1,      MaxPool2d,         [2, 2]],               # 6
            [-1,     1,      Conv2d,            [128, 3, 1, 1]],       # 7
            [-1,     1,      BatchNorm2d,       [128]],                # 8
            [-1,     1,      ReLU,              [True]],               # 9
            [-1,     1,      Conv2d,            [128, 3, 1, 1]],       # 10
            [-1,     1,      BatchNorm2d,       [128]],                # 11
            [-1,     1,      ReLU,              [True]],               # 12
            [-1,     1,      MaxPool2d,         [2, 2]],               # 13
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 14
            [-1,     1,      BatchNorm2d,       [256]],                # 15
            [-1,     1,      ReLU,              [True]],               # 16
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 17
            [-1,     1,      BatchNorm2d,       [256]],                # 18
            [-1,     1,      ReLU,              [True]],               # 19
            [-1,     1,      MaxPool2d,         [2, 2]],               # 20
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 21
            [-1,     1,      BatchNorm2d,       [512]],                # 22
            [-1,     1,      ReLU,              [True]],               # 23
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 24
            [-1,     1,      BatchNorm2d,       [512]],                # 25
            [-1,     1,      ReLU,              [True]],               # 26
            [-1,     1,      MaxPool2d,         [2, 2]],               # 27
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 28
            [-1,     1,      BatchNorm2d,       [512]],                # 29
            [-1,     1,      ReLU,              [True]],               # 30
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 31
            [-1,     1,      BatchNorm2d,       [512]],                # 32
            [-1,     1,      ReLU,              [True]],               # 33
            [-1,     1,      MaxPool2d,         [2, 2]],               # 34
            [-1,     1,      AdaptiveAvgPool2d, [7]],                  # 35
        ],
        "head": [
            [-1,     1,      VGGClassifier,     [512]],                # 36
        ]
    },
    "vgg16": {
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64,  3, 1, 1]],       # 0
            [-1,     1,      ReLU,              [True]],               # 1
            [-1,     1,      Conv2d,            [64,  3, 1, 1]],       # 2
            [-1,     1,      ReLU,              [True]],               # 3
            [-1,     1,      MaxPool2d,         [2, 2]],               # 4
            [-1,     1,      Conv2d,            [128, 3, 1, 1]],       # 5
            [-1,     1,      ReLU,              [True]],               # 6
            [-1,     1,      Conv2d,            [128, 3, 1, 1]],       # 7
            [-1,     1,      ReLU,              [True]],               # 8
            [-1,     1,      MaxPool2d,         [2, 2]],               # 9
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 10
            [-1,     1,      ReLU,              [True]],               # 11
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 12
            [-1,     1,      ReLU,              [True]],               # 13
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 14
            [-1,     1,      ReLU,              [True]],               # 15
            [-1,     1,      MaxPool2d,         [2, 2]],               # 16
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 17
            [-1,     1,      ReLU,              [True]],               # 18
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 19
            [-1,     1,      ReLU,              [True]],               # 20
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 21
            [-1,     1,      ReLU,              [True]],               # 22
            [-1,     1,      MaxPool2d,         [2, 2]],               # 23
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 24
            [-1,     1,      ReLU,              [True]],               # 25
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 26
            [-1,     1,      ReLU,              [True]],               # 27
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 28
            [-1,     1,      ReLU,              [True]],               # 29
            [-1,     1,      MaxPool2d,         [2, 2]],               # 30
            [-1,     1,      AdaptiveAvgPool2d, [7]],                  # 31
        ],
        "head": [
            [-1,     1,      VGGClassifier,     [512]],                # 32
        ]
    },
    "vgg16-bn": {
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64,  3, 1, 1]],       # 0
            [-1,     1,      BatchNorm2d,       [64]],                 # 1
            [-1,     1,      ReLU,              [True]],               # 2
            [-1,     1,      Conv2d,            [64,  3, 1, 1]],       # 3
            [-1,     1,      BatchNorm2d,       [64]],                 # 4
            [-1,     1,      ReLU,              [True]],               # 5
            [-1,     1,      MaxPool2d,         [2, 2]],               # 6
            [-1,     1,      Conv2d,            [128, 3, 1, 1]],       # 7
            [-1,     1,      BatchNorm2d,       [128]],                # 8
            [-1,     1,      ReLU,              [True]],               # 9
            [-1,     1,      Conv2d,            [128, 3, 1, 1]],       # 10
            [-1,     1,      BatchNorm2d,       [128]],                # 11
            [-1,     1,      ReLU,              [True]],               # 12
            [-1,     1,      MaxPool2d,         [2, 2]],               # 13
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 14
            [-1,     1,      BatchNorm2d,       [256]],                # 15
            [-1,     1,      ReLU,              [True]],               # 16
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 17
            [-1,     1,      BatchNorm2d,       [256]],                # 18
            [-1,     1,      ReLU,              [True]],               # 19
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 20
            [-1,     1,      BatchNorm2d,       [256]],                # 21
            [-1,     1,      ReLU,              [True]],               # 22
            [-1,     1,      MaxPool2d,         [2, 2]],               # 23
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 24
            [-1,     1,      BatchNorm2d,       [512]],                # 25
            [-1,     1,      ReLU,              [True]],               # 26
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 27
            [-1,     1,      BatchNorm2d,       [512]],                # 28
            [-1,     1,      ReLU,              [True]],               # 29
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 30
            [-1,     1,      BatchNorm2d,       [512]],                # 31
            [-1,     1,      ReLU,              [True]],               # 32
            [-1,     1,      MaxPool2d,         [2, 2]],               # 33
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 34
            [-1,     1,      BatchNorm2d,       [512]],                # 35
            [-1,     1,      ReLU,              [True]],               # 36
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 37
            [-1,     1,      BatchNorm2d,       [512]],                # 38
            [-1,     1,      ReLU,              [True]],               # 39
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 40
            [-1,     1,      BatchNorm2d,       [512]],                # 41
            [-1,     1,      ReLU,              [True]],               # 42
            [-1,     1,      MaxPool2d,         [2, 2]],               # 43
            [-1,     1,      AdaptiveAvgPool2d, [7]],                  # 44
        ],
        "head": [
            [-1,     1,      VGGClassifier,     [512]],                # 45
        ]
    },
    "vgg19": {
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64,  3, 1, 1]],       # 0
            [-1,     1,      ReLU,              [True]],               # 1
            [-1,     1,      Conv2d,            [64,  3, 1, 1]],       # 2
            [-1,     1,      ReLU,              [True]],               # 3
            [-1,     1,      MaxPool2d,         [2, 2]],               # 4
            [-1,     1,      Conv2d,            [128, 3, 1, 1]],       # 5
            [-1,     1,      ReLU,              [True]],               # 6
            [-1,     1,      Conv2d,            [128, 3, 1, 1]],       # 7
            [-1,     1,      ReLU,              [True]],               # 8
            [-1,     1,      MaxPool2d,         [2, 2]],               # 9
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 10
            [-1,     1,      ReLU,              [True]],               # 11
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 12
            [-1,     1,      ReLU,              [True]],               # 13
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 14
            [-1,     1,      ReLU,              [True]],               # 15
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 16
            [-1,     1,      ReLU,              [True]],               # 17
            [-1,     1,      MaxPool2d,         [2, 2]],               # 18
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 19
            [-1,     1,      ReLU,              [True]],               # 20
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 21
            [-1,     1,      ReLU,              [True]],               # 22
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 23
            [-1,     1,      ReLU,              [True]],               # 24
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 25
            [-1,     1,      ReLU,              [True]],               # 26
            [-1,     1,      MaxPool2d,         [2, 2]],               # 27
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 28
            [-1,     1,      ReLU,              [True]],               # 29
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 30
            [-1,     1,      ReLU,              [True]],               # 31
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 32
            [-1,     1,      ReLU,              [True]],               # 33
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 34
            [-1,     1,      ReLU,              [True]],               # 35
            [-1,     1,      MaxPool2d,         [2, 2]],               # 36
            [-1,     1,      AdaptiveAvgPool2d, [7]],                  # 37
        ],
        "head": [
            [-1,     1,      VGGClassifier,     [512]],                # 38
        ]
    },
    "vgg19-bn": {
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64,  3, 1, 1]],       # 0
            [-1,     1,      BatchNorm2d,       [64]],                 # 1
            [-1,     1,      ReLU,              [True]],               # 2
            [-1,     1,      Conv2d,            [64,  3, 1, 1]],       # 3
            [-1,     1,      BatchNorm2d,       [64]],                 # 4
            [-1,     1,      ReLU,              [True]],               # 5
            [-1,     1,      MaxPool2d,         [2, 2]],               # 6
            [-1,     1,      Conv2d,            [128, 3, 1, 1]],       # 7
            [-1,     1,      BatchNorm2d,       [128]],                # 8
            [-1,     1,      ReLU,              [True]],               # 9
            [-1,     1,      Conv2d,            [128, 3, 1, 1]],       # 10
            [-1,     1,      BatchNorm2d,       [128]],                # 11
            [-1,     1,      ReLU,              [True]],               # 12
            [-1,     1,      MaxPool2d,         [2, 2]],               # 13
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 14
            [-1,     1,      BatchNorm2d,       [256]],                # 15
            [-1,     1,      ReLU,              [True]],               # 16
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 17
            [-1,     1,      BatchNorm2d,       [256]],                # 18
            [-1,     1,      ReLU,              [True]],               # 19
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 20
            [-1,     1,      BatchNorm2d,       [256]],                # 21
            [-1,     1,      ReLU,              [True]],               # 22
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 23
            [-1,     1,      BatchNorm2d,       [256]],                # 24
            [-1,     1,      ReLU,              [True]],               # 25
            [-1,     1,      MaxPool2d,         [2, 2]],               # 26
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 27
            [-1,     1,      BatchNorm2d,       [512]],                # 28
            [-1,     1,      ReLU,              [True]],               # 29
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 30
            [-1,     1,      BatchNorm2d,       [512]],                # 31
            [-1,     1,      ReLU,              [True]],               # 32
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 33
            [-1,     1,      BatchNorm2d,       [512]],                # 34
            [-1,     1,      ReLU,              [True]],               # 35
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 36
            [-1,     1,      BatchNorm2d,       [512]],                # 37
            [-1,     1,      ReLU,              [True]],               # 38
            [-1,     1,      MaxPool2d,         [2, 2]],               # 39
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 40
            [-1,     1,      BatchNorm2d,       [512]],                # 41
            [-1,     1,      ReLU,              [True]],               # 42
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 43
            [-1,     1,      BatchNorm2d,       [512]],                # 44
            [-1,     1,      ReLU,              [True]],               # 45
            [-1,     1,      Conv2d,            [512, 3, 1, 1]],       # 46
            [-1,     1,      BatchNorm2d,       [512]],                # 47
            [-1,     1,      ReLU,              [True]],               # 48
            [-1,     1,      MaxPool2d,         [2, 2]],               # 49
            [-1,     1,      BatchNorm2d,       [512]],                # 50
            [-1,     1,      ReLU,              [True]],               # 51
            [-1,     1,      MaxPool2d,         [2, 2]],               # 52
            [-1,     1,      AdaptiveAvgPool2d, [7]],                  # 53
        ],
        "head": [
            [-1,     1,      VGGClassifier,     [512]],                # 54
        ]
    },
}


@MODELS.register(name="vgg")
class VGG(ImageClassificationModel):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        project (str | None): Project name. Defaults to None.
        name (str | None): Model's name. In case None is given, it will be
            `self.__class__.__name__`. Defaults to None.
        fullname (str | None): Model's fullname in the following format:
            {name}-{data_name}-{postfix}. In case None is given, it will be
            `self.name`. Defaults to None.
        channels (int): Input channel. Defaults to 3.
        num_classes (int | None): Number of classes for classification or
            detection tasks. Defaults to None.
        classlabels (ClassLabels | None): ClassLabels object that contains all
            labels in the dataset. Defaults to None.
        pretrained (Pretrained): Initialize weights from pretrained.
            - If True, use the original pretrained described by the author
              (usually, ImageNet or COCO). By default, it is the first element
              in the `model_zoo` dictionary.
            - If str and is a file/path, then load weights from saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
        phase (ModelPhase_): Model's running phase. Defaults to training.
        loss (Losses_ | None): Loss function for training model.
            Defaults to None.
        metrics (Metrics_ | None): Metric(s) for validating and testing model.
            Defaults to None.
        optimizers (Optimizers_ | None): Optimizer(s) for training model.
            Defaults to None.
        debug (dict | Munch | None): Debug configs. Defaults to None.
        verbose (bool): Verbosity.
    """
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "vgg11.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "vgg",
        fullname   : str          | None = "vgg11",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "vgg11"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
        
        super().__init__(
            cfg         = cfg,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = pretrained,
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def init_weights(self, m: Module):
        if isinstance(m, Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


@MODELS.register(name="vgg11")
class VGG11(VGG):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        project (str | None): Project name. Defaults to None.
        name (str | None): Model's name. In case None is given, it will be
            `self.__class__.__name__`. Defaults to None.
        fullname (str | None): Model's fullname in the following format:
            {name}-{data_name}-{postfix}. In case None is given, it will be
            `self.name`. Defaults to None.
        channels (int): Input channel. Defaults to 3.
        num_classes (int | None): Number of classes for classification or
            detection tasks. Defaults to None.
        classlabels (ClassLabels | None): ClassLabels object that contains all
            labels in the dataset. Defaults to None.
        pretrained (Pretrained): Initialize weights from pretrained.
            - If True, use the original pretrained described by the author
              (usually, ImageNet or COCO). By default, it is the first element
              in the `model_zoo` dictionary.
            - If str and is a file/path, then load weights from saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
        phase (ModelPhase_): Model's running phase. Defaults to training.
        loss (Losses_ | None): Loss function for training model.
            Defaults to None.
        metrics (Metrics_ | None): Metric(s) for validating and testing model.
            Defaults to None.
        optimizers (Optimizers_ | None): Optimizer(s) for training model.
            Defaults to None.
        debug (dict | Munch | None): Debug configs. Defaults to None.
        verbose (bool): Verbosity.
    """
    
    model_zoo = {
        "imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/vgg11-8a719046.pth",
            filename    = "vgg11-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "vgg11.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "vgg",
        fullname   : str          | None = "vgg11",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "vgg11"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
        
        super().__init__(
            cfg         = cfg,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = VGG11.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) and self.pretrained["name"] == "imagenet":
            state_dict = load_state_dict_from_path(
                model_dir=self.pretrained_dir, **self.pretrained
            )
            """
            for k in sorted(self.model.state_dict().keys()):
                print(f"model_state_dict[\"{k}\"] = ")
            for k in sorted(state_dict.keys()):
                print(f"state_dict[\"{k}\"]")
            """
            model_state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                k = k.replace("features.", "")
                if k in model_state_dict:
                    model_state_dict[k] = v
            model_state_dict["22.linear1.weight"] = state_dict["classifier.0.weight"]
            model_state_dict["22.linear1.bias"]   = state_dict["classifier.0.bias"]
            model_state_dict["22.linear2.weight"] = state_dict["classifier.3.weight"]
            model_state_dict["22.linear2.bias"]   = state_dict["classifier.3.bias"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["22.linear3.weight"] = state_dict["classifier.6.weight"]
                model_state_dict["22.linear3.bias"]   = state_dict["classifier.6.bias"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="vgg11_bn")
class VGG11Bn(VGG):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        project (str | None): Project name. Defaults to None.
        name (str | None): Model's name. In case None is given, it will be
            `self.__class__.__name__`. Defaults to None.
        fullname (str | None): Model's fullname in the following format:
            {name}-{data_name}-{postfix}. In case None is given, it will be
            `self.name`. Defaults to None.
        channels (int): Input channel. Defaults to 3.
        num_classes (int | None): Number of classes for classification or
            detection tasks. Defaults to None.
        classlabels (ClassLabels | None): ClassLabels object that contains all
            labels in the dataset. Defaults to None.
        pretrained (Pretrained): Initialize weights from pretrained.
            - If True, use the original pretrained described by the author
              (usually, ImageNet or COCO). By default, it is the first element
              in the `model_zoo` dictionary.
            - If str and is a file/path, then load weights from saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
        phase (ModelPhase_): Model's running phase. Defaults to training.
        loss (Losses_ | None): Loss function for training model.
            Defaults to None.
        metrics (Metrics_ | None): Metric(s) for validating and testing model.
            Defaults to None.
        optimizers (Optimizers_ | None): Optimizer(s) for training model.
            Defaults to None.
        debug (dict | Munch | None): Debug configs. Defaults to None.
        verbose (bool): Verbosity.
    """
    
    model_zoo = {
        "imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
            filename    = "vgg11_bn-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "vgg11-bn.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "vgg",
        fullname   : str          | None = "vgg11-bn",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "vgg11-bn"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
        
        super().__init__(
            cfg         = cfg,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = VGG11Bn.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) and self.pretrained["name"] == "imagenet":
            state_dict = load_state_dict_from_path(
                model_dir=self.pretrained_dir, **self.pretrained
            )
            """
            for k in sorted(self.model.state_dict().keys()):
                print(f"model_state_dict[\"{k}\"] = ")
            for k in sorted(state_dict.keys()):
                print(f"state_dict[\"{k}\"]")
            """
            model_state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                k = k.replace("features.", "")
                if k in model_state_dict:
                    model_state_dict[k] = v
            model_state_dict["30.linear1.weight"] = state_dict["classifier.0.weight"]
            model_state_dict["30.linear1.bias"]   = state_dict["classifier.0.bias"]
            model_state_dict["30.linear2.weight"] = state_dict["classifier.3.weight"]
            model_state_dict["30.linear2.bias"]   = state_dict["classifier.3.bias"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["30.linear3.weight"] = state_dict["classifier.6.weight"]
                model_state_dict["30.linear3.bias"]   = state_dict["classifier.6.bias"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="vgg13")
class VGG13(VGG):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        project (str | None): Project name. Defaults to None.
        name (str | None): Model's name. In case None is given, it will be
            `self.__class__.__name__`. Defaults to None.
        fullname (str | None): Model's fullname in the following format:
            {name}-{data_name}-{postfix}. In case None is given, it will be
            `self.name`. Defaults to None.
        channels (int): Input channel. Defaults to 3.
        num_classes (int | None): Number of classes for classification or
            detection tasks. Defaults to None.
        classlabels (ClassLabels | None): ClassLabels object that contains all
            labels in the dataset. Defaults to None.
        pretrained (Pretrained): Initialize weights from pretrained.
            - If True, use the original pretrained described by the author
              (usually, ImageNet or COCO). By default, it is the first element
              in the `model_zoo` dictionary.
            - If str and is a file/path, then load weights from saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
        phase (ModelPhase_): Model's running phase. Defaults to training.
        loss (Losses_ | None): Loss function for training model.
            Defaults to None.
        metrics (Metrics_ | None): Metric(s) for validating and testing model.
            Defaults to None.
        optimizers (Optimizers_ | None): Optimizer(s) for training model.
            Defaults to None.
        debug (dict | Munch | None): Debug configs. Defaults to None.
        verbose (bool): Verbosity.
    """
    
    model_zoo = {
        "imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/vgg13-19584684.pth",
            filename    = "vgg13-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "vgg13.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "vgg",
        fullname   : str          | None = "vgg13",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "vgg13"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
        
        super().__init__(
            cfg         = cfg,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = VGG13.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) and self.pretrained["name"] == "imagenet":
            state_dict = load_state_dict_from_path(
                model_dir=self.pretrained_dir, **self.pretrained
            )
            """
            for k in sorted(self.model.state_dict().keys()):
                print(f"model_state_dict[\"{k}\"] = ")
            for k in sorted(state_dict.keys()):
                print(f"state_dict[\"{k}\"]")
            """
            model_state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                k = k.replace("features.", "")
                if k in model_state_dict:
                    model_state_dict[k] = v
            model_state_dict["26.linear1.weight"] = state_dict["classifier.0.weight"]
            model_state_dict["26.linear1.bias"]   = state_dict["classifier.0.bias"]
            model_state_dict["26.linear2.weight"] = state_dict["classifier.3.weight"]
            model_state_dict["26.linear2.bias"]   = state_dict["classifier.3.bias"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["26.linear3.weight"] = state_dict["classifier.6.weight"]
                model_state_dict["26.linear3.bias"]   = state_dict["classifier.6.bias"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="vgg13_bn")
class VGG13Bn(VGG):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        project (str | None): Project name. Defaults to None.
        name (str | None): Model's name. In case None is given, it will be
            `self.__class__.__name__`. Defaults to None.
        fullname (str | None): Model's fullname in the following format:
            {name}-{data_name}-{postfix}. In case None is given, it will be
            `self.name`. Defaults to None.
        channels (int): Input channel. Defaults to 3.
        num_classes (int | None): Number of classes for classification or
            detection tasks. Defaults to None.
        classlabels (ClassLabels | None): ClassLabels object that contains all
            labels in the dataset. Defaults to None.
        pretrained (Pretrained): Initialize weights from pretrained.
            - If True, use the original pretrained described by the author
              (usually, ImageNet or COCO). By default, it is the first element
              in the `model_zoo` dictionary.
            - If str and is a file/path, then load weights from saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
        phase (ModelPhase_): Model's running phase. Defaults to training.
        loss (Losses_ | None): Loss function for training model.
            Defaults to None.
        metrics (Metrics_ | None): Metric(s) for validating and testing model.
            Defaults to None.
        optimizers (Optimizers_ | None): Optimizer(s) for training model.
            Defaults to None.
        debug (dict | Munch | None): Debug configs. Defaults to None.
        verbose (bool): Verbosity.
    """
    
    model_zoo = {
        "imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
            filename    = "vgg13_bn-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "vgg13-bn.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "vgg",
        fullname   : str          | None = "vgg13-bn",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "vgg13-bn"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
        
        super().__init__(
            cfg         = cfg,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = VGG13Bn.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) and self.pretrained["name"] == "imagenet":
            state_dict = load_state_dict_from_path(
                model_dir=self.pretrained_dir, **self.pretrained
            )
            """
            for k in sorted(self.model.state_dict().keys()):
                print(f"model_state_dict[\"{k}\"] = ")
            for k in sorted(state_dict.keys()):
                print(f"state_dict[\"{k}\"]")
            """
            model_state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                k = k.replace("features.", "")
                if k in model_state_dict:
                    model_state_dict[k] = v
            model_state_dict["36.linear1.weight"] = state_dict["classifier.0.weight"]
            model_state_dict["36.linear1.bias"]   = state_dict["classifier.0.bias"]
            model_state_dict["36.linear2.weight"] = state_dict["classifier.3.weight"]
            model_state_dict["36.linear2.bias"]   = state_dict["classifier.3.bias"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["36.linear3.weight"] = state_dict["classifier.6.weight"]
                model_state_dict["36.linear3.bias"]   = state_dict["classifier.6.bias"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="vgg16")
class VGG16(VGG):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        project (str | None): Project name. Defaults to None.
        name (str | None): Model's name. In case None is given, it will be
            `self.__class__.__name__`. Defaults to None.
        fullname (str | None): Model's fullname in the following format:
            {name}-{data_name}-{postfix}. In case None is given, it will be
            `self.name`. Defaults to None.
        channels (int): Input channel. Defaults to 3.
        num_classes (int | None): Number of classes for classification or
            detection tasks. Defaults to None.
        classlabels (ClassLabels | None): ClassLabels object that contains all
            labels in the dataset. Defaults to None.
        pretrained (Pretrained): Initialize weights from pretrained.
            - If True, use the original pretrained described by the author
              (usually, ImageNet or COCO). By default, it is the first element
              in the `model_zoo` dictionary.
            - If str and is a file/path, then load weights from saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
        phase (ModelPhase_): Model's running phase. Defaults to training.
        loss (Losses_ | None): Loss function for training model.
            Defaults to None.
        metrics (Metrics_ | None): Metric(s) for validating and testing model.
            Defaults to None.
        optimizers (Optimizers_ | None): Optimizer(s) for training model.
            Defaults to None.
        debug (dict | Munch | None): Debug configs. Defaults to None.
        verbose (bool): Verbosity.
    """
    
    model_zoo = {
        "imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/vgg16-397923af.pth",
            filename    = "vgg16-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "vgg16.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "vgg",
        fullname   : str          | None = "vgg16",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "vgg16"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
        
        super().__init__(
            cfg         = cfg,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = VGG16.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) and self.pretrained["name"] == "imagenet":
            state_dict = load_state_dict_from_path(
                model_dir=self.pretrained_dir, **self.pretrained
            )
            """
            for k in sorted(self.model.state_dict().keys()):
                print(f"model_state_dict[\"{k}\"] = ")
            for k in sorted(state_dict.keys()):
                print(f"state_dict[\"{k}\"]")
            """
            model_state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                k = k.replace("features.", "")
                if k in model_state_dict:
                    model_state_dict[k] = v
            model_state_dict["32.linear1.weight"] = state_dict["classifier.0.weight"]
            model_state_dict["32.linear1.bias"]   = state_dict["classifier.0.bias"]
            model_state_dict["32.linear2.weight"] = state_dict["classifier.3.weight"]
            model_state_dict["32.linear2.bias"]   = state_dict["classifier.3.bias"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["32.linear3.weight"] = state_dict["classifier.6.weight"]
                model_state_dict["32.linear3.bias"]   = state_dict["classifier.6.bias"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="vgg16_bn")
class VGG16Bn(VGG):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        project (str | None): Project name. Defaults to None.
        name (str | None): Model's name. In case None is given, it will be
            `self.__class__.__name__`. Defaults to None.
        fullname (str | None): Model's fullname in the following format:
            {name}-{data_name}-{postfix}. In case None is given, it will be
            `self.name`. Defaults to None.
        channels (int): Input channel. Defaults to 3.
        num_classes (int | None): Number of classes for classification or
            detection tasks. Defaults to None.
        classlabels (ClassLabels | None): ClassLabels object that contains all
            labels in the dataset. Defaults to None.
        pretrained (Pretrained): Initialize weights from pretrained.
            - If True, use the original pretrained described by the author
              (usually, ImageNet or COCO). By default, it is the first element
              in the `model_zoo` dictionary.
            - If str and is a file/path, then load weights from saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
        phase (ModelPhase_): Model's running phase. Defaults to training.
        loss (Losses_ | None): Loss function for training model.
            Defaults to None.
        metrics (Metrics_ | None): Metric(s) for validating and testing model.
            Defaults to None.
        optimizers (Optimizers_ | None): Optimizer(s) for training model.
            Defaults to None.
        debug (dict | Munch | None): Debug configs. Defaults to None.
        verbose (bool): Verbosity.
    """
    
    model_zoo = {
        "imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
            filename    = "vgg16_bn-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "vgg16-bn.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "vgg",
        fullname   : str          | None = "vgg16-bn",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "vgg16-bn"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
        
        super().__init__(
            cfg         = cfg,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = VGG16Bn.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) and self.pretrained["name"] == "imagenet":
            state_dict = load_state_dict_from_path(
                model_dir=self.pretrained_dir, **self.pretrained
            )
            """
            for k in sorted(self.model.state_dict().keys()):
                print(f"model_state_dict[\"{k}\"] = ")
            for k in sorted(state_dict.keys()):
                print(f"state_dict[\"{k}\"]")
            """
            model_state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                k = k.replace("features.", "")
                if k in model_state_dict:
                    model_state_dict[k] = v
            model_state_dict["45.linear1.weight"] = state_dict["classifier.0.weight"]
            model_state_dict["45.linear1.bias"]   = state_dict["classifier.0.bias"]
            model_state_dict["45.linear2.weight"] = state_dict["classifier.3.weight"]
            model_state_dict["45.linear2.bias"]   = state_dict["classifier.3.bias"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["45.linear3.weight"] = state_dict["classifier.6.weight"]
                model_state_dict["45.linear3.bias"]   = state_dict["classifier.6.bias"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="vgg19")
class VGG19(VGG):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        project (str | None): Project name. Defaults to None.
        name (str | None): Model's name. In case None is given, it will be
            `self.__class__.__name__`. Defaults to None.
        fullname (str | None): Model's fullname in the following format:
            {name}-{data_name}-{postfix}. In case None is given, it will be
            `self.name`. Defaults to None.
        channels (int): Input channel. Defaults to 3.
        num_classes (int | None): Number of classes for classification or
            detection tasks. Defaults to None.
        classlabels (ClassLabels | None): ClassLabels object that contains all
            labels in the dataset. Defaults to None.
        pretrained (Pretrained): Initialize weights from pretrained.
            - If True, use the original pretrained described by the author
              (usually, ImageNet or COCO). By default, it is the first element
              in the `model_zoo` dictionary.
            - If str and is a file/path, then load weights from saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
        phase (ModelPhase_): Model's running phase. Defaults to training.
        loss (Losses_ | None): Loss function for training model.
            Defaults to None.
        metrics (Metrics_ | None): Metric(s) for validating and testing model.
            Defaults to None.
        optimizers (Optimizers_ | None): Optimizer(s) for training model.
            Defaults to None.
        debug (dict | Munch | None): Debug configs. Defaults to None.
        verbose (bool): Verbosity.
    """
    
    model_zoo = {
        "imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
            filename    = "vgg19-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "vgg19.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "vgg",
        fullname   : str          | None = "vgg19",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "vgg19"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
        
        super().__init__(
            cfg         = cfg,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = VGG19.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) and self.pretrained["name"] == "imagenet":
            state_dict = load_state_dict_from_path(
                model_dir=self.pretrained_dir, **self.pretrained
            )
            """
            for k in sorted(self.model.state_dict().keys()):
                print(f"model_state_dict[\"{k}\"] = ")
            for k in sorted(state_dict.keys()):
                print(f"state_dict[\"{k}\"]")
            """
            model_state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                k = k.replace("features.", "")
                if k in model_state_dict:
                    model_state_dict[k] = v
            model_state_dict["38.linear1.weight"] = state_dict["classifier.0.weight"]
            model_state_dict["38.linear1.bias"]   = state_dict["classifier.0.bias"]
            model_state_dict["38.linear2.weight"] = state_dict["classifier.3.weight"]
            model_state_dict["38.linear2.bias"]   = state_dict["classifier.3.bias"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["38.linear3.weight"] = state_dict["classifier.6.weight"]
                model_state_dict["38.linear3.bias"]   = state_dict["classifier.6.bias"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="vgg19_bn")
class VGG19Bn(VGG):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        project (str | None): Project name. Defaults to None.
        name (str | None): Model's name. In case None is given, it will be
            `self.__class__.__name__`. Defaults to None.
        fullname (str | None): Model's fullname in the following format:
            {name}-{data_name}-{postfix}. In case None is given, it will be
            `self.name`. Defaults to None.
        channels (int): Input channel. Defaults to 3.
        num_classes (int | None): Number of classes for classification or
            detection tasks. Defaults to None.
        classlabels (ClassLabels | None): ClassLabels object that contains all
            labels in the dataset. Defaults to None.
        pretrained (Pretrained): Initialize weights from pretrained.
            - If True, use the original pretrained described by the author
              (usually, ImageNet or COCO). By default, it is the first element
              in the `model_zoo` dictionary.
            - If str and is a file/path, then load weights from saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
        phase (ModelPhase_): Model's running phase. Defaults to training.
        loss (Losses_ | None): Loss function for training model.
            Defaults to None.
        metrics (Metrics_ | None): Metric(s) for validating and testing model.
            Defaults to None.
        optimizers (Optimizers_ | None): Optimizer(s) for training model.
            Defaults to None.
        debug (dict | Munch | None): Debug configs. Defaults to None.
        verbose (bool): Verbosity.
    """
    
    model_zoo = {
        "imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
            filename    = "vgg19_bn-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "vgg19-bn.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "vgg",
        fullname   : str          | None = "vgg19-bn",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "vgg19-bn"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
        
        super().__init__(
            cfg         = cfg,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = VGG19Bn.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) and self.pretrained["name"] == "imagenet":
            state_dict = load_state_dict_from_path(
                model_dir=self.pretrained_dir, **self.pretrained
            )
            """
            for k in sorted(self.model.state_dict().keys()):
                print(f"model_state_dict[\"{k}\"] = ")
            for k in sorted(state_dict.keys()):
                print(f"state_dict[\"{k}\"]")
            """
            model_state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                k = k.replace("features.", "")
                if k in model_state_dict:
                    model_state_dict[k] = v
            model_state_dict["54.linear1.weight"] = state_dict["classifier.0.weight"]
            model_state_dict["54.linear1.bias"]   = state_dict["classifier.0.bias"]
            model_state_dict["54.linear2.weight"] = state_dict["classifier.3.weight"]
            model_state_dict["54.linear2.bias"]   = state_dict["classifier.3.bias"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["54.linear3.weight"] = state_dict["classifier.6.weight"]
                model_state_dict["54.linear3.bias"]   = state_dict["classifier.6.bias"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
