#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Neural Networks (NN).

A neural network is a machine learning model that stacks simple "neurons" in
layers and learns pattern-recognizing weights and biases from data to map inputs
to outputs.

Notes:
    - In this package, we follow the same coding conventions as PyTorch to
      maintain consistency.
    - If you don't know what to do, just look at the PyTorch source code.

References:
    - https://www.ibm.com/think/topics/deep-learning#763338456
"""

__all__ = [
    "ModelMixin",
]

# noinspection PyUnusedImports
from torch.nn import *  # Export all modules from ``torch.nn``

from mon.nn.cnn import *
from mon.nn.cnn import (
    activation as activation,
    conv as conv,
    norm as norm,
    padding as padding,
    pooling as pooling,
)
from mon.nn.container import *
from mon.nn.inr import *
from mon.nn.misc import *
from mon.nn.misc import (
    dropout as dropout,
    fusion as fusion,
)
from mon.nn.mlp import *
from mon.nn.mlp import (
    linear as linear,
)
from mon.nn.model import ModelMixin
from mon.nn.rnn import *
from mon.nn.transformer import *
from mon.nn.transformer import (
    attention as attention,
)
