#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for recurrent neural networks (RNNs).

This module implements various recurrent neural network (RNN) components and
architectures.

References:
    - Definition: https://www.ibm.com/think/topics/recurrent-neural-networks#763338458
"""

__all__ = [
    "GRU",
    "GRUCell",
    "LSTM",
    "LSTMCell",
    "RNN",
    "RNNBase",
    "RNNCell",
    "RNNCellBase",
]

from torch.nn.modules.rnn import *
