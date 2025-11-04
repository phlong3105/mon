#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Recurrent Neural Networks (RNNs).

A recurrent neural network or RNN is a deep neural network trained on sequential
or time series data to create a machine learning (ML) model that can make
sequential predictions or conclusions based on sequential inputs.

References:
    - https://www.ibm.com/think/topics/recurrent-neural-networks#763338458
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

from .core import (
    GRU,
    GRUCell,
    LSTM,
    LSTMCell,
    RNN,
    RNNBase,
    RNNCell,
    RNNCellBase,
)
