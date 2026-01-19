#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Recurrent neural networks (RNNs).

This module provides various recurrent neural network (RNN) components.
"""

from __future__ import annotations

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

from torch.nn.modules.rnn import (
    GRU,
    GRUCell,
    LSTM,
    LSTMCell,
    RNN,
    RNNBase,
    RNNCell,
    RNNCellBase,
)


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
