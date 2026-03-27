#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ZS-N2N.

This package contains ZS-N2N model implementations, pre-trained weights, and
utilities for training and inference.

References:
    - Paper: "Zero-Shot Noise2Noise: Efficient Image Denoising without any
      Data," CVPR 2023.
    - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA
"""

from __future__ import annotations

from .model import *
from .predict import *
