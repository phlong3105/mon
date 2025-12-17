#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ZS-N2N model for zero-shot image denoising.

References:
    - Paper: "Zero-Shot Noise2Noise: Efficient Image Denoising without any Data," CVPR 2023.
    - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA
"""

__all__ = [
    "ZSN2N",
]

from .model import ZSN2N
