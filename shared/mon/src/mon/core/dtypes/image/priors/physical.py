#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for image physical priors.

This module implements physical-based image priors used in computer vision
and image processing tasks. These priors help in enhancing image quality by
modeling physical phenomena such as atmospheric scattering and light absorption.
"""

__all__ = [
    "atmospheric_point_spread_function",
    "atmospheric_prior",
]

import numpy as np
import torch
import torch.nn.functional as F

from .statistical import dark_channel_prior_paper


def atmospheric_point_spread_function(
    image: torch.Tensor,
    q    : float = 0.2,
    T    : float = 1.2,
    k    : float = 0.5,
) -> torch.Tensor:
    """Gets the atmospheric point spread function (APSF) from an RGB image.
    
    References:
        - Code: https://github.com/jinyeying/night-enhancement/blob/main/glow_rendering_code/repro_ICCV2007_Fig5.m
        
    Args:
        image (torch.Tensor): An RGB image as a torch.Tensor of shape (B, 3, H, W)
            with pixel values in [0.0, 1.0].
        q (float): Forward scattering param.
            - 0.00-0.20: air
            - 0.20-0.70: aerosol
            - 0.70-0.80: haze
            - 0.80-0.85: mist
            - 0.85-0.90: fog
            - 0.90-1.00: rain
            Defaults to 0.2.
        T (float): Optical thickness. Possibly: [0.7, 1.2, 4]. According to
            Narasimhan in CVPR03 paper:
            T = sigma * R (extinction coefficient * distance or depth),
            which is the same \beta d in haze modelling. Defaults to 1.2.
        k (float): Conversion param for kernel. Defaults to 0.5.
    
    Returns:
        torch.Tensor: A torch.Tensor of shape (B, 3, H, W) with pixel values in
            [0.0, 1.0], representing the APSF applied image.
    """
    from scipy.special import gamma
    
    def A(p: float, sigma: float):
        return np.sqrt(sigma ** 2 * gamma(1 / p) / gamma(3 / p))
    
    p       = k * T        # Eq (9)
    sigma   = (1 - q) / q  # Eq (1)
    # Generate APSF kernel
    x       = torch.linspace(-6, 6, 100)
    XX, YY  = torch.meshgrid(x, x, indexing='ij')
    A_val   = A(p, sigma)
    APSF2D  = torch.exp(-((XX ** 2 + YY ** 2) ** (p / 2)) / abs(A_val) ** p) / (2 * gamma(1 + 1 / p) * A_val) ** 2
    APSF2D /= torch.sum(APSF2D)
    # Apply convolution
    kernel  = APSF2D.unsqueeze(0).unsqueeze(0)   # Shape: (1, 1, H, W)
    kernel  = kernel.repeat(3, 1, 1, 1)          # Shape: (3, 1, H, W) for RGB channels
    apsf    = F.conv2d(image, kernel, padding="same", groups=3)
    apsf    = torch.clamp(apsf, 0, 1)  # Ensure valid pixel range
    return apsf


def atmospheric_prior(image: np.ndarray, ksize: int = 15, p: float = 0.0001) -> np.ndarray:
    """Gets the atmospheric light in an RGB image.

    Args:
        image (np.ndarray): An RGB image as a numpy.ndarray of shape (3, H, W)
            with pixel values in [0, 255].
        ksize (int): Window size for dark channel prior. Defaults to 15.
        p (float): Percentage of brightest pixels in the dark channel to be
            considered for atmospheric light estimation. Defaults to 0.0001.
            
    Returns:
        np.ndarray: A numpy.ndarray of shape (3,) representing the estimated
            atmospheric light for each RGB channel.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"``image`` must be numpy.ndarray, got {type(image)}.")
    
    image      = image.transpose(1, 2, 0)
    # Reference CVPR09, 4.4
    dark       = dark_channel_prior_paper(image=image, ksize=ksize)
    m, n       = dark.shape
    flat_i     = image.reshape(m * n, 3)
    flat_dark  = dark.ravel()
    search_idx = (-flat_dark).argsort()[:int(m * n * p)]  # find top M * N * p indexes
    # Return the highest intensity for each channel
    return np.max(flat_i.take(search_idx, axis=0), axis=0)
