#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for image feature priors.

This module implements boundary-aware image priors used in computer vision and
image processing tasks. These priors help in enhancing image quality by
focusing on important regions of the image, such as edges and boundaries.
"""

__all__ = [
    "BoundaryAwarePrior",
    "boundary_aware_prior",
]

import kornia
import torch
import torch.nn as nn


def boundary_aware_prior(
    image      : torch.Tensor,
    eps        : float = 0.05,
    as_gradient: bool  = False,
    normalized : bool  = False,
) -> torch.Tensor:
    """Gets the boundary prior from an RGB or grayscale image.

    Args:
        image (torch.Tensor): RGB or grayscale image.
        eps (float): Threshold to remove weak edges. Defaults to 0.05.
        as_gradient (bool): If True, returns the gradient image instead of binary
            boundary. Defaults to False.
        normalized (bool): L1 norm of the kernel is set to 1 if True. Defaults
            to False.
    
    Returns:
        torch.Tensor: Boundary prior as binary map or gradient image.
    """
    image    = image.to(torch.float32)
    gradient = kornia.filters.sobel(image, normalized=normalized, eps=1e-6)
    g_max    = torch.max(gradient)
    gradient = gradient / g_max
    boundary = (gradient > eps).float()
    # Return boundary, gradient
    if as_gradient:
        return gradient
    else:
        return boundary


class BoundaryAwarePrior(nn.Module):
    """A class to get the boundary prior from an RGB or grayscale image.
    
    Attributes:
        eps (float): Threshold to remove weak edges.
        as_gradient (bool): If True, returns the gradient image instead of
            binary boundary.
        normalized (bool): L1 norm of the kernel is set to 1 if True.
    """
    
    def __init__(
        self,
        eps        : float = 0.05,
        as_gradient: bool  = False,
        normalized : bool  = False
    ):
        """Initializes the BoundaryAwarePrior instance.
        
        Args:
            eps (float): Threshold to remove weak edges. Defaults to 0.05.
            as_gradient (bool): If True, returns the gradient image instead of
                binary boundary. Defaults to False.
            normalized (bool): L1 norm of the kernel is set to 1 if True.
                Defaults to False.
        """
        super().__init__()
        self.eps        = eps
        self.as_gradient = as_gradient
        self.normalized = normalized
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Computes the boundary prior from the input image.
        
        Args:
            image (torch.Tensor): RGB or grayscale image.
            
        Returns:
            torch.Tensor: Boundary prior as binary map or gradient image.
        """
        return boundary_aware_prior(image, self.eps, self.as_gradient, self.normalized)
