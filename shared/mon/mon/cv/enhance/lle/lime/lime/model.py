#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements LIME model for low-light image enhancement.

References:
    - Paper: "LIME: Low-light Image Enhancement via Illumination Map Estimation,"
      TIP 2006.
    - Code: https://github.com/pvnieo/Low-light-Image-Enhancement
"""

__all__ = [
    "LIME",
]

import box
import numpy as np

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .module import (
    correct_underexposure,
    create_spacial_affinity_kernel,
    fuse_multi_exposure_images,
)

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="lime", arch="lime")
class LIME(ModelMixin):
    """LIME model for low-light image enhancement.

    Args:
        gamma: Gamma correction factor.
        lambda_: Coefficient to balance the terms in the optimization problem.
        dual: Use DUAL method if ``True``, otherwise use LIME method.
        sigma: Spatial standard deviation for spatial affinity based Gaussian weights.
        bc: Parameter for controlling the influence of Mertens's contrast measure.
        bs: Parameter for controlling the influence of Mertens's saturation measure.
        be: Parameter for controlling the influence of Mertens's well exposedness measure.
        eps: Small constant to avoid computation instability.
    
    References:
        - Paper: "LIME: Low-light Image Enhancement via Illumination Map Estimation,"
          TIP 2006.
        - Code: https://github.com/pvnieo/Low-light-Image-Enhancement
    """
    
    arch     : str          = "lime"
    name     : str          = "lime"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.TRADITIONAL]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()

    def __init__(
        self,
        gamma  : float,
        lambda_: float,
        dual   : bool  = False,
        sigma  : int   = 3,
        bc     : float = 1,
        bs     : float = 1,
        be     : float = 1,
        eps    : float = 1e-3
    ):
        super().__init__()
        self.gamma   = gamma
        self.lambda_ = lambda_
        self.dual    = dual
        self.sigma   = sigma
        self.bc      = bc
        self.bs      = bs
        self.be      = be
        self.eps     = eps
    
    def __call__(self, image: np.ndarray, *args, **kwargs):
        # create spacial affinity kernel
        kernel = create_spacial_affinity_kernel(self.sigma)
    
        # correct under-exposure
        im_normalized   = image.astype(float) / 255.0
        under_corrected = correct_underexposure(im_normalized, self.gamma, self.lambda_, kernel, self.eps)
    
        if self.dual:
            # correct overexposure and merge if DUAL method is selected
            inv_im_normalized = 1 - im_normalized
            over_corrected    = 1 - correct_underexposure(inv_im_normalized, self.gamma, self.lambda_, kernel, self.eps)
            # Fuse images
            im_corrected = fuse_multi_exposure_images(im_normalized, under_corrected, over_corrected, self.bc, self.bs, self.be)
        else:
            im_corrected = under_corrected
    
        # convert to 8 bits and returns
        return np.clip(im_corrected * 255, 0, 255).astype("uint8")
