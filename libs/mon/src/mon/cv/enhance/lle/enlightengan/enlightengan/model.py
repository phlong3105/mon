#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements EnlightenGAN model for low-light image enhancement.

References:
    - Paper: "EnlightenGAN: Deep Light Enhancement without Paired Supervision," TIP 2021.
    - Code: https://github.com/arsenyinfo/EnlightenGAN-inference
"""

__all__ = [
    "EnlightenOnnxModel",
]

import os
from typing import Any, Union

import box
import numpy as np
from onnxruntime import InferenceSession

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


def get_relative_path(root, *args):
    return os.path.join(os.path.dirname(root), *args)


@MODELS.register(name="enlightengan", arch="enlightengan")
class EnlightenOnnxModel(nn.ModelMetadataMixin):
    """EnlightenGAN model for low-light image enhancement.
    
    References:
        - Paper: "EnlightenGAN: Deep Light Enhancement without Paired Supervision," TIP 2021.
        - Code: https://github.com/arsenyinfo/EnlightenGAN-inference
    """
    
    _arch     : str          = "enlightengan"
    _name     : str          = "enlightengan"
    _tasks    : list[Task]   = [Task.LLE]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box({
        "custom": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/enlightengan/enlightengan/custom/enlightengan.onnx",
            "num_classes": None,
        },
    })
    
    def __init__(
        self,
        model  : Union[bytes, str, None] = None,
        weights: Any = "custom",
    ):
        _, path, _   = self.parse_weights(weights, None)
        self.model   = model
        self.weights = Path(path)
        self.graph   = None
    
    def initialize(self):
        self.graph = InferenceSession(
            self.model or self.weights or str(ROOT_DIR / "zoo/cv/enhance/lle/enlightengan/custom/enlightengan.onnx"),
            providers=["AzureExecutionProvider", "CPUExecutionProvider"]
        )
        
    def _pad(self, img):
        h, w, _    = img.shape
        block_size = 16
        min_height = (h // block_size + 1) * block_size
        min_width  = (w // block_size + 1) * block_size
        img        = np.pad(img, ((0, min_height - h), (0, min_width - w), (0, 0)), mode="constant", constant_values=0)
        return img, (h, w)

    def _preprocess(self, img):
        if len(img.shape) != 3:
            raise ValueError(f"Incorrect shape: expected 3, got {len(img.shape)}")
        return np.expand_dims(np.transpose(img, (2, 0, 1)).astype(np.float32) / 255., 0)

    def predict(self, img):
        padded, (h, w) = self._pad(img)
        image_numpy,   = self.graph.run(["output"], {"input": self._preprocess(padded)})
        image_numpy    = (np.transpose(image_numpy[0], (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy    = np.clip(image_numpy, 0, 255)
        return image_numpy.astype("uint8")[:h, :w, :]
