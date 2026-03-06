#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Script.

This script provides a CLI for running CLODE prediction on a given dataset.

References:
    - Paper: "Continuous Exposure Learning for Low-light Image Enhancement using
      Neural ODEs," ICLR 2025.
    - Code: https://github.com/dgjung0220/CLODE
"""

from __future__ import annotations

__all__ = [
    "ZeroDCE_Predictor",
]

import torch
from typing_extensions import override

from mon.core import Path, Size, TimeProfiler
from mon.dataset import transform as T
from mon.runners import Predictor
from .model import clode

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

class ZeroDCE_Predictor(Predictor):
    """Predictor for Zero-DCE models."""

    # --- Properties ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.weights or config.finetune

        model = clode(**config.model | { "weights": weights})
        model = model.to(device)
        model.eval()
        self._model = model

    @override
    def _init_transforms(self):
        """Initialize ``self._transforms`` attribute."""
        config = self.config
        imgsz = Size.from_value(config.eval_imgsz)

        self._transforms = T.Compose([
            T.ResizeDivisibleBy(height=imgsz.h, width=imgsz.w, divisor=32),
            T.Normalize(normalization="min_max"),
            T.ToTensorV2(transpose_mask=True),
        ])

    # --- Prediction ---
    @override
    def _predict_step(self, datapoint: dict, timers: TimeProfiler) -> dict:
        """Predict the output of the model for a single data point.

        Args:
            datapoint (dict): The dictionary containing the data point to predict.
            timers (TimeProfiler): The time profiler to record timing information
                during prediction.

        Returns:
            dict: The dictionary containing the prediction results.
        """
        config = self.config
        device = self.device

        # 1. Prepare inputs
        timers.preprocess.tick()
        image = datapoint["image"]
        image = image.to(device)
        time_eval = torch.tensor([0, config.T]).float().to(device)
        timers.preprocess.tock()

        # 2. Inference
        timers.infer.tick()
        outputs = self.model(image, eval_time=time_eval, inference=True)
        timers.infer.tock()

        return outputs

    # --- Utilities ---
    @override
    def _save(self, outputs: dict, meta: dict):
        """Save the main prediction results to a file.

        Args:
            outputs (dict): The dictionary containing the main prediction results.
            meta (dict): The dictionary containing the metadata.
        """
        path = Path(meta["path"])
        size = Size.from_value(meta["imgsz"])
        self._save_image(path, outputs["enhanced"], size)

    @override
    def _save_debug(self, outputs: dict, meta: dict):
        """Save debugging results for visualization.

        Args:
            outputs (dict): The dictionary containing the debugging results.
            meta (dict): The dictionary containing the metadata.
        """
        pass

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
