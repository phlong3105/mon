#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for Zero-DCE and Zero-DCE++ models.
"""

from __future__ import annotations

__all__ = [
    "ZeroDCE_Predictor",
]

from typing_extensions import override

from mon.core import MODELS, Path, Size, TimeProfiler
from mon.dataset import transform as T
from mon.metrics import benchmark
from mon.runners import Predictor
# noinspection PyUnusedImports
from .model import zero_dce, zero_dce_pp

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

        model = MODELS.build(**config.model | { "weights": weights})
        model = model.to(device)
        model.eval()
        self._model = model

    @override
    def _init_transforms(self):
        """Initialize ``self._transforms`` attribute."""
        config = self.config
        imgsz = Size.from_value(config.eval_imgsz)

        scale_factor = config.model.get("scale_factor")
        if scale_factor:
            imgsz = Size(height=imgsz.h // scale_factor, width=imgsz.w // scale_factor)

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
        save_debug = config.save_debug

        # 1. Prepare inputs
        timers.preprocess.tick()
        image = datapoint["image"]
        image = image.to(device)
        timers.preprocess.tock()

        # 2. Inference
        timers.infer.tick()
        outputs = self.model(image, save_debug=save_debug)
        timers.infer.tock()

        return outputs

    # --- Utilities ---
    @override
    def _benchmark(self):
        """Run the benchmark for the model."""
        config = self.config
        imgsz = Size.from_value(config.eval_imgsz)

        scale_factor = config.model.get("scale_factor")
        if scale_factor:
            imgsz = Size(height=imgsz.h // scale_factor, width=imgsz.w // scale_factor)

        if self.benchmark:
            benchmark(self.model, imgsz=imgsz)

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
