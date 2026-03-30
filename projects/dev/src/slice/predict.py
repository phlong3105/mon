#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for SLICE models.
"""

from __future__ import annotations

__all__ = [
    "SLICE_Predictor",
]

from typing import Any

import torch
from typing_extensions import override

from mon.core import (
    K,
    Path,
    PREDICTORS,
    resolve_project_root,
    RunMode,
    Size,
    Task,
    TimeProfiler,
)
from mon.dataset import transform as T
from mon.runners import Predictor
# noinspection PyUnusedImports
from .model import slice

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

@PREDICTORS.register(name="slice")
class SLICE_Predictor(Predictor):
    """Predictor for SLICE models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.weights or config.finetune

        model = slice(**config.model | {"weights": weights})
        model = model.to(device)
        model.eval()
        self._model = model

    @override
    def _init_transforms(self):
        """Initialize ``self._transforms`` attribute."""
        config = self.config

        transforms = T.Compose([
            T.Normalize(normalization="min_max"),
            T.ToTensorV2(transpose_mask=True),
        ])

        if config.eval_resize:
            imgsz = Size.from_value(config.eval_imgsz)
            resize = T.ResizeDivisibleBy(height=imgsz.h, width=imgsz.w, divisor=32)
            transforms = resize + transforms

        self._transforms = transforms

    # --- Prediction ---
    @override
    @torch.inference_mode()
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
        use_depth = config.model.use_depth

        # 1. Prepare inputs
        timers.preprocess.tick()
        image = datapoint["image"]
        image = image.to(device)
        depth = datapoint.get("depth", None)
        depth = depth.to(device) if use_depth and depth is not None else None
        t = torch.tensor([0, config.eval_T]).float().to(device)
        timers.preprocess.tock()

        # 2. Inference
        timers.infer.tick()
        outputs = self.model(image=image, depth=depth, t=t, save_debug=self.save_debug)
        timers.infer.tock()

        return outputs

    # --- Output ---
    @override
    def _save(self, datapoint: dict[str, Any], outputs: dict[str, Any]):
        """Save the main prediction results to a file.

        Args:
            datapoint (dict): The dictionary containing the input data.
            outputs (dict): The dictionary containing the main prediction results.
                Each key in the dictionary is a batched of prediction results.
        """
        self._save_batch_image(
            keys=["enhanced"],
            datapoint=datapoint,
            outputs=outputs,
            dirname=K.PRED_DIR,
            subdirname="",
            use_stem=False,
        )

    @override
    def _save_debug(self, datapoint: dict[str, Any], outputs: dict[str, Any]):
        """Save debugging results for visualization.

        Args:
            datapoint (dict): The dictionary containing the input data.
            outputs (dict): The dictionary containing the main prediction results.
                Each key in the dictionary is a batched of prediction results.
        """
        pass

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

def main():
    """Unit test for SLICE_Predictor."""
    predictor = SLICE_Predictor.from_cli(
        prompt=True,
        root=resolve_project_root(current_dir),
        config_file="slice_sice_me.yaml",
        task=Task.LLE,
        mode=RunMode.PREDICT,
        arch="slice",
        model="slice",
        device="auto",
        save=True,
        save_debug=True,
        exist_ok=True,
        verbose=True,
    )
    predictor.predict()

if __name__ == "__main__":
    pass

# endregion
