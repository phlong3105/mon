#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for RetinexNet models.
"""

from __future__ import annotations

__all__ = [
    "RetinexNet_Predictor",
]

from typing import Any

import torch
from tensordict import TensorDict
from typing_extensions import override

from mon.core import K, MODELS, Path, PREDICTORS, Size, TimeProfiler
from mon.dataset import transform as T
from mon.runners import Predictor
# noinspection PyUnusedImports
from .model import retinexnet

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

@PREDICTORS.register(name="retinexnet")
class RetinexNet_Predictor(Predictor):
    """Predictor for RetinexNet models."""

    # --- Lifecycle & Initialization ---
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
    def _predict_step(self, datapoint: TensorDict, timers: TimeProfiler) -> TensorDict:
        """Predict the output of the model for a single data point.

        Args:
            datapoint (TensorDict): The dictionary containing the data point
                to predict.
            timers (TimeProfiler): The time profiler to record timing information
                during prediction.

        Returns:
            TensorDict: The dictionary containing the prediction results.
        """
        device = self.device

        # 1. Prepare inputs
        timers.preprocess.tick()
        datapoint = datapoint.to(device)
        timers.preprocess.tock()

        # 2. Inference
        timers.infer.tick()
        outputs = self.model(data=datapoint, decom=False, save_debug=self.save_debug)
        timers.infer.tock()

        return outputs

    # --- Output ---
    @override
    def _save(self, datapoint: TensorDict, outputs: TensorDict):
        """Save the main prediction results to a file.

        Args:
            datapoint (TensorDict): The dictionary containing the input data.
            outputs (TensorDict): The dictionary containing the main
                prediction results. Each key in the dictionary is a batch of
                prediction results.
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
    def _save_debug(self, datapoint: TensorDict, outputs: TensorDict):
        """Save debugging results for visualization.

        Args:
            datapoint (TensorDict): The dictionary containing the input data.
            outputs (TensorDict): The dictionary containing the main
                prediction results. Each key in the dictionary is a batch of
                prediction results.
        """
        self._save_batch_image(
            keys=["R", "L", "L_delta"],
            datapoint=datapoint,
            outputs=outputs,
            dirname=K.DEBUG_DIR,
            subdirname="",
            use_stem=True,
        )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
