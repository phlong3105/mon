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
    "CLODE_Predictor",
]

from typing import Any

import torch
from tensordict import TensorDict
from typing_extensions import override

from mon.core import K, Path, PREDICTORS, TimeProfiler
from mon.dataset import transform as T
from mon.runners import Predictor
from .model import clode

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

@PREDICTORS.register(name="clode")
class CLODE_Predictor(Predictor):
    """Predictor for CLODE models."""

    # --- Lifecycle & Initialization ---
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

        transforms = T.Compose([
            T.Normalize(normalization="min_max"),
            T.ToTensorV2(transpose_mask=True),
        ])

        if config.eval_resize:
            imgsz = config.eval_imgsz
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
        config = self.config
        device = self.device

        # 1. Prepare inputs
        timers.preprocess.tick()
        datapoint = datapoint.to(device)
        eval_time = torch.tensor([0, config.T]).float().to(device)
        timers.preprocess.tock()

        # 2. Inference
        timers.infer.tick()
        outputs = self.model(
            data=datapoint,
            eval_time=eval_time,
            inference=True,
            save_debug=self.save_debug,
        )
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
        pass

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
