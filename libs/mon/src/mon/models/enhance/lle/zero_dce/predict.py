#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for Zero-DCE and Zero-DCE++ models.
"""

from __future__ import annotations

__all__ = [
    "ZeroDCE_Predictor",
]

import torch
from tensordict import TensorDict
from typing_extensions import override

from mon import Strategy
from mon.core import K, MODELS, Path, PREDICTORS, Size, SizeLike, TimeProfiler
from mon.dataset import transform as T
from mon.runners import Predictor
# noinspection PyUnusedImports
from .model import zero_dce, zero_dce_pp

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

@PREDICTORS.register(name="zero_dce")
class ZeroDCE_Predictor(Predictor):
    """Predictor for Zero-DCE models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.weights or config.finetune

        model = MODELS.build(**config.model | { "weights": weights })
        model = model.to(device)
        model.eval()
        self._model = model

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
        timers.preprocess.tock()

        # 2. Inference
        timers.infer.tick()
        outputs = self.model(
            data=datapoint,
            use_patch=(config.strategy == Strategy.PATCH),
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
