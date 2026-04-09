#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for CoLIE models.
"""

from __future__ import annotations

__all__ = [
    "CoLIE_Predictor",
]

from typing import Any

from typing_extensions import override

from mon.core import K, Path, PREDICTORS, TimeProfiler
from mon.dataset import transform as T
from mon.runners import Predictor
from .model import colie

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

@PREDICTORS.register(name="colie")
class CoLIE_Predictor(Predictor):
    """Predictor for CoLIE models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        self._model = None

    @override
    def _init_transforms(self):
        """Initialize ``self._transforms`` attribute."""
        self._transforms = T.Compose([
            T.Normalize(normalization="min_max"),
            T.ToTensorV2(transpose_mask=True),
        ])

    # --- Prediction ---
    @override
    def _predict_step(self, datapoint: dict[str, Any], timers: TimeProfiler) -> dict[str, Any]:
        """Predict the output of the model for a single data point.

        Args:
            datapoint (dict[str, Any]): The dictionary containing the data point
                to predict.
            timers (TimeProfiler): The time profiler to record timing information
                during prediction.

        Returns:
            dict[str, Any]: The dictionary containing the prediction results.
        """
        config = self.config
        device = self.device

        # 1. Prepare inputs
        timers.preprocess.tick()
        image = datapoint["image"]
        image = image.to(device)
        timers.preprocess.tock()

        # 3. Inference
        timers.infer.tick()
        model = colie(device=device, optimizer=config.optimizer, **config.model)
        outputs = model(
            data={
                "image": image,
                "epochs": config.epochs,
                "E": config.loss.E,
            },
            save_debug=self.save_debug,
        )
        timers.infer.tock()

        return outputs

    # --- Output ---
    @override
    def _save(self, datapoint: dict[str, Any], outputs: dict[str, Any]):
        """Save the main prediction results to a file.

        Args:
            datapoint (dict[str, Any]): The dictionary containing the input data.
            outputs (dict[str, Any]): The dictionary containing the main
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
    def _save_debug(self, datapoint: dict[str, Any], outputs: dict[str, Any]):
        """Save debugging results for visualization.

        Args:
            datapoint (dict[str, Any]): The dictionary containing the input data.
            outputs (dict[str, Any]): The dictionary containing the main
                prediction results. Each key in the dictionary is a batch of
                prediction results.
        """
        self._save_batch_image(
            keys=["image_i", "image_i_res", "image_i_fixed", "image_r"],
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
