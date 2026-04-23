#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for CoLIE models.
"""

from __future__ import annotations

__all__ = [
    "CoLIE_Predictor",
]

from tensordict import TensorDict
from typing_extensions import override

from mon.core import K, Path, PREDICTORS, TimeProfiler
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

        # Update model's hidden_dim with eval_imgsz
        self.config.model.hidden_dim = self.config.imgsz.h

    # --- Prediction ---
    @override
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

        # 3. Inference
        timers.infer.tick()
        model = colie(
            device=device,
            optimizer=config.optimizer,
            **config.model
        ).to(device)
        outputs = model(
            data=datapoint,
            epochs=config.epochs,
            E=config.loss.E,
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
