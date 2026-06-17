#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for TensorMoG models.
"""

from __future__ import annotations

__all__ = [
    "TensorMOG_Predictor",
]

from tensordict import TensorDict
from typing_extensions import override

from mon.core import K, MODELS, Path, PREDICTORS, TimeProfiler
from mon.runners import Predictor
from mon.dataset import transform as T

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

@PREDICTORS.register(name="tensormog")
class TensorMOG_Predictor(Predictor):
    """Predictor for TensorMoG models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device

        imgsz = config.imgsz
        model = MODELS.build(
            **config.model | {
                "height": imgsz.h,
                "width": imgsz.w,
                "device": device,
            }
        )
        model = model.to(device)
        self._model = model

    @override
    def _init_transforms(self):
        """Initialize ``self._transforms`` attribute for pre-processing the
        input data.
        """
        config = self.config

        transforms = T.Compose(
            [
                T.Normalize(normalization="min_max"),
                T.ToTensorV2(transpose_mask=True),
            ],
            is_check_shapes=False,
        )

        if config is not None:
            if config.use_resize:
                imgsz = config.imgsz
                resize = T.ResizeDivisibleBy(height=imgsz.h, width=imgsz.w, divisor=32)
                transforms = resize + transforms

        self._transforms = transforms

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
        device = self.device

        # 1. Prepare inputs
        timers.preprocess.tick()
        datapoint = datapoint.to(device)
        timers.preprocess.tock()

        # 3. Inference
        timers.infer.tick()
        outputs = self.model(data=datapoint, save_debug=self.save_debug)
        timers.infer.tock()

        return outputs

    # --- Output ---
    @override
    def _save(self, datapoint: TensorDict, outputs: TensorDict):
        """Save the main prediction results to a file.

        Args:
            datapoint (TensorDict): The dictionary containing the input data.
            outputs (TensorDict): The dictionary containing the main results.
                Each key in the dictionary is a batch of results.
        """
        self._save_batch_image(
            keys=["background"],
            datapoint=datapoint,
            outputs=outputs,
            dirname=K.IMAGES_DIR,
            subdirname="",
            use_stem=False,
        )

    @override
    def _save_debug(self, datapoint: TensorDict, outputs: TensorDict):
        """Save debugging results for visualization.

        Args:
            datapoint (TensorDict): The dictionary containing the input data.
            outputs (TensorDict): The dictionary containing the main results.
                Each key in the dictionary is a batch of results.
        """
        self._save_batch_image(
            keys=["foreground"],
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
