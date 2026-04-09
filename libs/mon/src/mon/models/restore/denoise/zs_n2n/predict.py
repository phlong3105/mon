#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for ZS-N2N models.
"""

from __future__ import annotations

__all__ = [
    "ZS_N2N_Predictor",
]

from typing import Any

from typing_extensions import override

from mon.core import K, Path, PREDICTORS, TimeProfiler
from mon.dataset import transform as T
from mon.runners import Predictor
from .model import ZS_N2N

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

@PREDICTORS.register(name="zs_n2n")
class ZS_N2N_Predictor(Predictor):
    """Predictor for ZS-N2N models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device

        model = ZS_N2N(device=device, **config.model)
        model = model.to(device)
        model.train()
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
        epochs = config.epochs

        # 1. Prepare inputs
        timers.preprocess.tick()
        image = datapoint["image"]
        image = image.to(device)
        timers.preprocess.tock()

        # 2. Inference
        timers.infer.tick()
        outputs = self.model.fit(
            image=image,
            epochs=epochs,
            reset_weights=True,
            optimizer=config.optimizer,
            scheduler=config.lr_scheduler,
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
            keys=["restored"],
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
        pass

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
