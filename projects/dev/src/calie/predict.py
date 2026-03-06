#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for CALIE models.
"""

from __future__ import annotations

__all__ = [
    "CALIE_Predictor",
]

from typing_extensions import override

from mon.core import MODELS, Path, Size, TimeProfiler
from mon.dataset import transform as T
from mon.runners import Predictor
# noinspection PyUnusedImports
from .model import calie_ffsiren, calie_siren

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

class CALIE_Predictor(Predictor):
    """Predictor for CALIE models."""

    # --- Properties ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device

        model = MODELS.build(device=device, **config.model)
        model = model.to(device)
        model.train()
        self._model = model

    @override
    def _init_transforms(self):
        """Initialize ``self._transforms`` attribute."""
        self._transforms = T.Compose([
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
        epochs = config.epochs
        E = config.loss.E
        save_debug = config.save_debug

        # 1. Prepare inputs
        timers.preprocess.tick()
        image = datapoint["image"]
        image = image.to(device)
        timers.preprocess.tock()

        # 2. Inference
        timers.infer.tick()
        outputs = self.model(
            image=image,
            epochs=epochs,
            batch_size=8,
            E=E,
            save_debug=save_debug,
        )
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
        self._save_image(outputs["enhanced"], size, path)

    @override
    def _save_debug(self, outputs: dict, meta: dict):
        """Save debugging results for visualization.

        Args:
            outputs (dict): The dictionary containing the debugging results.
            meta (dict): The dictionary containing the metadata.
        """
        path = Path(meta["path"])
        size = Size.from_value(meta["imgsz"])
        self._save_image(outputs["image_i"], size, path, "image_i")
        self._save_image(outputs["image_i_res"], size, path, "image_i_res")
        self._save_image(outputs["image_i_fixed"], size, path, "image_i_fixed")
        self._save_image(outputs["image_r"], size, path, "image_r")

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
