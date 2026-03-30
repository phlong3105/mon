#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for SALEO models.
"""

from __future__ import annotations

__all__ = [
    "SALEO_Predictor",
]

from typing import Any

from typing_extensions import override

from mon.core import (
    K,
    MODELS,
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
from .model import saleo_ffsiren, saleo_siren

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

@PREDICTORS.register(name="saleo")
class SALEO_Predictor(Predictor):
    """Predictor for SALEO models."""

    # --- Lifecycle & Initialization ---
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
        depth = datapoint["depth"]
        depth = depth.to(device) if depth is not None else None
        timers.preprocess.tock()

        # 2. Inference
        timers.infer.tick()
        outputs = self.model(
            image=image,
            depth=depth,
            epochs=epochs,
            batch_size=8,
            E=E,
            color_func="hsv",
            save_debug=save_debug,
        )
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

def main():
    """Unit test for SALEO_Predictor."""
    predictor = SALEO_Predictor.from_cli(
        prompt=True,
        root=resolve_project_root(current_dir),
        config_file="saleo_ffsiren.yaml",
        task=Task.LLE,
        mode=RunMode.PREDICT,
        arch="saleo",
        model="saleo_ffsiren",
        save=True,
        exist_ok=True,
        verbose=True,
    )
    predictor.predict()


if __name__ == "__main__":
    pass

# endregion
