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

from mon.core import (
    Path,
    resolve_project_root,
    RunMode,
    Size,
    Task,
    TimeProfiler,
)
from mon.dataset import transform as T
from mon.runners import Predictor
from .model import ZS_N2N

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

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
    def _save(self, outputs: dict[str, Any], meta: list[dict[str, Any]]):
        """Save the main prediction results to a file.

        Args:
            outputs (dict): The dictionary containing the main prediction results.
                Each key in the dictionary is a batched of prediction results.
            meta (list[dict]): The list of dictionaries containing the metadata
                for each data point.
        """
        for i, meta_i in enumerate(meta):
            path = Path(meta_i["path"])
            size = Size.from_value(meta_i["imgsz"])
            self._save_image(outputs["restored"][i:i+1], size, path)

    @override
    def _save_debug(self, outputs: dict[str, Any], meta: list[dict[str, Any]]):
        """Save the main prediction results to a file.

        Args:
            outputs (dict): The dictionary containing the main prediction results.
                Each key in the dictionary is a batched of prediction results.
            meta (list[dict]): The list of dictionaries containing the metadata
                for each data point.
        """
        pass

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

def main():
    """Unit test for ZSN2N_Predictor."""
    predictor = ZS_N2N_Predictor.from_cli(
        prompt=True,
        root=resolve_project_root(current_dir),
        config_file="zsn2n.yaml",
        task=Task.RESTORE,
        mode=RunMode.PREDICT,
        arch="zsn2n",
        model="zsn2n",
        save=True,
        exist_ok=True,
        verbose=True,
    )
    predictor.predict()


if __name__ == "__main__":
    pass

# endregion
