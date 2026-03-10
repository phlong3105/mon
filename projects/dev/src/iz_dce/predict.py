#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for IZ-DCE and IZ-DCE-ODE models.
"""

from __future__ import annotations

__all__ = [
    "IZDCE_Predictor",
    "IZDCE_ODE_Predictor",
]

from typing import Any

import torch
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
# noinspection PyUnusedImports
from .model import iz_dce, iz_dce_ode

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

class IZDCE_Predictor(Predictor):
    """Predictor for IZ-DCE models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.weights or config.finetune

        model = iz_dce(**config.model | { "weights": weights})
        model = model.to(device)
        model.eval()
        self._model = model

    @override
    def _init_transforms(self):
        """Initialize ``self._transforms`` attribute."""
        config = self.config
        imgsz = Size.from_value(config.eval_imgsz)

        self._transforms = T.Compose([
            T.ResizeDivisibleBy(height=imgsz.h, width=imgsz.w, divisor=32),
            T.Normalize(normalization="min_max"),
            T.ToTensorV2(transpose_mask=True),
        ])

    # --- Prediction ---
    @override
    @torch.no_grad()
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
        use_depth = config.model.use_depth

        # 1. Prepare inputs
        timers.preprocess.tick()
        image = datapoint["image"]
        image = image.to(device)
        depth = datapoint.get("depth", None)
        depth = depth.to(device) if use_depth and depth is not None else None
        timers.preprocess.tock()

        # 2. Inference
        timers.infer.tick()
        outputs = self.model(image, depth)
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
            self._save_image(outputs["enhanced"][i:i+1], size, path)

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


class IZDCE_ODE_Predictor(IZDCE_Predictor):
    """Predictor for IZ-DCE-ODE models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.weights or config.finetune

        model = iz_dce_ode(**config.model | { "weights": weights})
        model = model.to(device)
        model.eval()
        self._model = model

    # --- Prediction ---
    @override
    @torch.no_grad()
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
        use_depth = config.model.use_depth

        # 1. Prepare inputs
        timers.preprocess.tick()
        image = datapoint["image"]
        image = image.to(device)
        depth = datapoint.get("depth", None)
        depth = depth.to(device) if use_depth and depth is not None else None
        time_eval = torch.tensor([0, config.T]).float().to(device)
        timers.preprocess.tock()

        # 2. Inference
        timers.infer.tick()
        outputs = self.model(image, depth, time_eval)
        timers.infer.tock()

        return outputs

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

def main():
    """Unit test for IZDCE_Predictor."""
    predictor = IZDCE_Predictor.from_cli(
        prompt=True,
        root=resolve_project_root(current_dir),
        config_file="iz_dce_sice_me.yaml",
        task=Task.ENHANCE,
        mode=RunMode.PREDICT,
        arch="iz_dce",
        model="iz_dce",
        save=True,
        save_debug=True,
        exist_ok=True,
        verbose=True,
    )
    predictor.predict()

if __name__ == "__main__":
    pass

# endregion
