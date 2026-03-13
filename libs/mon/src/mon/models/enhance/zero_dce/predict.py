#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for Zero-DCE and Zero-DCE++ models.
"""

from __future__ import annotations

__all__ = [
    "ZeroDCE_Predictor",
]

from typing import Any

import torch
from typing_extensions import override

from mon.core import (
    MODELS,
    Path,
    resolve_project_root,
    RunMode,
    Size,
    Task,
    TimeProfiler,
)
from mon.dataset import transform as T
from mon.metrics import benchmark
from mon.runners import Predictor
# noinspection PyUnusedImports
from .model import zero_dce, zero_dce_pp

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

class ZeroDCE_Predictor(Predictor):
    """Predictor for Zero-DCE models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.weights or config.finetune

        model = MODELS.build(**config.model | { "weights": weights})
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
            imgsz = Size.from_value(config.eval_imgsz)
            scale_factor = config.model.get("scale_factor")
            if scale_factor:
                imgsz = Size(height=imgsz.h // scale_factor, width=imgsz.w // scale_factor)
            resize = T.ResizeDivisibleBy(height=imgsz.h, width=imgsz.w, divisor=32)
            transforms = resize + transforms

        self._transforms = transforms

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
        save_debug = config.save_debug

        # 1. Prepare inputs
        timers.preprocess.tick()
        image = datapoint["image"]
        image = image.to(device)
        timers.preprocess.tock()

        # 2. Inference
        timers.infer.tick()
        outputs = self.model(image, save_debug=save_debug)
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

    # --- Utilities ---
    @override
    def benchmark(self):
        """Run the benchmark for the model."""
        config = self.config
        imgsz = config.eval_imgsz

        scale_factor = config.model.get("scale_factor")
        if scale_factor:
            imgsz = Size(height=imgsz.h // scale_factor, width=imgsz.w // scale_factor)

        if self.benchmark:
            benchmark(self.model, imgsz=imgsz)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

def main():
    predictor = ZeroDCE_Predictor.from_cli(
        prompt=True,
        root=resolve_project_root(current_dir),
        config_file="zero_dce_sice_me.yaml",
        task=Task.ENHANCE,
        mode=RunMode.PREDICT,
        arch="zero_dce",
        model="zero_dce",
        device="auto",
        save=True,
        save_debug=True,
        exist_ok=True,
        verbose=True,
    )
    predictor.predict()

if __name__ == "__main__":
    pass

# endregion
