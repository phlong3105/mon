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

from mon.core import (
    K,
    Path,
    resolve_project_root,
    RunMode,
    Size,
    Task,
    TimeProfiler,
)
from mon.dataset import transform as T
from mon.runners import Predictor
from .model import colie

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

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

        # 1. Prepare inputs
        timers.preprocess.tick()
        image = datapoint["image"]
        image = image.to(device)
        timers.preprocess.tock()

        # 3. Inference
        timers.infer.tick()
        model = colie(device=device, **config.model)
        outputs = model(
            image=image,
            epochs=config.epochs,
            E=config.loss.E,
            optimizer=config.optimizer,
            save_debug=self.save_debug,
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
            self._save_image(outputs["enhanced"][i:i+1], size, path, dirname=K.PRED_DIR)

    @override
    def _save_debug(self, outputs: dict[str, Any], meta: list[dict[str, Any]]):
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
            debug_images = {
                "image_i": outputs["image_i"][i:i+1],
                "image_i_res": outputs["image_i_res"][i:i+1],
                "image_i_fixed": outputs["image_i_fixed"][i:i+1],
                "image_r": outputs["image_r"][i:i+1],
            }
            for stem, image in debug_images.items():
                self._save_image(image, size, path, dirname=K.DEBUG_DIR, stem=stem)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

def main():
    """Unit test for CoLIE_Predictor."""
    predictor = CoLIE_Predictor.from_cli(
        prompt=True,
        root=resolve_project_root(current_dir),
        config_file="colie.yaml",
        task=Task.ENHANCE,
        mode=RunMode.PREDICT,
        arch="colie",
        model="colie",
        device="auto",
        save=True,
        exist_ok=True,
        verbose=True,
    )
    predictor.predict()


if __name__ == "__main__":
    pass

# endregion
