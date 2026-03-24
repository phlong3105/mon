#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for TensorMoG models.
"""

from __future__ import annotations

__all__ = [
    "TensorMOG_Predictor",
]

from typing import Any

from typing_extensions import override

from mon.core import K, MODELS, Path, Size, TimeProfiler
from mon.dataset import transform as T
from mon.runners import Predictor

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

class TensorMOG_Predictor(Predictor):
    """Predictor for TensorMoG models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device

        imgsz = config.eval_imgsz
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
        """Initialize ``self._transforms`` attribute."""
        config = self.config

        imgsz = Size.from_value(config.eval_imgsz)
        transforms = T.Compose([
            T.ResizeDivisibleBy(height=imgsz.h, width=imgsz.w, divisor=32),
            T.Normalize(normalization="min_max"),
            T.ToTensorV2(transpose_mask=True),
        ])
        self._transforms = transforms

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
        outputs = self.model(image=image)
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
            self._save_image(outputs["background"][i:i+1], size, path, dirname=K.PRED_DIR)

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
                "foreground": outputs["image_i"][i:i+1],
            }
            for stem, image in debug_images.items():
                self._save_image(image, size, path, dirname=K.DEBUG_DIR, stem=stem)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
