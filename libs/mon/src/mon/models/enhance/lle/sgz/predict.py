#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Runners.

This module provides prediction runner classes for SGZ models.
"""

from __future__ import annotations

__all__ = [
    "SGZ_Predictor",
]

import torch
from tensordict import TensorDict
from typing_extensions import override

from mon.core import K, MODELS, Path, PREDICTORS, Size, SizeLike, TimeProfiler
from mon.dataset import transform as T
from mon.runners import Predictor
# noinspection PyUnusedImports
from .model import sgz

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

@PREDICTORS.register(name="sgz")
class SGZ_Predictor(Predictor):
    """Predictor for SGZ models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.weights or config.finetune

        model = MODELS.build(**config.model | {"weights": weights})
        model = model.to(device)
        model.eval()
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
                scale_factor = config.model.get("scale_factor", 1)
                imgsz = config.imgsz
                imgsz = Size(height=imgsz.h//scale_factor, width=imgsz.w//scale_factor)
                resize = T.ResizeDivisibleBy(height=imgsz.h, width=imgsz.w, divisor=32)
                transforms = resize + transforms

        self._transforms = transforms

    # --- Prediction ---
    @override
    @torch.inference_mode()
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

        # 2. Inference
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
        pass

    # --- Utilities ---
    @override
    def benchmark(self, imgsz: SizeLike | None = None):
        """Run the benchmark for the model.

        Args:
            imgsz (SizeLike, optional): The input image size for benchmarking.
                Defaults to None, which means using the default size.
        """
        config = self.config
        imgsz = Size.from_value(imgsz or config.imgsz)

        scale_factor = config.model.get("scale_factor")
        if scale_factor:
            imgsz = Size(height=imgsz.h // scale_factor, width=imgsz.w // scale_factor)

        if config.benchmark:
            self.model.benchmark(imgsz=imgsz, verbose=self.verbose)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
