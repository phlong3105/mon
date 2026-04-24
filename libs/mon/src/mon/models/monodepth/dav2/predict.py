#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Script.

This script provides a CLI for running Depth Anything V2 prediction on a given
dataset.

References:
    - Paper: "Depth Anything V2. A More Capable Foundation Model for Monocular
      Depth Estimation," NeurIPS 2024.
    - Code: https://github.com/DepthAnything/Depth-Anything-V2
"""

from __future__ import annotations

__all__ = [
    "DAV2_Predictor",
]

import numpy as np
import torch
from tensordict import TensorDict
from typing_extensions import override

from mon.core import (
    DictLike,
    K,
    MODELS,
    Path,
    PathLike,
    PREDICTORS,
    Size,
    Split,
    SplitLike,
    TimeProfiler,
)
from mon.dataset import build_dataset, DataLoader, Dataset, transform as T
from mon.ops import normalize_minmax, vis_heatmap
from mon.runners import Predictor

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region PREDICTOR
# ==============================================================================

@PREDICTORS.register(name="dav2")
class DAV2_Predictor(Predictor):
    """Predictor for DAV2 models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = "default"

        model = MODELS.build(**config.model | { "weights": weights, "device": device})
        model = model.to(device)
        model.eval()
        self._model = model

    @override
    def _init_transforms(self):
        """Initialize ``self._transforms`` attribute for pre-processing the
        input data.
        """
        self._transforms = None

    @override
    def _build_dataloader(
        self,
        source: DictLike | PathLike,
        split: SplitLike = Split.TEST,
        transforms: T.Compose | None = None,
    ) -> tuple[str, Dataset | DataLoader]:
        """Initialize and return a dataset or dataloader.

        Args:
            source (DictLike | PathLike): A dataset/dataloader configuration
                dictionary or a source path.
            split (SplitLike, optional): The data split to use.
                Defaults to Split.TEST.
            transforms (T.Compose, optional): The data transformations to apply.
                Defaults to None.

        Returns:
            tuple[str, Dataset | DataLoader]: A tuple containing the name of the
                dataset/dataloader and the dataset/dataloader object itself.
        """
        return build_dataset(
            src=source,
            dataset_dir=self.config.data_dir,
            split=split,
            transforms=transforms,
        )

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
        config = self.config
        imgsz = config.imgsz

        # 1. Prepare inputs
        timers.preprocess.tick()
        image = datapoint["image"]
        timers.preprocess.tock()

        # 2. Inference
        timers.infer.tick()
        outputs = self.model(image=image, input_size=imgsz.h)
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
        meta = datapoint["meta"]
        meta = meta[0] if isinstance(meta, (list, tuple)) else meta
        path = Path(meta["path"])
        size = Size.from_value(meta["imgsz"])

        depth = outputs["depth"]
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

        # Resize the image if needed
        imgsz = Size.from_value(depth)
        if imgsz != size:
            depth = self._upsampler(x_lr=depth, imgsz=size)["y_hr"]

        self._save_image(depth, path, dirname=K.PRED_DIR)

    @override
    def _save_debug(self, datapoint: TensorDict, outputs: TensorDict):
        """Save debugging results for visualization.

        Args:
            datapoint (TensorDict): The dictionary containing the input data.
            outputs (TensorDict): The dictionary containing the main
                prediction results. Each key in the dictionary is a batch of
                prediction results.
        """
        meta = datapoint["meta"]
        meta = meta[0] if isinstance(meta, (list, tuple)) else meta
        path = Path(meta["path"])
        size = Size.from_value(meta["imgsz"])

        depth_c = vis_heatmap(outputs["depth"], colormap="Spectral_r")

        # Resize the image if needed
        imgsz = Size.from_value(depth_c)
        if imgsz != size:
            depth_c = self._upsampler(x_lr=depth_c, imgsz=size)["y_hr"]

        self._save_image(depth_c, path, dirname=K.DEBUG_DIR, stem="depth_c")

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
