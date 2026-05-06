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
    K,
    MODELS,
    Path,
    PREDICTORS,
    Size,
    Split,
    TimeProfiler,
)
from mon.dataset import build_dataset, DataLoader, Dataset, transform as T
from mon.ops import vis_heatmap
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
        source: dict | Path,
        split: Split = Split.TEST,
        transforms: T.Compose | None = None,
    ) -> tuple[str, Dataset | DataLoader]:
        """Initialize and return a dataset or dataloader.

        Args:
            source (dict | Path): A dataset/dataloader configuration dictionary
                or a source path.
            split (Split, optional): The data split to use. Defaults to Split.TEST.
            transforms (T.Compose | None, optional): The data transformations
                to apply. Defaults to None.

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
            outputs (TensorDict): The dictionary containing the main results.
                Each key in the dictionary is a batch of results.
        """
        upsampler = self._upsampler

        meta = datapoint["meta"]
        meta = meta[0] if isinstance(meta, (list, tuple)) else meta
        path = Path(meta["path"])
        size = Size.from_any(meta["imgsz"])

        depth = outputs["depth"]
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

        # Resize the image if needed
        imgsz = Size.from_any(depth)
        if imgsz != size:
            depth = upsampler(x_lr=depth, y_hr=None, imgsz=size)

        self._save_image(
            image=depth,
            src_path=path,
            dirname=K.DEPTH_DIR,
            subdirname="",
            stem="",
        )

    @override
    def _save_debug(self, datapoint: TensorDict, outputs: TensorDict):
        """Save debugging results for visualization.

        Args:
            datapoint (TensorDict): The dictionary containing the input data.
            outputs (TensorDict): The dictionary containing the main results.
                Each key in the dictionary is a batch of results.
        """
        upsampler = self._upsampler

        meta = datapoint["meta"]
        meta = meta[0] if isinstance(meta, (list, tuple)) else meta
        path = Path(meta["path"])
        size = Size.from_any(meta["imgsz"])

        depth_c = vis_heatmap(outputs["depth"], colormap="Spectral_r")

        # Resize the image if needed
        imgsz = Size.from_any(depth_c)
        if imgsz != size:
            depth_c = upsampler(x_lr=depth_c, y_hr=None, imgsz=size)

        self._save_image(
            image=depth_c,
            src_path=path,
            dirname=K.DEPTH_DIR,
            subdirname="",
            stem="depth_c",
        )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
