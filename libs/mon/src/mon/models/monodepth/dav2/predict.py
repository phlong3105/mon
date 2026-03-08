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
from typing_extensions import override

from mon.core import (
    DictLike,
    K,
    MODELS,
    Path,
    PathLike,
    resolve_project_root,
    RunMode,
    Size,
    Split,
    SplitLike,
    Task,
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
        """Initialize ``self._transforms`` attribute."""
        self._transforms = None

    @override
    def _init_data(
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
        imgsz = config.eval_imgsz

        # 1. Prepare inputs
        timers.preprocess.tick()
        image = datapoint["image"]
        timers.preprocess.tock()

        # 2. Inference
        timers.infer.tick()
        depth = self.model(image, imgsz.h)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        outputs = {"depth": depth}
        timers.infer.tock()

        return outputs

    # --- Output ---
    @override
    def _save(self, outputs: dict, meta: dict):
        """Save the main prediction results to a file.

        Args:
            outputs (dict): The dictionary containing the main prediction results.
            meta (dict): The dictionary containing the metadata.
        """
        path = Path(meta["path"])
        size = Size.from_value(meta["imgsz"])
        depth = outputs["depth"]
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        self._save_image(depth, size, path)

    @override
    def _save_debug(self, outputs: dict, meta: dict):
        """Save debugging results for visualization.

        Args:
            outputs (dict): The dictionary containing the debugging results.
            meta (dict): The dictionary containing the metadata.
        """
        path = Path(meta["path"])
        size = Size.from_value(meta["imgsz"])
        depth_c = vis_heatmap(outputs["depth"], colormap="Spectral_r")
        self._save_image(depth_c, size, path, dirname=K.DEBUG_DIR, stem="depth_c")

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

def main():
    """Unit test for DAV2_Predictor."""
    predictor = DAV2_Predictor.from_cli(
        prompt=True,
        root=resolve_project_root(current_dir),
        config_file="dav2_vitb_da2k.yaml",
        task=Task.MONODEPTH,
        mode=RunMode.PREDICT,
        arch="dav2",
        model="dav2_vitb",
        save=True,
        exist_ok=True,
        verbose=True,
    )
    predictor.predict()


if __name__ == "__main__":
    pass

# endregion
