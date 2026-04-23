#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Models.

This module defines the base class for super-resolution models.
"""

from __future__ import annotations

__all__ = [
    "SuperResolutionModel",
]

from abc import ABC
from typing import override

from tensordict import NonTensorData, TensorDict
from torch import Tensor

from mon.core import Config, Size
from mon.dataset import transform as T
from mon.nn import Model


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class SuperResolutionModel(Model, ABC):
    """A base class for all super-resolution models."""

    in_keys: set = {"x_lr", "y_hr", "imgsz"}
    out_keys: set = {"x_hr"}

    # --- Callable & Context Manager ---
    def __call__(
        self,
        data: TensorDict | None = None,
        save_debug: bool = False,
        *args, **kwargs
    ) -> TensorDict:
        """Forward the input through the model.

        Args:
            data (TensorDict, optional): Input data dictionary. Defaults to None.
            save_debug (bool, optional): If True, return intermediate results
                for debugging. Defaults to False.
            **kwargs: Direct keyword arguments to pass to the forward step.
                Useful for simple inference, but should be used with caution as
                it bypasses the input validation. It's recommended to use the
                ``data`` dictionary for structured inputs.

        Returns:
            TensorDict: Output dictionary.
        """
        # 1. Split kwargs into Data (for TensorDict) and Flags (for logic)
        # data_kwargs go into the TensorDict, flags stay for the method call
        data_kwargs = {k: v for k, v in kwargs.items() if k in self.in_keys}
        flags = {k: v for k, v in kwargs.items() if k not in data_kwargs}

        y_hr = data_kwargs.get("y_hr", None)
        imgsz = data_kwargs.get("imgsz", None)
        if y_hr is not None and imgsz is None:
            imgsz = Size.from_value(y_hr)

        data_kwargs["imgsz"] = imgsz
        data_kwargs["y_hr"] = y_hr

        # 2. Convert data input to TensorDict
        data_kwargs = {
            k: v if isinstance(v, Tensor) else NonTensorData(v)
            for k, v in data_kwargs.items()
        }
        if data is None:
            data = TensorDict(data_kwargs, batch_size=[])
        elif isinstance(data, TensorDict):
            data.update(data_kwargs)
        else:
            raise TypeError(
                f"Expected 'data' to be TensorDict, but got {type(data).__name__}."
            )

        # 3. Contract enforcement (Input)
        missing_inputs = self.in_keys - data.keys()
        if missing_inputs:
            raise KeyError(
                f"{self.__class__.__name__} missing required inputs: "
                f"{missing_inputs}. Provided keys: {list(data.keys())}"
            )

        # 4. Execution
        outputs = super().__call__(data=data, **flags)

        # 5. Ensure outputs is a TensorDict
        if not isinstance(outputs, TensorDict):
            outputs = TensorDict(outputs, batch_size=[])

        # 6. Contract enforcement (Output)
        missing_outputs = self.out_keys - outputs.keys()
        if missing_outputs:
            raise KeyError(
                f"{self.__class__.__name__} missing required outputs: "
                f"{missing_outputs}. Provided keys: {list(outputs.keys())}"
            )

        # 7. Filtering & return
        if not save_debug:
            # select() returns a new TensorDict with only the 'provides' keys
            outputs = outputs.select(*self.out_keys, strict=False)

        return outputs

    # --- Interfaces ---
    @override
    def build_transforms(self, config: Config | None = None) -> T.Compose:
        """Define the model's transformations.

        Args:
            config (Config, optional): The configuration object containing any
                necessary parameters for defining the transformations.
                Defaults to None.

        Returns:
            Callable: A callable (e.g., a torchvision transform or a custom
                function) that takes in the raw input data and returns the
                transformed data ready for the forward step.
        """
        transforms = T.Compose([
            T.Normalize(normalization="min_max"),
            T.ToTensorV2(transpose_mask=True),
        ])
        return transforms

    # --- Benchmark ---
    @override
    def benchmark(self, imgsz: Size, *args, **kwargs) -> dict[str, float]:
        """Perform a single forward step of the model to benchmark its performance.

        Args:
            imgsz (Size): Input image size.
            **kwargs: Additional arguments for benchmarking, such as number
                of runs, device, etc.

        Returns:
            dict[str, float]: A dictionary containing the benchmark results,
                such as latency, FLOPs, and parameter count.
        """
        pass

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
