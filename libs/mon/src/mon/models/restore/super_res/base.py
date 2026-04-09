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
from typing import Any, override

from mon.core import Size
from mon.nn import Model


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class SuperResolutionModel(Model, ABC):
    """A base class for all super-resolution models."""

    requires: set = {"x_lr", "y_hr", "imgsz"}
    provides: set = {"x_hr"}

    # --- Callable & Context Manager ---
    @override
    def forward(
        self,
        data: dict[str, Any] | None = None,
        save_debug: bool = False,
        *args, **kwargs
    ) -> dict[str, Any]:
        """Forward the input through the model.

        Args:
            data (dict[str, Any], optional): Input data dictionary. Defaults to None.
            save_debug (bool, optional): If True, return intermediate results
                for debugging. Defaults to False.
            **kwargs: Direct keyword arguments to pass to the forward step.
                Useful for simple inference, but should be used with caution as
                it bypasses the input validation. It's recommended to use the
                ``data`` dictionary for structured inputs.

        Returns:
            dict[str, Any]: Output dictionary.
        """
        # Validate inputs
        data = data or {}
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected 'data' to be a dict, but got {type(data).__name__}."
            )

        # 1. The escape hatch for simple inference, check **kwargs
        data |= kwargs
        y_hr = data.get("y_hr", None)
        imgsz = data.get("imgsz", None)

        if y_hr is not None and imgsz is None:
            data["imgsz"] = Size.from_value(y_hr)

        if "y_hr" not in data:
            data["y_hr"] = y_hr

        # 2. The enforcement
        missing_inputs = self.requires - data.keys()
        if missing_inputs:
            raise KeyError(
                f"{self.__class__.__name__} missing required inputs: {missing_inputs}. "
                f"Provided keys: {list(data.keys())}"
            )

        # 3. Execute forward pass logic
        outputs = self.forward_step(data=data, *args, **kwargs)

        # 4. Validate outputs
        if isinstance(outputs, dict):
            missing_outputs = self.provides - outputs.keys()
            if missing_outputs:
                raise KeyError(
                    f"{self.__class__.__name__} missing required outputs: {missing_outputs}. "
                    f"Provided keys: {list(outputs.keys())}"
                )
        else:
            raise TypeError(
                f"Expected 'outputs' to be a dict, but got {type(outputs).__name__}."
            )

        # 5. Return outputs
        if not save_debug:
            # Filter outputs to only include keys defined in `provides`
            outputs = {k: v for k, v in outputs.items() if k in self.provides}

        return outputs

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
