#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Metrics.

This module provides base metrics.
"""

from __future__ import annotations

__all__ = [
    "Metric",
]

from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Metric(nn.Module, ABC):
    """A base class for metric functions.

    This class provides a template for defining new metrics. It can be extended
    to implement specific metric functions.

    Attributes:
        metric_opts (dict): A dictionary of options for the metric function.
        metric_mode (str): The mode of the metric, either "FR" (full-reference)
            or "NR" (no-reference).
        lower_better (bool): Whether a lower score indicates better performance.
        score_range (tuple[float, float]): The valid range of scores for this
            metric, as a tuple of (min_score, max_score).
    """

    metric_opts: dict = {}
    metric_mode: str = "FR"             # ["FR" or "NR"]
    lower_better: bool = False          # True if lower score is better
    score_range: str = ""               # (min, max)

    # --- Lifecycle & Initialization ---
    def __init__(self, device: torch.device = torch.device("cpu")):
        """Initialize a new instance.

        Args:
            device (torch.device): The device to run the metric on.
                Defaults to "cpu".
        """
        super().__init__()

        # Assign attributes
        self.device = device

    # --- Callable & Context Manager ---
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Forward the input through the network."""
        pass

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
