#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Schedulers.

This package contains various custom learning rate schedulers for training
neural networks.
"""

from __future__ import annotations

__all__ = [
    "CosineAnnealingRestartLR",
    "CosineAnnealingRestartCyclicLR",
    "GradualWarmupScheduler",
]

import importlib
import inspect
import math
import pkgutil

import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from mon.core import SCHEDULERS


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================

def __register_schedulers(module, prefix: str = ""):
    """Register all scheduler classes from the given module and its submodules
    into the SCHEDULERS registry.

    Args:
        module (ModuleType): The module to inspect for scheduler classes.
        prefix (str, optional): The prefix for submodule names. Defaults to "".
    """

    def is_transform_class(obj):
        return (
            inspect.isclass(obj) and
            issubclass(obj, LRScheduler) and
            not inspect.isabstract(obj)
        )

    for _, module_name, is_pkg in pkgutil.walk_packages(module.__path__, prefix=module.__name__ + "."):
        try:
            # Import the submodule
            sub_module = importlib.import_module(module_name)

            # Inspect all members of the submodule
            for name, obj in inspect.getmembers(sub_module):
                if is_transform_class(obj) and not name.startswith("_"):
                    # if name not in __all__:
                        # Add to __all__ and SCHEDULERS registry __all__.append(name)
                        globals()[name] = obj
                        SCHEDULERS.register(name=name, module=obj, replace=True)

            # If it's a package, recursively inspect its submodules
            if is_pkg:
                __register_schedulers(sub_module, prefix=module_name + ".")
        except ImportError as e:
            # Skip modules that can't be imported
            continue


__register_schedulers(optim)
del __register_schedulers
SCHEDULERS.sort()

# endregion


# ==============================================================================
# region SCHEDULERS
# ==============================================================================

@SCHEDULERS.register(name="CosineAnnealingRestartLR")
class CosineAnnealingRestartLR(LRScheduler):
    """Cosine annealing with restarts learning rate scheme.

    An example of config:

        - periods = [10, 10, 10, 10]
        - restart_weights = [1, 0.5, 0.5, 0.5]
        - eta_min=1e-7
        - It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
          scheduler will restart with the weights in restart_weights.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        periods: tuple,
        restart_weights: tuple = (1, ),
        eta_min: int = 0,
        last_epoch: int = -1
    ):
        """Initialize a new instance.

        Args:
            optimizer (optim.Optimizer): Torch optimizer.
            periods (tuple): Period for each cosine anneling cycle.
            restart_weights (tuple, optional): Restart weights at each restart
                iteration. Defaults to (1, ).
            eta_min (int, optional): The mimimum lr. Defaults to 0.
            last_epoch (int, optional): Used in _LRScheduler. Defaults to -1.
        """
        # Validate inputs
        if len(periods) != len(restart_weights):
            raise ValueError(
                f"'periods' and 'restart_weights' should have the same length, "
                f"but got: {len(periods)} != {len(restart_weights)}."
            )

        # Assign attributes
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        self.cumulative_period = [
            sum(self.periods[0:i + 1])
            for i in range(0, len(self.periods))
        ]

        # Continue the initialization chain
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    # --- Interfaces ---
    def get_lr(self) -> list[float]:
        idx: int = _get_position_from_periods(
            iteration=self.last_epoch,
            cumulative_period=self.cumulative_period
        )
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


@SCHEDULERS.register(name="CosineAnnealingRestartCyclicLR")
class CosineAnnealingRestartCyclicLR(LRScheduler):
    """Cosine annealing with restarts learning rate scheme.

    An example of config:

        - periods = [10, 10, 10, 10]
        - restart_weights = [1, 0.5, 0.5, 0.5]
        - eta_min=1e-7
        - It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
          scheduler will restart with the weights in restart_weights.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        optimizer: optim.Optimizer,
        periods: tuple,
        restart_weights: tuple = (1, ),
        eta_mins: tuple = (0, ),
        last_epoch: int = -1
    ):
        """Initialize a new instance.

        Args:
            optimizer (optim.Optimizer): Torch optimizer.
            periods (tuple): Period for each cosine anneling cycle.
            restart_weights (tuple, optional): Restart weights at each restart
                iteration. Defaults to (1, ).
            eta_mins (tuple, optional): The mimimum lr for each cycle.
                Defaults to (0, ).
            last_epoch (int, optional): Used in _LRScheduler. Defaults to -1.
        """
        # Validate inputs
        if len(periods) != len(restart_weights):
            raise ValueError(
                f"'periods' and 'restart_weights' should have the same length, "
                f"but got: {len(periods)} != {len(restart_weights)}."
            )

        # Assign attributes
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        # Continue the initialization chain
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    # --- Interfaces ---
    def get_lr(self) -> list[float]:
        idx: int = _get_position_from_periods(
            iteration=self.last_epoch,
            cumulative_period=self.cumulative_period
        )
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]

# endregion


# ==============================================================================
# region WARMUP SCHEDULERS
# ==============================================================================

@SCHEDULERS.register(name="GradualWarmupScheduler")
class GradualWarmupScheduler(LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        optimizer: optim.Optimizer,
        multiplier: float,
        total_epoch: int,
        after_scheduler: LRScheduler | None = None
    ):
        """Initialize a new instance.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            multiplier (float): Target learning rate = base lr * multiplier.
                If multiplier = 1.0, the learning rate starts from 0 and ends
                up with the base lr. If multiplier > 1.0, the learning rate
                starts from the base lr and ends up with the target learning rate.
            total_epoch (int): Target learning rate is reached at total_epoch.
            after_scheduler (LRScheduler | None, optional): After target_epoch,
                use this scheduler(e.g. ReduceLROnPlateau). Defaults to None.
        """
        # Validate inputs
        if multiplier < 1.0:
            raise ValueError(
                f"'multiplier' should be greater than or equal to 1.0, "
                f"but got: {multiplier}."
            )

        # Assign attributes
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False

        # Continue the initialization chain
        super().__init__(optimizer=optimizer)

    # --- Interfaces ---
    def get_lr(self) -> list[float]:
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier
                        for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step(self, epoch: int | None = None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

    def step_ReduceLROnPlateau(self, metrics, epoch: int | None = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

# endregion


# ==============================================================================
# region UTILITIES
# ==============================================================================

def _get_position_from_periods(iteration: int, cumulative_period: list[int]) -> int | None:
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int | None: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i
    return None

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
