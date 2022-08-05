#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Sequence
from typing import Union

import torch
from one.imgproc import rgb_to_yuv
from torch import tensor
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.image.psnr import _psnr_compute
from torchmetrics.functional.image.psnr import _psnr_update
from torchmetrics.utilities import rank_zero_warn

from one.core import METRICS

__all__ = [
    "PeakSignalNoiseRatioY",
]


# MARK: - Modules

@METRICS.register(name="psnry")
@METRICS.register(name="peak_signal_noise_ratio_y")
class PeakSignalNoiseRatioY(Metric):
    r"""Computes `Computes Peak Signal-to-Noise Ratio` (PSNR) on Y channel only :

    .. math:: \text{PSNR}(I, J) = 10 * \log_{10} \left(\frac{\max(I)^2}{\text{MSE}(I, J)}\right)

    Where :math:`\text{MSE}` denotes the `mean-squared-error`_ function.

    Args:
        data_range:
            the range of the data. If None, it is determined from the data (max - min).
            The ``data_range`` must be given when ``dim`` is not None.
        base: a base of a logarithm to use.
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        dim:
            Dimensions to reduce PSNR scores over, provided as either an integer or a list of integers. Default is
            None meaning scores will be reduced across all dimensions and all batches.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called.

    Raises:
        ValueError:
            If ``dim`` is not ``None`` and ``data_range`` is not given.

    Example:
        >>> from torchmetrics import PeakSignalNoiseRatio
        >>> psnr = PeakSignalNoiseRatio()
        >>> preds = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> psnr(preds, target)
        tensor(2.5527)

    .. note::
        Half precision is only support on GPU for this metric

    """
    
    min_target      : Tensor
    max_target      : Tensor
    higher_is_better: bool = False
    
    # MARK: Configure
    
    def __init__(
        self,
        data_range       : Optional[float]                       = None,
        base             : float                                 = 10.0,
        reduction        : str                                   = "elementwise_mean",
        dim              : Optional[Union[int, tuple[int, ...]]] = None,
        compute_on_step  : bool                                  = True,
        dist_sync_on_step: bool                                  = False,
        process_group    : Optional[Any]                         = None,
    ) -> None:
        super().__init__(
            compute_on_step   = compute_on_step,
            dist_sync_on_step = dist_sync_on_step,
            process_group     = process_group,
        )

        if dim is None and reduction != "elementwise_mean":
            rank_zero_warn(f"The `reduction={reduction}` will not have any "
                           f"effect when `dim` is None.")

        if dim is None:
            self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        else:
            self.add_state("sum_squared_error", default=[])
            self.add_state("total", default=[])

        if data_range is None:
            if dim is not None:
                # Maybe we could use `torch.amax(target, dim=dim) - torch.amin(target, dim=dim)` in PyTorch 1.7 to
                # calculate `data_range` in the future.
                raise ValueError("The `data_range` must be given when `dim` is not None.")

            self.data_range = None
            self.add_state("min_target", default=tensor(0.0), dist_reduce_fx=torch.min)
            self.add_state("max_target", default=tensor(0.0), dist_reduce_fx=torch.max)
        else:
            self.add_state("data_range", default=tensor(float(data_range)), dist_reduce_fx="mean")
        
        self.base      = base
        self.reduction = reduction
        self.dim       = tuple(dim) if isinstance(dim, Sequence) else dim
    
    # MARK: Update
    
    def update(self, preds: Tensor, target: Tensor):
        """Update state with predictions and targets.

        Args:
            preds (Tensor):
                Predictions from model.
            target (Tensor):
                Ground truth values.
        """
        preds_yuv  = rgb_to_yuv(preds)
        target_yuv = rgb_to_yuv(target)
        preds_y    = preds_yuv[..., 0, :, :]
        target_y   = target_yuv[..., 0, :, :]
        
        sum_squared_error, n_obs = _psnr_update(preds_y, target_y, dim=self.dim)
        if self.dim is None:
            if self.data_range is None:
                # Keep track of min and max target values
                self.min_target = min(target.min(), self.min_target)
                self.max_target = max(target.max(), self.max_target)

            self.sum_squared_error += sum_squared_error
            self.total             += n_obs
        else:
            self.sum_squared_error.append(sum_squared_error)
            self.total.append(n_obs)
    
    # MARK: Compute
    
    def compute(self) -> Tensor:
        """Compute peak signal-to-noise ratio over state."""
        if self.data_range is not None:
            data_range = self.data_range
        else:
            data_range = self.max_target - self.min_target

        if self.dim is None:
            sum_squared_error = self.sum_squared_error
            total             = self.total
        else:
            sum_squared_error = torch.cat([v.flatten() for v in self.sum_squared_error])
            total             = torch.cat([v.flatten() for v in self.total])
        
        return _psnr_compute(
            sum_squared_error, total, data_range, base=self.base,
            reduction=self.reduction
        )
