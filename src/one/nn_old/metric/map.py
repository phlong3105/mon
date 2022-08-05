#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import math
from typing import Any
from typing import Optional

import torch
from torch import Tensor
from torchmetrics.metric import Metric

from one.core import Callable
from one.core import METRICS

__all__ = [
    "MAP",
]


@METRICS.register(name="map")
class MAP(Metric):
    """The APMeter measures the average precision per class.

    The APMeter is designed to operate on `NxK` Tensors `output` and `target`,
    and optionally a `Nx1` Tensor weight where (1) the `output` contains model
    output scores for `N` examples and `K` classes that ought to be higher when
    the model is more convinced that the example should be positively labeled,
    and smaller when the model believes the example should be negatively labeled
    (for measurement, the output of a sigmoid function); (2) the `target` contains
    only values 0 (for negative examples) and 1 (for positive examples); and
    (3) the `weight` ( > 0) represents weight for each sample.
    """

    def __init__(
        self,
        compute_on_step  : bool          = True,
        dist_sync_on_step: bool          = False,
        process_group    : Optional[Any] = None,
        dist_sync_fn     : Callable      = None,
    ):
        super().__init__(
            compute_on_step   = compute_on_step,
            dist_sync_on_step = dist_sync_on_step,
            process_group     = process_group,
            dist_sync_fn      = dist_sync_fn,
        )
        self.name = "map"
        self.add_state("scores", default=torch.FloatTensor(torch.FloatStorage()),
                       dist_reduce_fx=None)
        self.add_state("targets", default=torch.LongTensor(torch.LongStorage()),
                       dist_reduce_fx=None)
        self.add_state("weights", default=torch.FloatTensor(torch.FloatStorage()),
                       dist_reduce_fx=None)
        self.reset()

    def reset(self):
        """Resets the meter with empty member variables."""
        self.scores  = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.weights = torch.FloatTensor(torch.FloatStorage())

    def update(
        self, pred: Tensor, target: Tensor, weight: Optional[Tensor] = None
    ):
        """Add predictions and targets to the metric.
        
        Args:
            pred (Tensor):
                `NxK` tensor that for each of the N examples indicates the
                probability of the example belonging to each of the `K` classes,
                according to the model. The probabilities should sum to one over
                all classes.
            target (Tensor):
                Binary `NxK` tensor that encodes which of the `K` classes are
                associated with the N-th input (eg: a row [0, 1, 0, 1] indicates
                that the example is associated with classes 2 and 4)
            weight (Tensor, optional):
                `Nx1` tensor representing the weight for each example
                (each weight > 0).
        """
        if not torch.is_tensor(pred):
            pred = torch.from_numpy(pred)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if weight is not None:
            if not torch.is_tensor(weight):
                weight = torch.from_numpy(weight)
            weight = weight.squeeze()
       
        if pred.dim() == 1:
            pred = pred.view(-1, 1)
        elif pred.dim() != 2:
            raise ValueError("Wrong pred size (should be 1D or 2D with one "
                             "column per class).")
                
        if target.dim() == 1:
            target = target.view(-1, 1)
        elif target.dim() != 2:
            raise ValueError("Wrong target size (should be 1D or 2D with one "
                             "column per class).")
        
        if weight is not None:
            if weight.dim() != 1:
                raise ValueError("Weight dimension should be 1.")
            if weight.numel() != target.size()[0]:
                raise ValueError("Weight dimension 1 should be the same as "
                                 "that of target.")
            if torch.min(weight) >= 0:
                raise ValueError("Weight should be non-negative only.")
        
        if not torch.equal(target**2, target):
            raise ValueError("targets should be binary (0 or 1).")
        
        if self.scores.numel() > 0:
            if target.size()[1] != self.targets.size()[1]:
                raise ValueError("Dimensions for pred should match previously "
                                 "added examples.")
                
        # NOTE: Make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + pred.numel():
            new_size        = math.ceil(self.scores.storage().size() * 1.5)
            new_weight_size = math.ceil(self.weights.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + pred.numel()))
            self.targets.storage().resize_(int(new_size + pred.numel()))
            if weight is not None:
                self.weights.storage().resize_(
                    int(new_weight_size + pred.size()[0])
                )

        # NOTE: Store scores and targets
        offset = self.scores.size()[0] if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + pred.size()[0], pred.size()[1])
        self.targets.resize_(offset + target.size()[0], target.size()[1])
        self.scores.narrow(0, offset, pred.size()[0]).copy_(pred)
        self.targets.narrow(0, offset, target.size()[0]).copy_(target)

        if weight is not None:
            self.weights.resize_(offset + weight.size()[0])
            self.weights.narrow(0, offset, weight.size()[0]).copy_(weight)

    def compute(self) -> tuple[Tensor, Tensor]:
        """Returns the average precision for each class and the
        `Mean-Average-Precision (mAP)`.

        Return:
            map (Tensor):
                The `Mean-Average-Precision (mAP)` tensor.
            cls_ap (Tensor):
                `1xK` tensor, with avg precision for each class `K`.
        """
        if self.scores.numel() == 0:
            return Tensor(0), Tensor(0)
        ap = torch.zeros(self.scores.size()[1])
        rg = torch.range(1, self.scores.size()[0]).float()
        if self.weights.numel() > 0:
            weight         = self.weights.new(self.weights.size())
            weighted_truth = self.weights.new(self.weights.size())

        # NOTE: Compute average precision for each class
        for k in range(self.scores.size()[1]):
            # sort scores
            scores     = self.scores[:, k]
            targets    = self.targets[:, k]
            _, sortind = torch.sort(scores, 0, True)
            truth = targets[sortind]
            if self.weights.numel() > 0:
                weight         = self.weights[sortind]
                weighted_truth = truth.float() * weight
                rg             = weight.cumsum(0)

            # NOTE: Compute true positive sums
            if self.weights.numel() > 0:
                tp = weighted_truth.cumsum(0)
            else:
                tp = truth.float().cumsum(0)

            # NOTE: Compute precision curve
            precision = tp.div(rg)

            # NOTE: Compute average precision
            ap[k] = precision[truth.byte()].sum() / max(truth.sum(), 1)

        # NOTE: Compute mean average precision
        map = ap.mean()
        
        return map, ap
