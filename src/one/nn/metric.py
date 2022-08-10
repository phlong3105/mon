#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import torch
from torch import tensor
from torch import Tensor
from torch.nn.functional import mse_loss as mse
from torchmetrics import *
from torchmetrics import Metric
from torchmetrics.functional.image.psnr import _psnr_compute
from torchmetrics.functional.image.psnr import _psnr_update
from torchmetrics.utilities import rank_zero_warn

from one.constants import METRICS
from one.core import Ints
from one.vision.transformation import rgb_to_yuv


# H1: - Classification ---------------------------------------------------------

METRICS.register(name="accuracy",                         module=Accuracy)
METRICS.register(name="average_precision",                module=AveragePrecision)
METRICS.register(name="auc",                              module=AUC)
METRICS.register(name="auroc",                            module=AUROC)
METRICS.register(name="binned_average_precision",         module=BinnedAveragePrecision)
METRICS.register(name="binned_precision_recall_curve",    module=BinnedPrecisionRecallCurve)
METRICS.register(name="binned_recall_at_fixed_precision", module=BinnedRecallAtFixedPrecision)
METRICS.register(name="calibration_error",                module=CalibrationError)
METRICS.register(name="cohen_kappa",                      module=CohenKappa)
METRICS.register(name="confusion_matrix",                 module=ConfusionMatrix)
METRICS.register(name="coverage_error",                   module=CoverageError)
METRICS.register(name="dice",                             module=Dice)
METRICS.register(name="f1_score",                         module=F1Score)
METRICS.register(name="fbeta_Score",                      module=FBetaScore)
METRICS.register(name="hamming_distance",                 module=HammingDistance)
METRICS.register(name="hinge_loss",                       module=HingeLoss)
METRICS.register(name="jaccard_index",                    module=JaccardIndex)
METRICS.register(name="kl_divergence",                    module=KLDivergence)
METRICS.register(name="label_ranking_average_precision",  module=LabelRankingAveragePrecision)
METRICS.register(name="label_ranking_loss",               module=LabelRankingLoss)
METRICS.register(name="matthews_corr_coef",               module=MatthewsCorrCoef)
METRICS.register(name="precision",                        module=Precision)
METRICS.register(name="precision_recall_curve",           module=PrecisionRecallCurve)
METRICS.register(name="recall",                           module=Recall)
METRICS.register(name="roc",                              module=ROC)
METRICS.register(name="specificity",                      module=Specificity)
METRICS.register(name="stat_scores",                      module=StatScores)


# H1: - Image ------------------------------------------------------------------

def psnr(input: Tensor, target: Tensor, max_val: float) -> Tensor:
    """
    Peek Signal to Noise Ratio, which is similar to mean squared error.
    Given an m x n image, the PSNR is:

    .. math::
        \text{PSNR} = 10 \log_{10} \bigg(\frac{\text{MAX}_I^2}{MSE(I,T)}\bigg)

    where

    .. math::
        \text{MSE}(I,T) = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1} [I(i,
        j) - T(i,j)]^2

    and :math:`\text{MAX}_I` is the maximum possible input value
    (e.g for floating point images :math:`\text{MAX}_I=1`).

    Reference:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
        
    Args:
        input (Tensor): Input image with arbitrary shape [*].
        target (Tensor): Labels image with arbitrary shape [*].
        max_val (float): Maximum value in the input image.

    Return:
        Computed metric.
    """
    if input.shape != target.shape:
        raise TypeError(
            f"`input` and `target` must have equal shapes. "
            f"But got: {input.shape} != {target.shape}."
        )
    return 10.0 * torch.log10(
        max_val ** 2 / mse(input, target, reduction="mean")
    )


class PeakSignalNoiseRatioY(Metric):
    """
    Computes `Computes Peak Signal-to-Noise Ratio` (PSNR) on Y channel only :

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
    """
    
    min_target      : Tensor
    max_target      : Tensor
    higher_is_better: bool = False
    
    def __init__(
        self,
        data_range       : float | None = None,
        base             : float        = 10.0,
        reduction        : str          = "elementwise_mean",
        dim              : Ints | None  = None,
        compute_on_step  : bool         = True,
        dist_sync_on_step: bool         = False,
        process_group    : Any | None   = None,
    ) -> None:
        super().__init__(
            compute_on_step   = compute_on_step,
            dist_sync_on_step = dist_sync_on_step,
            process_group     = process_group,
        )

        if dim is None and reduction != "elementwise_mean":
            rank_zero_warn(
                f"The `reduction={reduction}` will not have any effect when "
                f"`dim` is None."
            )

        if dim is None:
            self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total",             default=tensor(0),   dist_reduce_fx="sum")
        else:
            self.add_state("sum_squared_error", default=[])
            self.add_state("total", default=[])

        if data_range is None:
            if dim is not None:
                # Maybe we could use `torch.amax(target, dim=dim) - torch.amin(target, dim=dim)` in PyTorch 1.7 to
                # calculate `data_range` in the future.
                raise ValueError(
                    "The `data_range` must be given when `dim` is not None."
                )

            self.data_range = None
            self.add_state("min_target", default=tensor(0.0), dist_reduce_fx=torch.min)
            self.add_state("max_target", default=tensor(0.0), dist_reduce_fx=torch.max)
        else:
            self.add_state("data_range", default=tensor(float(data_range)), dist_reduce_fx="mean")
        
        self.base      = base
        self.reduction = reduction
        self.dim       = tuple(dim) if isinstance(dim, Sequence) else dim
        
    def update(self, preds: Tensor, target: Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds (Tensor): Predictions from model.
            target (Tensor): Ground truth values.
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
        
    def compute(self) -> Tensor:
        """
        Compute peak signal-to-noise ratio over state.
        """
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


METRICS.register(name="error_relative_global_dimensionless_synthesis",
                 module=ErrorRelativeGlobalDimensionlessSynthesis)
METRICS.register(name="multi_scale_ssim",
                 module=MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register(name="multi_scale_structural_similarity_index_measure",
                 module=MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register(name="psnr",                                module=PeakSignalNoiseRatio)
METRICS.register(name="psnr_y",                              module=PeakSignalNoiseRatioY)
METRICS.register(name="peak_signal_noise_ratio",             module=PeakSignalNoiseRatio)
METRICS.register(name="peak_signal_noise_ratio_y",           module=PeakSignalNoiseRatioY)
METRICS.register(name="spectral_angle_mapper",               module=SpectralAngleMapper)
METRICS.register(name="spectral_distortion_index",           module=SpectralDistortionIndex)
METRICS.register(name="ssim",                                module=StructuralSimilarityIndexMeasure)
METRICS.register(name="structural_similarity_index_measure", module=StructuralSimilarityIndexMeasure)
METRICS.register(name="universal_image_quality_index",       module=UniversalImageQualityIndex)


# H1: - Regression -------------------------------------------------------------

METRICS.register(name="cosine_similarity",                        module=CosineSimilarity)
METRICS.register(name="explained_variance",                       module=ExplainedVariance)
METRICS.register(name="mean_absolute_error",                      module=MeanAbsoluteError)
METRICS.register(name="mean_absolute_percentage_error",           module=MeanAbsolutePercentageError)
METRICS.register(name="mean_squared_error",                       module=MeanSquaredError)
METRICS.register(name="mean_squared_log_error",                   module=MeanSquaredLogError)
METRICS.register(name="pearson_corr_coef",                        module=PearsonCorrCoef)
METRICS.register(name="r2_score",                                 module=R2Score)
METRICS.register(name="spearman_corr_coef",                       module=SpearmanCorrCoef)
METRICS.register(name="symmetric_mean_absolute_percentage_error", module=SymmetricMeanAbsolutePercentageError)
METRICS.register(name="tweedie_deviance_score",                   module=TweedieDevianceScore)
METRICS.register(name="weighted_mean_absolute_percentage_error",  module=WeightedMeanAbsolutePercentageError)


# H1: - Retrieval --------------------------------------------------------------

METRICS.register(name="retrieval_fallout",                   module=RetrievalFallOut)
METRICS.register(name="retrieval_hit_rate",                  module=RetrievalHitRate)
METRICS.register(name="retrieval_map",                       module=RetrievalMAP)
METRICS.register(name="retrieval_mrr",                       module=RetrievalMRR)
METRICS.register(name="retrieval_normalized_dcg",            module=RetrievalNormalizedDCG)
METRICS.register(name="retrieval_precision",                 module=RetrievalPrecision)
METRICS.register(name="retrieval_precision_recall_curve",    module=RetrievalPrecisionRecallCurve)
METRICS.register(name="retrieval_recall",                    module=RetrievalRecall)
METRICS.register(name="retrieval_recall_at_fixed_precision", module=RetrievalRecallAtFixedPrecision)
METRICS.register(name="retrieval_r_precision",               module=RetrievalRPrecision)
