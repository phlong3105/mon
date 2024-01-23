#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements evaluation metrics using the :mod:`torchmetrics`
package.
"""

from __future__ import annotations

__all__ = [
    "AUROC",
    "Accuracy",
    "AveragePrecision",
    "BootStrapper",
    "CalibrationError",
    "CatMetric",
    "ClasswiseWrapper",
    "CohenKappa",
    "ConcordanceCorrCoef",
    "ConfusionMatrix",
    "CosineSimilarity",
    "CramersV",
    "Dice",
    "ExactMatch",
    "ExplainedVariance",
    "F1Score",
    "FBetaScore",
    "FleissKappa",
    "HammingDistance",
    "HingeLoss",
    "JaccardIndex",
    "KLDivergence",
    "KendallRankCorrCoef",
    "LogCoshError",
    "MatthewsCorrCoef",
    "MaxMetric",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanMetric",
    "MeanSquaredError",
    "MeanSquaredLogError",
    "Metric",
    "MetricCollection",
    "MetricTracker",
    "MinMaxMetric",
    "MinMetric",
    "MinkowskiDistance",
    "MultioutputWrapper",
    "MultitaskWrapper",
    "PearsonCorrCoef",
    "PearsonsContingencyCoefficient",
    "Precision",
    "PrecisionAtFixedRecall",
    "PrecisionRecallCurve",
    "R2Score",
    "ROC",
    "Recall",
    "RecallAtFixedPrecision",
    "RelativeSquaredError",
    "RetrievalFallOut",
    "RetrievalHitRate",
    "RetrievalMAP",
    "RetrievalMRR",
    "RetrievalNormalizedDCG",
    "RetrievalPrecision",
    "RetrievalPrecisionRecallCurve",
    "RetrievalRPrecision",
    "RetrievalRecall",
    "RetrievalRecallAtFixedPrecision",
    "RunningMean",
    "RunningSum",
    "SpearmanCorrCoef",
    "Specificity",
    "SpecificityAtSensitivity",
    "StatScores",
    "SumMetric",
    "SymmetricMeanAbsolutePercentageError",
    "TheilsU",
    "TschuprowsT",
    "TweedieDevianceScore",
    "WeightedMeanAbsolutePercentageError",
]

from abc import ABC
from typing import Literal

import torchmetrics

from mon.globals import METRICS


# region Base Metric

class Metric(torchmetrics.Metric, ABC):
    """The base class for all loss functions.

    Args:
        mode: One of: ``'FR'`` or ``'NR'``. Default: ``'FR'``.
        lower_is_better: Default: ``False``.
    """
    def __init__(
        self,
        mode           : Literal["FR", "NR"] = "FR",
        lower_is_better: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mode            = mode
        self.lower_is_better = lower_is_better

# endregion


# region Aggregation

MetricCollection = torchmetrics.MetricCollection

CatMetric        = torchmetrics.CatMetric
MaxMetric        = torchmetrics.MaxMetric
MeanMetric       = torchmetrics.MeanMetric
MinMetric        = torchmetrics.MinMetric
RunningMean      = torchmetrics.RunningMean
RunningSum       = torchmetrics.RunningSum
SumMetric        = torchmetrics.SumMetric

# endregion


# region Wrapper

BootStrapper         = torchmetrics.BootStrapper
ClasswiseWrapper     = torchmetrics.ClasswiseWrapper
MetricTracker        = torchmetrics.MetricTracker
MinMaxMetric         = torchmetrics.MinMaxMetric
MultioutputWrapper   = torchmetrics.MultioutputWrapper
MultitaskWrapper     = torchmetrics.MultitaskWrapper

# endregion


# region Classification Metric

AUROC                    = torchmetrics.AUROC
Accuracy                 = torchmetrics.Accuracy
AveragePrecision         = torchmetrics.AveragePrecision
CalibrationError         = torchmetrics.CalibrationError
CohenKappa               = torchmetrics.CohenKappa
ConfusionMatrix          = torchmetrics.ConfusionMatrix
Dice                     = torchmetrics.Dice
ExactMatch               = torchmetrics.ExactMatch
F1Score                  = torchmetrics.F1Score
FBetaScore               = torchmetrics.FBetaScore
HammingDistance          = torchmetrics.HammingDistance
HingeLoss                = torchmetrics.HingeLoss
JaccardIndex             = torchmetrics.JaccardIndex
MatthewsCorrCoef         = torchmetrics.MatthewsCorrCoef
Precision                = torchmetrics.Precision
PrecisionAtFixedRecall   = torchmetrics.PrecisionAtFixedRecall
PrecisionRecallCurve     = torchmetrics.PrecisionRecallCurve
ROC                      = torchmetrics.ROC
Recall                   = torchmetrics.Recall
RecallAtFixedPrecision   = torchmetrics.RecallAtFixedPrecision
Specificity              = torchmetrics.Specificity
SpecificityAtSensitivity = torchmetrics.SpecificityAtSensitivity
StatScores               = torchmetrics.StatScores

METRICS.register(name="auroc",                      module=AUROC)
METRICS.register(name="accuracy",                   module=Accuracy)
METRICS.register(name="average_precision",          module=AveragePrecision)
METRICS.register(name="calibration_error",          module=CalibrationError)
METRICS.register(name="cohen_kappa",                module=CohenKappa)
METRICS.register(name="confusion_matrix",           module=ConfusionMatrix)
METRICS.register(name="dice",                       module=Dice)
METRICS.register(name="exact_match",                module=ExactMatch)
METRICS.register(name="f1_score ",                  module=F1Score)
METRICS.register(name="f_beta_score",               module=FBetaScore)
METRICS.register(name="hamming_distance",           module=HammingDistance)
METRICS.register(name="hinge_loss",                 module=HingeLoss)
METRICS.register(name="jaccard_index",              module=JaccardIndex)
METRICS.register(name="matthews_corr_coef",         module=MatthewsCorrCoef)
METRICS.register(name="precision",                  module=Precision)
METRICS.register(name="precision_at_fixed_recall",  module=PrecisionAtFixedRecall)
METRICS.register(name="precision_recall_curve",     module=PrecisionRecallCurve)
METRICS.register(name="roc",                        module=ROC)
METRICS.register(name="recall",                     module=Recall)
METRICS.register(name="recall_at_fixed_precision",  module=RecallAtFixedPrecision)
METRICS.register(name="specificity",                module=Specificity)
METRICS.register(name="specificity_at_sensitivity", module=SpecificityAtSensitivity)
METRICS.register(name="stat_scores",                module=StatScores)

# endregion


# region Nominal Metric

CramersV                       = torchmetrics.CramersV
FleissKappa                    = torchmetrics.FleissKappa
PearsonsContingencyCoefficient = torchmetrics.PearsonsContingencyCoefficient
TheilsU                        = torchmetrics.TheilsU
TschuprowsT                    = torchmetrics.TschuprowsT

METRICS.register(name="cramers_v",                        module=CramersV)
METRICS.register(name="fleiss_kappa",                     module=FleissKappa)
METRICS.register(name="pearsons_contingency_coefficient", module=PearsonsContingencyCoefficient)
METRICS.register(name="theils_u",                         module=TheilsU)
METRICS.register(name="tschuprows_t",                     module=TschuprowsT)

# endregion


# region Regression Metric

ConcordanceCorrCoef                  = torchmetrics.ConcordanceCorrCoef
CosineSimilarity                     = torchmetrics.CosineSimilarity
ExplainedVariance                    = torchmetrics.ExplainedVariance
KLDivergence                         = torchmetrics.KLDivergence
KendallRankCorrCoef                  = torchmetrics.KendallRankCorrCoef
LogCoshError                         = torchmetrics.LogCoshError
MeanAbsoluteError                    = torchmetrics.MeanAbsoluteError
MeanAbsolutePercentageError          = torchmetrics.MeanAbsolutePercentageError
MeanSquaredError                     = torchmetrics.MeanSquaredError
MeanSquaredLogError                  = torchmetrics.MeanSquaredLogError
MinkowskiDistance                    = torchmetrics.MinkowskiDistance
PearsonCorrCoef                      = torchmetrics.PearsonCorrCoef
R2Score                              = torchmetrics.R2Score
RelativeSquaredError                 = torchmetrics.RelativeSquaredError
SpearmanCorrCoef                     = torchmetrics.SpearmanCorrCoef
SymmetricMeanAbsolutePercentageError = torchmetrics.SymmetricMeanAbsolutePercentageError
TweedieDevianceScore                 = torchmetrics.TweedieDevianceScore
WeightedMeanAbsolutePercentageError  = torchmetrics.WeightedMeanAbsolutePercentageError

METRICS.register(name="concordance_corr_coef",                    module=ConcordanceCorrCoef)
METRICS.register(name="cosine_similarity",                        module=CosineSimilarity)
METRICS.register(name="explained_variance",                       module=ExplainedVariance)
METRICS.register(name="kl_divergence",                            module=KLDivergence)
METRICS.register(name="kendall_rank_corr_coef",                   module=KendallRankCorrCoef)
METRICS.register(name="log_cosh_error",                           module=LogCoshError)
METRICS.register(name="mae",                                      module=MeanAbsoluteError)
METRICS.register(name="mean_absolute_error",                      module=MeanAbsoluteError)
METRICS.register(name="mean_absolute_percentage_error",           module=MeanAbsolutePercentageError)
METRICS.register(name="mean_squared_error",                       module=MeanSquaredError)
METRICS.register(name="mse",                                      module=MeanSquaredError)
METRICS.register(name="mean_squared_log_error",                   module=MeanSquaredLogError)
METRICS.register(name="minkowski_distance",                       module=MinkowskiDistance)
METRICS.register(name="pearson_corr_coef",                        module=PearsonCorrCoef)
METRICS.register(name="r2_score",                                 module=R2Score)
METRICS.register(name="relative_squared_error",                   module=RelativeSquaredError)
METRICS.register(name="spearman_corr_coef",                       module=SpearmanCorrCoef)
METRICS.register(name="symmetric_mean_absolute_percentage_error", module=SymmetricMeanAbsolutePercentageError)
METRICS.register(name="tweedie_deviance_score",                   module=TweedieDevianceScore)
METRICS.register(name="weighted_mean_absolute_percentage_error",  module=WeightedMeanAbsolutePercentageError)

# endregion


# region Retrieval Metric

RetrievalFallOut                = torchmetrics.RetrievalFallOut
RetrievalHitRate                = torchmetrics.RetrievalHitRate
RetrievalMAP                    = torchmetrics.RetrievalMAP
RetrievalMRR                    = torchmetrics.RetrievalMRR
RetrievalNormalizedDCG          = torchmetrics.RetrievalNormalizedDCG
RetrievalPrecision              = torchmetrics.RetrievalPrecision
RetrievalPrecisionRecallCurve   = torchmetrics.RetrievalPrecisionRecallCurve
RetrievalRecall                 = torchmetrics.RetrievalRecall
RetrievalRecallAtFixedPrecision = torchmetrics.RetrievalRecallAtFixedPrecision
RetrievalRPrecision             = torchmetrics.RetrievalRPrecision

METRICS.register(name="retrieval_fall_out",                  module=RetrievalFallOut)
METRICS.register(name="retrieval_hit_rate",                  module=RetrievalHitRate)
METRICS.register(name="retrieval_map",                       module=RetrievalMAP)
METRICS.register(name="retrieval_mrr",                       module=RetrievalMRR)
METRICS.register(name="retrieval_normalized_dcg",            module=RetrievalNormalizedDCG)
METRICS.register(name="retrieval_precision",                 module=RetrievalPrecision)
METRICS.register(name="retrieval_precision_recall_curve",    module=RetrievalPrecisionRecallCurve)
METRICS.register(name="retrieval_recall",                    module=RetrievalRecall)
METRICS.register(name="retrieval_recall_at_fixed_precision", module=RetrievalRecallAtFixedPrecision)
METRICS.register(name="retrieval_r_precision",               module=RetrievalRPrecision)

# endregion
