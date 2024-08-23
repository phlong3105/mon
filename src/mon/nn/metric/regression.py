#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Regression Metrics.

This module implements regression metrics.
"""

from __future__ import annotations

__all__ = [
    "ConcordanceCorrCoef",
    "CosineSimilarity",
    "ExplainedVariance",
    "KendallRankCorrCoef",
    "KLDivergence",
    "LogCoshError",
    "MeanSquaredLogError",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MinkowskiDistance",
    "MeanSquaredError",
    "PearsonCorrCoef",
    "R2Score",
    "RelativeSquaredError",
    "SpearmanCorrCoef",
    "SymmetricMeanAbsolutePercentageError",
    "TweedieDevianceScore",
    "WeightedMeanAbsolutePercentageError",
]

import torchmetrics

from mon.globals import METRICS

# region Regression Metric

ConcordanceCorrCoef                  = torchmetrics.regression.ConcordanceCorrCoef
CosineSimilarity                     = torchmetrics.regression.CosineSimilarity
ExplainedVariance                    = torchmetrics.regression.ExplainedVariance
KLDivergence                         = torchmetrics.regression.KLDivergence
KendallRankCorrCoef                  = torchmetrics.regression.KendallRankCorrCoef
LogCoshError                         = torchmetrics.regression.LogCoshError
MeanAbsoluteError                    = torchmetrics.regression.MeanAbsoluteError
MeanAbsolutePercentageError          = torchmetrics.regression.MeanAbsolutePercentageError
MeanSquaredError                     = torchmetrics.regression.MeanSquaredError
MeanSquaredLogError                  = torchmetrics.regression.MeanSquaredLogError
MinkowskiDistance                    = torchmetrics.regression.MinkowskiDistance
PearsonCorrCoef                      = torchmetrics.regression.PearsonCorrCoef
R2Score                              = torchmetrics.regression.R2Score
RelativeSquaredError                 = torchmetrics.regression.RelativeSquaredError
SpearmanCorrCoef                     = torchmetrics.regression.SpearmanCorrCoef
SymmetricMeanAbsolutePercentageError = torchmetrics.regression.SymmetricMeanAbsolutePercentageError
TweedieDevianceScore                 = torchmetrics.regression.TweedieDevianceScore
WeightedMeanAbsolutePercentageError  = torchmetrics.regression.WeightedMeanAbsolutePercentageError

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
