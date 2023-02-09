#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements evaluation metrics using the :mod:`torchmetrics`
package.
"""

from __future__ import annotations

__all__ = [
    "AUROC", "Accuracy", "AveragePrecision", "BLEUScore", "BootStrapper",
    "CHRFScore", "CalibrationError", "CatMetric", "CharErrorRate",
    "ClasswiseWrapper", "CohenKappa", "ConcordanceCorrCoef", "ConfusionMatrix",
    "CosineSimilarity", "Dice", "ErrorRelativeGlobalDimensionlessSynthesis",
    "ExactMatch", "ExplainedVariance", "ExtendedEditDistance", "F1Score",
    "FBetaScore", "HammingDistance", "HingeLoss", "JaccardIndex",
    "KLDivergence", "KendallRankCorrCoef", "LogCoshError", "MatchErrorRate",
    "MatthewsCorrCoef", "MaxMetric", "MeanAbsoluteError",
    "MeanAbsolutePercentageError", "MeanMetric", "MeanSquaredError",
    "MeanSquaredLogError", "Metric", "MetricCollection", "MetricTracker",
    "MinMaxMetric", "MinMetric", "MultiScaleStructuralSimilarityIndexMeasure",
    "MultioutputWrapper", "PeakSignalNoiseRatio", "PearsonCorrCoef",
    "Perplexity", "Precision", "PrecisionRecallCurve", "R2Score", "ROC",
    "Recall", "RetrievalFallOut", "RetrievalHitRate", "RetrievalMAP",
    "RetrievalMRR", "RetrievalNormalizedDCG", "RetrievalPrecision",
    "RetrievalPrecisionRecallCurve", "RetrievalRPrecision", "RetrievalRecall",
    "RetrievalRecallAtFixedPrecision", "SQuAD", "SacreBLEUScore",
    "SpearmanCorrCoef", "Specificity", "SpectralAngleMapper",
    "SpectralDistortionIndex", "StatScores", "StructuralSimilarityIndexMeasure",
    "SumMetric", "SymmetricMeanAbsolutePercentageError", "TotalVariation",
    "TranslationEditRate", "TweedieDevianceScore", "UniversalImageQualityIndex",
    "WeightedMeanAbsolutePercentageError", "WordErrorRate", "WordInfoLost",
    "WordInfoPreserved",
]

import torchmetrics

from mon.globals import METRICS

Metric           = torchmetrics.Metric
MetricCollection = torchmetrics.MetricCollection


# region Aggregation

CatMetric  = torchmetrics.CatMetric
MaxMetric  = torchmetrics.MaxMetric
MeanMetric = torchmetrics.MeanMetric
MinMetric  = torchmetrics.MinMetric
SumMetric  = torchmetrics.SumMetric

# endregion


# region Wrapper

BootStrapper         = torchmetrics.BootStrapper
ClasswiseWrapper     = torchmetrics.ClasswiseWrapper
MetricTracker        = torchmetrics.MetricTracker
MinMaxMetric         = torchmetrics.MinMaxMetric
MultioutputWrapper   = torchmetrics.MultioutputWrapper

# endregion


# region Classification Metric

Accuracy             = torchmetrics.Accuracy
AveragePrecision     = torchmetrics.AveragePrecision
AUROC                = torchmetrics.AUROC
CalibrationError     = torchmetrics.CalibrationError
CohenKappa           = torchmetrics.CohenKappa
ConfusionMatrix      = torchmetrics.ConfusionMatrix
Dice                 = torchmetrics.Dice
ExactMatch           = torchmetrics.ExactMatch
F1Score              = torchmetrics.F1Score
FBetaScore           = torchmetrics.FBetaScore
HammingDistance      = torchmetrics.HammingDistance
HingeLoss            = torchmetrics.HingeLoss
JaccardIndex         = torchmetrics.JaccardIndex
MatthewsCorrCoef     = torchmetrics.MatthewsCorrCoef
Precision            = torchmetrics.Precision
PrecisionRecallCurve = torchmetrics.PrecisionRecallCurve
Recall               = torchmetrics.Recall              
ROC                  = torchmetrics.ROC
Specificity          = torchmetrics.Specificity
StatScores           = torchmetrics.StatScores

METRICS.register(name="accuracy",               module=Accuracy)
METRICS.register(name="average_precision",      module=AveragePrecision)
METRICS.register(name="auroc",                  module=AUROC)
METRICS.register(name="calibration_error",      module=CalibrationError)
METRICS.register(name="cohen_kappa",            module=CohenKappa)
METRICS.register(name="confusion_matrix",       module=ConfusionMatrix)
METRICS.register(name="dice",                   module=Dice)
METRICS.register(name="exact_match",            module=ExactMatch)
METRICS.register(name="f1_score ",              module=F1Score)
METRICS.register(name="f_beta_score",           module=FBetaScore)
METRICS.register(name="hamming_distance",       module=HammingDistance)
METRICS.register(name="hinge_loss",             module=HingeLoss)
METRICS.register(name="jaccard_index",          module=JaccardIndex)
METRICS.register(name="matthews_corr_coef",     module=MatthewsCorrCoef)
METRICS.register(name="precision",              module=Precision)
METRICS.register(name="precision_recall_curve", module=PrecisionRecallCurve)
METRICS.register(name="recall",                 module=Recall)
METRICS.register(name="roc",                    module=ROC)
METRICS.register(name="specificity",            module=Specificity)
METRICS.register(name="stat_scores",            module=StatScores)

# endregion


# region Image Metric

ErrorRelativeGlobalDimensionlessSynthesis  = torchmetrics.ErrorRelativeGlobalDimensionlessSynthesis
MultiScaleStructuralSimilarityIndexMeasure = torchmetrics.MultiScaleStructuralSimilarityIndexMeasure
PeakSignalNoiseRatio                       = torchmetrics.PeakSignalNoiseRatio
SpectralAngleMapper                        = torchmetrics.SpectralAngleMapper
SpectralDistortionIndex                    = torchmetrics.SpectralDistortionIndex
StructuralSimilarityIndexMeasure           = torchmetrics.StructuralSimilarityIndexMeasure
TotalVariation                             = torchmetrics.TotalVariation
UniversalImageQualityIndex                 = torchmetrics.UniversalImageQualityIndex

METRICS.register(name="error_relative_global_dimensionless_synthesis",  module=ErrorRelativeGlobalDimensionlessSynthesis)
METRICS.register(name="multiscale_structural_similarity_index_measure", module=MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register(name="multiscale_ssim",                                module=MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register(name="peak_signal_noise_ratio",                        module=PeakSignalNoiseRatio)
METRICS.register(name="psnr",                                           module=PeakSignalNoiseRatio)
METRICS.register(name="spectral_angle_mapper",                          module=SpectralAngleMapper)
METRICS.register(name="spectral_distortion_index",                      module=SpectralDistortionIndex)
METRICS.register(name="structural_similarity_index_measure",            module=StructuralSimilarityIndexMeasure)
METRICS.register(name="ssim",                                           module=StructuralSimilarityIndexMeasure)
METRICS.register(name="total_variation",                                module=TotalVariation)
METRICS.register(name="universal_image_quality_index",                  module=UniversalImageQualityIndex)

# endregion


# region Regression Metric

ConcordanceCorrCoef                  = torchmetrics.ConcordanceCorrCoef
CosineSimilarity                     = torchmetrics.CosineSimilarity
ExplainedVariance                    = torchmetrics.ExplainedVariance
KendallRankCorrCoef                  = torchmetrics.KendallRankCorrCoef
KLDivergence                         = torchmetrics.KLDivergence
LogCoshError                         = torchmetrics.LogCoshError
MeanAbsoluteError                    = torchmetrics.MeanAbsoluteError
MeanAbsolutePercentageError          = torchmetrics.MeanAbsolutePercentageError
MeanSquaredError                     = torchmetrics.MeanSquaredError
MeanSquaredLogError                  = torchmetrics.MeanSquaredLogError
PearsonCorrCoef                      = torchmetrics.PearsonCorrCoef
R2Score                              = torchmetrics.R2Score
SpearmanCorrCoef                     = torchmetrics.SpearmanCorrCoef
SymmetricMeanAbsolutePercentageError = torchmetrics.SymmetricMeanAbsolutePercentageError
TweedieDevianceScore                 = torchmetrics.TweedieDevianceScore
WeightedMeanAbsolutePercentageError  = torchmetrics.WeightedMeanAbsolutePercentageError

METRICS.register(name="concordance_corr_coef",                    module=ConcordanceCorrCoef)
METRICS.register(name="cosine_similarity",                        module=CosineSimilarity)
METRICS.register(name="explained_variance",                       module=ExplainedVariance)
METRICS.register(name="kendall_rank_corr_coef",                   module=KendallRankCorrCoef)
METRICS.register(name="kl_divergence",                            module=KLDivergence)
METRICS.register(name="log_cosh_error",                           module=LogCoshError)
METRICS.register(name="mean_absolute_error",                      module=MeanAbsoluteError)
METRICS.register(name="mae",                                      module=MeanAbsoluteError)
METRICS.register(name="mean_absolute_percentage_error",           module=MeanAbsolutePercentageError)
METRICS.register(name="mean_squared_error",                       module=MeanSquaredError)
METRICS.register(name="mse",                                      module=MeanSquaredError)
METRICS.register(name="mean_squared_log_error",                   module=MeanSquaredLogError)
METRICS.register(name="pearson_corr_coef",                        module=PearsonCorrCoef)
METRICS.register(name="r2_score",                                 module=R2Score)
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


# region Text Metric

BLEUScore            = torchmetrics.BLEUScore
CharErrorRate        = torchmetrics.CharErrorRate
CHRFScore            = torchmetrics.CHRFScore
ExtendedEditDistance = torchmetrics.ExtendedEditDistance
MatchErrorRate       = torchmetrics.MatchErrorRate
Perplexity           = torchmetrics.Perplexity
SacreBLEUScore       = torchmetrics.SacreBLEUScore
SQuAD                = torchmetrics.SQuAD
TranslationEditRate  = torchmetrics.TranslationEditRate
WordErrorRate        = torchmetrics.WordErrorRate
WordInfoLost         = torchmetrics.WordInfoLost
WordInfoPreserved    = torchmetrics.WordInfoPreserved

METRICS.register(name="blue_score",              module=BLEUScore)
METRICS.register(name="char_error_rate",         module=CharErrorRate)
METRICS.register(name="chrf_score",              module=CHRFScore)
METRICS.register(name="extended_edit_distance",  module=ExtendedEditDistance)
METRICS.register(name="match_error_rate",        module=MatchErrorRate)
METRICS.register(name="perplexity",              module=Perplexity)
METRICS.register(name="pacre_blue_score",        module=SacreBLEUScore)
METRICS.register(name="squad",                   module=SQuAD)
METRICS.register(name="translation_edit_rate",   module=TranslationEditRate)
METRICS.register(name="word_error_rate",         module=WordErrorRate)
METRICS.register(name="word_info_lost",          module=WordInfoLost)
METRICS.register(name="word_info_preserved",     module=WordInfoPreserved)

# endregion
