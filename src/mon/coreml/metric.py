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

from mon.coreml import constant

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

constant.METRIC.register(name="accuracy",               module=Accuracy)
constant.METRIC.register(name="average_precision",      module=AveragePrecision)
constant.METRIC.register(name="auroc",                  module=AUROC)
constant.METRIC.register(name="calibration_error",      module=CalibrationError)
constant.METRIC.register(name="cohen_kappa",            module=CohenKappa)
constant.METRIC.register(name="confusion_matrix",       module=ConfusionMatrix)
constant.METRIC.register(name="dice",                   module=Dice)
constant.METRIC.register(name="exact_match",            module=ExactMatch)
constant.METRIC.register(name="f1_score ",              module=F1Score)
constant.METRIC.register(name="f_beta_score",           module=FBetaScore)
constant.METRIC.register(name="hamming_distance",       module=HammingDistance)
constant.METRIC.register(name="hinge_loss",             module=HingeLoss)
constant.METRIC.register(name="jaccard_index",          module=JaccardIndex)
constant.METRIC.register(name="matthews_corr_coef",     module=MatthewsCorrCoef)
constant.METRIC.register(name="precision",              module=Precision)
constant.METRIC.register(name="precision_recall_curve", module=PrecisionRecallCurve)
constant.METRIC.register(name="recall",                 module=Recall)
constant.METRIC.register(name="roc",                    module=ROC)
constant.METRIC.register(name="specificity",            module=Specificity)
constant.METRIC.register(name="stat_scores",            module=StatScores)

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

constant.METRIC.register(name="error_relative_global_dimensionless_synthesis",  module=ErrorRelativeGlobalDimensionlessSynthesis)
constant.METRIC.register(name="multiscale_structural_similarity_index_measure", module=MultiScaleStructuralSimilarityIndexMeasure)
constant.METRIC.register(name="multiscale_ssim",                                module=MultiScaleStructuralSimilarityIndexMeasure)
constant.METRIC.register(name="peak_signal_noise_ratio",                        module=PeakSignalNoiseRatio)
constant.METRIC.register(name="psnr",                                           module=PeakSignalNoiseRatio)
constant.METRIC.register(name="spectral_angle_mapper",                          module=SpectralAngleMapper)
constant.METRIC.register(name="spectral_distortion_index",                      module=SpectralDistortionIndex)
constant.METRIC.register(name="structural_similarity_index_measure",            module=StructuralSimilarityIndexMeasure)
constant.METRIC.register(name="ssim",                                           module=StructuralSimilarityIndexMeasure)
constant.METRIC.register(name="total_variation",                                module=TotalVariation)
constant.METRIC.register(name="universal_image_quality_index",                  module=UniversalImageQualityIndex)

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

constant.METRIC.register(name="concordance_corr_coef",                    module=ConcordanceCorrCoef)
constant.METRIC.register(name="cosine_similarity",                        module=CosineSimilarity)
constant.METRIC.register(name="explained_variance",                       module=ExplainedVariance)
constant.METRIC.register(name="kendall_rank_corr_coef",                   module=KendallRankCorrCoef)
constant.METRIC.register(name="kl_divergence",                            module=KLDivergence)
constant.METRIC.register(name="log_cosh_error",                           module=LogCoshError)
constant.METRIC.register(name="mean_absolute_error",                      module=MeanAbsoluteError)
constant.METRIC.register(name="mae",                                      module=MeanAbsoluteError)
constant.METRIC.register(name="mean_absolute_percentage_error",           module=MeanAbsolutePercentageError)
constant.METRIC.register(name="mean_squared_error",                       module=MeanSquaredError)
constant.METRIC.register(name="mse",                                      module=MeanSquaredError)
constant.METRIC.register(name="mean_squared_log_error",                   module=MeanSquaredLogError)
constant.METRIC.register(name="pearson_corr_coef",                        module=PearsonCorrCoef)
constant.METRIC.register(name="r2_score",                                 module=R2Score)
constant.METRIC.register(name="spearman_corr_coef",                       module=SpearmanCorrCoef)
constant.METRIC.register(name="symmetric_mean_absolute_percentage_error", module=SymmetricMeanAbsolutePercentageError)
constant.METRIC.register(name="tweedie_deviance_score",                   module=TweedieDevianceScore)
constant.METRIC.register(name="weighted_mean_absolute_percentage_error",  module=WeightedMeanAbsolutePercentageError)

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

constant.METRIC.register(name="retrieval_fall_out",                  module=RetrievalFallOut)
constant.METRIC.register(name="retrieval_hit_rate",                  module=RetrievalHitRate)
constant.METRIC.register(name="retrieval_map",                       module=RetrievalMAP)
constant.METRIC.register(name="retrieval_mrr",                       module=RetrievalMRR)
constant.METRIC.register(name="retrieval_normalized_dcg",            module=RetrievalNormalizedDCG)
constant.METRIC.register(name="retrieval_precision",                 module=RetrievalPrecision)
constant.METRIC.register(name="retrieval_precision_recall_curve",    module=RetrievalPrecisionRecallCurve)
constant.METRIC.register(name="retrieval_recall",                    module=RetrievalRecall)
constant.METRIC.register(name="retrieval_recall_at_fixed_precision", module=RetrievalRecallAtFixedPrecision)
constant.METRIC.register(name="retrieval_r_precision",               module=RetrievalRPrecision)

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

constant.METRIC.register(name="blue_score",              module=BLEUScore)
constant.METRIC.register(name="char_error_rate",         module=CharErrorRate)
constant.METRIC.register(name="chrf_score",              module=CHRFScore)
constant.METRIC.register(name="extended_edit_distance",  module=ExtendedEditDistance)
constant.METRIC.register(name="match_error_rate",        module=MatchErrorRate)
constant.METRIC.register(name="perplexity",              module=Perplexity)
constant.METRIC.register(name="pacre_blue_score",        module=SacreBLEUScore)
constant.METRIC.register(name="squad",                   module=SQuAD)
constant.METRIC.register(name="translation_edit_rate",   module=TranslationEditRate)
constant.METRIC.register(name="word_error_rate",         module=WordErrorRate)
constant.METRIC.register(name="word_info_lost",          module=WordInfoLost)
constant.METRIC.register(name="word_info_preserved",     module=WordInfoPreserved)

# endregion
