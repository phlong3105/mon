#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torchmetrics

from one.core import METRICS

# MARK: - Register (torchmetrics)

# MARK: Classification Metrics

METRICS.register(name="accuracy",                         module=torchmetrics.Accuracy)
METRICS.register(name="average_precision",                module=torchmetrics.AveragePrecision)
METRICS.register(name="auc",                              module=torchmetrics.AUC)
METRICS.register(name="auroc",                            module=torchmetrics.AUROC)
METRICS.register(name="binned_average_precision",         module=torchmetrics.BinnedAveragePrecision)
METRICS.register(name="binned_precision_recall_curve",    module=torchmetrics.BinnedPrecisionRecallCurve)
METRICS.register(name="binned_recall_at_fixed_precision", module=torchmetrics.BinnedRecallAtFixedPrecision)
METRICS.register(name="calibration_error",                module=torchmetrics.CalibrationError)
METRICS.register(name="cohen_kappa",                      module=torchmetrics.CohenKappa)
METRICS.register(name="confusion_matrix",                 module=torchmetrics.ConfusionMatrix)
METRICS.register(name="coverage_error",                   module=torchmetrics.CoverageError)
METRICS.register(name="f1_score",                         module=torchmetrics.F1Score)
METRICS.register(name="fbeta_Score",                      module=torchmetrics.FBetaScore)
METRICS.register(name="hamming_distance",                 module=torchmetrics.HammingDistance)
METRICS.register(name="hinge_loss",                       module=torchmetrics.HingeLoss)
METRICS.register(name="jaccard_index",                    module=torchmetrics.JaccardIndex)
METRICS.register(name="kl_divergence",                    module=torchmetrics.KLDivergence)
METRICS.register(name="label_ranking_average_precision",  module=torchmetrics.LabelRankingAveragePrecision)
METRICS.register(name="label_ranking_loss",               module=torchmetrics.LabelRankingLoss)
METRICS.register(name="matthews_corr_coef",               module=torchmetrics.MatthewsCorrCoef)
METRICS.register(name="precision",                        module=torchmetrics.Precision)
METRICS.register(name="precision_recall_curve",           module=torchmetrics.PrecisionRecallCurve)
METRICS.register(name="recall",                           module=torchmetrics.Recall)
METRICS.register(name="roc",                              module=torchmetrics.ROC)
METRICS.register(name="specificity",                      module=torchmetrics.Specificity)
METRICS.register(name="stat_scores",                      module=torchmetrics.StatScores)


# MARK: Image Metrics

METRICS.register(name="error_relative_global_dimensionless_synthesis",
                 module=torchmetrics.ErrorRelativeGlobalDimensionlessSynthesis)
METRICS.register(name="multi_scale_ssim",
                 module=torchmetrics.MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register(name="multi_scale_structural_similarity_index_measure",
                 module=torchmetrics.MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register(name="psnr",
                 module=torchmetrics.PeakSignalNoiseRatio)
METRICS.register(name="peak_signal_noise_ratio",
                 module=torchmetrics.PeakSignalNoiseRatio)
METRICS.register(name="spectral_angle_mapper",
                 module=torchmetrics.SpectralAngleMapper)
METRICS.register(name="spectral_distortion_index",
                 module=torchmetrics.SpectralDistortionIndex)
METRICS.register(name="ssim",
                 module=torchmetrics.StructuralSimilarityIndexMeasure)
METRICS.register(name="structural_similarity_index_measure",
                 module=torchmetrics.StructuralSimilarityIndexMeasure)
METRICS.register(name="universal_image_quality_index",
                 module=torchmetrics.UniversalImageQualityIndex)


# MARK: Regression Metrics

METRICS.register(name="cosine_similarity",                        module=torchmetrics.CosineSimilarity)
METRICS.register(name="explained_variance",                       module=torchmetrics.ExplainedVariance)
METRICS.register(name="mean_absolute_error",                      module=torchmetrics.MeanAbsoluteError)
METRICS.register(name="mean_absolute_percentage_error",           module=torchmetrics.MeanAbsolutePercentageError)
METRICS.register(name="mean_squared_error",                       module=torchmetrics.MeanSquaredError)
METRICS.register(name="mean_squared_log_error",                   module=torchmetrics.MeanSquaredLogError)
METRICS.register(name="pearson_corr_coef",                        module=torchmetrics.PearsonCorrCoef)
METRICS.register(name="r2_score",                                 module=torchmetrics.R2Score)
METRICS.register(name="spearman_corr_coef",                       module=torchmetrics.SpearmanCorrCoef)
METRICS.register(name="symmetric_mean_absolute_percentage_error", module=torchmetrics.SymmetricMeanAbsolutePercentageError)
METRICS.register(name="tweedie_deviance_score",                   module=torchmetrics.TweedieDevianceScore)
METRICS.register(name="weighted_mean_absolute_percentage_error",  module=torchmetrics.WeightedMeanAbsolutePercentageError)


# MARK: Retrieval Metrics

METRICS.register(name="retrieval_fallout",        module=torchmetrics.RetrievalFallOut)
METRICS.register(name="retrieval_hit_rate",       module=torchmetrics.RetrievalHitRate)
METRICS.register(name="retrieval_map",            module=torchmetrics.RetrievalMAP)
METRICS.register(name="retrieval_mrr",            module=torchmetrics.RetrievalMRR)
METRICS.register(name="retrieval_normalized_dcg", module=torchmetrics.RetrievalNormalizedDCG)
METRICS.register(name="retrieval_precision",      module=torchmetrics.RetrievalPrecision)
METRICS.register(name="retrieval_recall",         module=torchmetrics.RetrievalRecall)
METRICS.register(name="retrieval_r_precision",    module=torchmetrics.RetrievalRPrecision)
