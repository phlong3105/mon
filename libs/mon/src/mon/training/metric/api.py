#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""External APIs.

This module collects all external functionalities that are commonly used in
this package. It is intended to be imported by other modules for convenience.
"""

from __future__ import annotations

__all__ = [
    "AUROC",
    "Accuracy",
    "AveragePrecision",
    "BLEUScore",
    "BootStrapper",
    "CHRFScore",
    "CalibrationError",
    "CatMetric",
    "CharErrorRate",
    "ClasswiseWrapper",
    "CohenKappa",
    "ConcordanceCorrCoef",
    "ConfusionMatrix",
    "CosineSimilarity",
    "CramersV",
    "CriticalSuccessIndex",
    "ErrorRelativeGlobalDimensionlessSynthesis",
    "ExactMatch",
    "ExplainedVariance",
    "ExtendedEditDistance",
    "F1Score",
    "FBetaScore",
    "FleissKappa",
    "HammingDistance",
    "HingeLoss",
    "JaccardIndex",
    "KLDivergence",
    "KendallRankCorrCoef",
    "LogAUC",
    "LogCoshError",
    "MatchErrorRate",
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
    "ModifiedPanopticQuality",
    "MultiScaleStructuralSimilarityIndexMeasure",
    "MultioutputWrapper",
    "MultitaskWrapper",
    "NegativePredictiveValue",
    "NormalizedRootMeanSquaredError",
    "PanopticQuality",
    "PeakSignalNoiseRatio",
    "PearsonCorrCoef",
    "PearsonsContingencyCoefficient",
    "PermutationInvariantTraining",
    "Perplexity",
    "Precision",
    "PrecisionAtFixedRecall",
    "PrecisionRecallCurve",
    "R2Score",
    "ROC",
    "Recall",
    "RecallAtFixedPrecision",
    "RelativeAverageSpectralError",
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
    "RootMeanSquaredErrorUsingSlidingWindow",
    "RunningMean",
    "RunningSum",
    "SQuAD",
    "SacreBLEUScore",
    "ScaleInvariantSignalDistortionRatio",
    "ScaleInvariantSignalNoiseRatio",
    "SensitivityAtSpecificity",
    "SignalDistortionRatio",
    "SignalNoiseRatio",
    "SpearmanCorrCoef",
    "Specificity",
    "SpecificityAtSensitivity",
    "SpectralAngleMapper",
    "SpectralDistortionIndex",
    "StatScores",
    "StructuralSimilarityIndexMeasure",
    "SumMetric",
    "SymmetricMeanAbsolutePercentageError",
    "TheilsU",
    "TotalVariation",
    "TranslationEditRate",
    "TschuprowsT",
    "TweedieDevianceScore",
    "UniversalImageQualityIndex",
    "WeightedMeanAbsolutePercentageError",
    "WordErrorRate",
    "WordInfoLost",
    "WordInfoPreserved",
]

from torchmetrics import *


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
