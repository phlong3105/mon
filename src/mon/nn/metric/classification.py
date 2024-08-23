#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classification Metric Module.

This module implements classification metrics.
"""

from __future__ import annotations

__all__ = [
    "AUROC",
    "Accuracy",
    "AveragePrecision",
    "CalibrationError",
    "CohenKappa",
    "ConfusionMatrix",
    "Dice",
    "ExactMatch",
    "F1Score",
    "FBetaScore",
    "HammingDistance",
    "HingeLoss",
    "JaccardIndex",
    "MatthewsCorrCoef",
    "Precision",
    "PrecisionAtFixedRecall",
    "PrecisionRecallCurve",
    "ROC",
    "Recall",
    "RecallAtFixedPrecision",
    "Specificity",
    "SpecificityAtSensitivity",
    "StatScores",
]

import torchmetrics

from mon.globals import METRICS


# region Classification Metric

AUROC                    = torchmetrics.classification.AUROC
Accuracy                 = torchmetrics.classification.Accuracy
AveragePrecision         = torchmetrics.classification.AveragePrecision
CalibrationError         = torchmetrics.classification.CalibrationError
CohenKappa               = torchmetrics.classification.CohenKappa
ConfusionMatrix          = torchmetrics.classification.ConfusionMatrix
Dice                     = torchmetrics.classification.Dice
ExactMatch               = torchmetrics.classification.ExactMatch
F1Score                  = torchmetrics.classification.F1Score
FBetaScore               = torchmetrics.classification.FBetaScore
HammingDistance          = torchmetrics.classification.HammingDistance
HingeLoss                = torchmetrics.classification.HingeLoss
JaccardIndex             = torchmetrics.classification.JaccardIndex
MatthewsCorrCoef         = torchmetrics.classification.MatthewsCorrCoef
Precision                = torchmetrics.classification.Precision
PrecisionAtFixedRecall   = torchmetrics.classification.PrecisionAtFixedRecall
PrecisionRecallCurve     = torchmetrics.classification.PrecisionRecallCurve
ROC                      = torchmetrics.classification.ROC
Recall                   = torchmetrics.classification.Recall
RecallAtFixedPrecision   = torchmetrics.classification.RecallAtFixedPrecision
Specificity              = torchmetrics.classification.Specificity
SpecificityAtSensitivity = torchmetrics.classification.SpecificityAtSensitivity
StatScores               = torchmetrics.classification.StatScores

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
