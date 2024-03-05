#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements nominal metrics."""

from __future__ import annotations

__all__ = [
    "CramersV",
    "FleissKappa",
    "PearsonsContingencyCoefficient",
    "TheilsU",
    "TschuprowsT",
]

import torchmetrics

from mon.globals import METRICS

# region Nominal Metric

CramersV                       = torchmetrics.nominal.CramersV
FleissKappa                    = torchmetrics.nominal.FleissKappa
PearsonsContingencyCoefficient = torchmetrics.nominal.PearsonsContingencyCoefficient
TheilsU                        = torchmetrics.nominal.TheilsU
TschuprowsT                    = torchmetrics.nominal.TschuprowsT

METRICS.register(name="cramers_v",                        module=CramersV)
METRICS.register(name="fleiss_kappa",                     module=FleissKappa)
METRICS.register(name="pearsons_contingency_coefficient", module=PearsonsContingencyCoefficient)
METRICS.register(name="theils_u",                         module=TheilsU)
METRICS.register(name="tschuprows_t",                     module=TschuprowsT)

# endregion
