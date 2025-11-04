#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements model training.

Model training is the process of “teaching” a machine learning model to optimize
performance on a training dataset of sample tasks relevant to the model’s eventual
use cases. If training data closely resembles real-world problems that the model
will be tasked with, learning its patterns and correlations will enable a trained
model to make accurate predictions on new data.

References:
    - https://www.ibm.com/think/topics/model-training#1580786329
"""

from mon.training import (
    albumentations as albumentations,
    foundation as foundation,
    losses as losses,
    metrics as metrics,
    optims as optims,
    runtime as rt,
)
