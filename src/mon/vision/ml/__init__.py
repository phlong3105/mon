#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package provides the base machine learning and deep learning components
for vision tasks. We try to make this package as generic as possible. For
specific modules and layers, just implement them in the corresponding model.

This package is built on top of :mod:`mon.coreml`. Think of it as an interface
to :mod:`mon.coreml` module.

The development process should be as follows:
    1. Implement specific modules or layers in their dedicated model file.
    2. Refactor any modules or layers that are used across several models in
       this package.
    3. Refactor any modules or layers that are used across several packages
       (i.e., vision and text) in the package :mod:`mon.coreml`.
"""

import mon.vision.ml.layer
import mon.vision.ml.loss
import mon.vision.ml.metric
# noinspection PyUnresolvedReferences
from mon.coreml import *
from mon.vision.ml.layer import *
from mon.vision.ml.loss import *
from mon.vision.ml.metric import *
