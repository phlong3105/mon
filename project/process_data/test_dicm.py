#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

import mon

dataset     = mon.DICM(split=mon.Split.TEST, verbose=True)
iterator    = iter(dataset)
datapoint   = next(iterator)
classlabels = dataset.classlabels
image       = datapoint["image"]
depth       = datapoint["depth"]
