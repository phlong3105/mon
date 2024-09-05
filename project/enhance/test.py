#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import mon

dataset     = mon.NightCity(split=mon.Split.VAL)
iterator    = iter(dataset)
datapoint   = next(iterator)
classlabels = dataset.classlabels
image       = datapoint["image"]
semantic    = datapoint["semantic"]
# color_mask  = mon.parse_color_to_label_ids(semantic,   classlabels)
color_mask  = mon.label_map_id_to_color(semantic, classlabels)
color_mask  = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
cv2.imshow("Color Mask", color_mask)
cv2.waitKey(0)
