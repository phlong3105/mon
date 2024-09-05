#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import torch

import mon

dataset     = mon.NightCity(split=mon.Split.VAL)
iterator    = iter(dataset)
datapoint   = next(iterator)
datapoint2  = next(iterator)
classlabels = dataset.classlabels
image       = datapoint["image"]
semantic    = datapoint["semantic"]
semantic2   = datapoint["semantic"]
semantic_t  = mon.to_image_tensor(semantic)
semantic2_t = mon.to_image_tensor(semantic)
concat      = torch.concat([semantic_t, semantic2_t], dim=0)
one_hot     = mon.label_map_id_to_one_hot(semantic, classlabels=classlabels)
one_hot_t   = mon.label_map_id_to_one_hot(concat,   classlabels=classlabels)
print(one_hot.shape, one_hot_t.shape)
color_mask  = mon.draw_semantic(image, semantic, classlabels)
color_mask  = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
cv2.imshow("Color Mask", color_mask)
cv2.waitKey(0)
