#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

import mon

dataset     = mon.Cityscapes(split=mon.Split.VAL, verbose=True)
iterator    = iter(dataset)
datapoint   = next(iterator)
classlabels = dataset.classlabels
image       = datapoint["image"]
semantic    = datapoint["semantic"]
semantic_t  = mon.to_image_tensor(semantic)
one_hot     = mon.label_map_id_to_one_hot(semantic,   classlabels=classlabels)
one_hot_t   = mon.label_map_id_to_one_hot(semantic_t, classlabels=classlabels)
color_mask  = mon.draw_semantic(image, semantic, classlabels)
color_mask  = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
print(one_hot.shape, one_hot_t.shape)
cv2.imshow("Color Mask", color_mask)
cv2.waitKey(0)
