#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test CV Image Processing operations.
"""

from __future__ import annotations

import numpy as np

from one import cosine_distance
from one import euclidean_distance

import pickle as pkl

import cv2
import torch

from one import adjust_gamma
from one import read_image_cv
from one import rotate_image_box
from one import to_channel_first
from one import to_channel_last


# MARK: - Test Distance Functions

def test_cosine_distance():
	x = np.array((1, 2, 3))
	y = np.array((1, 1, 1))
	print(cosine_distance(x, y))


def test_euclidean_distance():
	x = np.array((1, 2, 3))
	y = np.array((1, 1, 1))
	print(euclidean_distance(x, y))


# MARK: - Test Transformation

def draw_rect(im, cords, color=None):
	im = im.copy()
	cords = cords[:, :4]
	cords = cords.reshape(-1, 4)
	
	if not color:
		color = [255, 255, 255]
	for cord in cords:
		pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])
		pt1 = int(pt1[0]), int(pt1[1])
		pt2 = int(pt2[0]), int(pt2[1])
		im = cv2.rectangle(im.copy(), pt1, pt2, color, int(max(im.shape[:2]) / 200))
		print(pt1, pt2)
	return im


def test_image_box_transformation():
	image  = read_image_cv("../data/messi.jpeg")
	image  = image[:, :, ::-1]
	bboxes = pkl.load(open("../data/messi_ann.pkl", "rb"))
	bboxes = bboxes[:, 0:4]
	
	image  = to_channel_first(image)
	tensor = torch.from_numpy(image)
	bboxes = torch.from_numpy(bboxes)
	
	image  = adjust_gamma(image,  0.5)
	tensor = adjust_gamma(tensor, 0.5)
	# image  = affine(image,  angle=20, translate=[50, 50], scale=1.0, shear=[0, 0])
	# tensor = affine(tensor, angle=20, translate=[50, 50], scale=1.0, shear=[0, 0])

	tensor, bboxes = rotate_image_box(tensor, bboxes, 45)

	image  = to_channel_last(image)
	tensor = tensor.numpy()
	tensor = to_channel_last(tensor)
	bboxes = bboxes.numpy()

	cv2.imshow("numpy",  image)
	cv2.imshow("tensor", tensor)
	cv2.imshow("box",    draw_rect(tensor, bboxes))
	cv2.waitKey(0)
