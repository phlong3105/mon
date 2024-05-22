#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import glob
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

random.seed(1143)


class LowlightLoader(data.Dataset):
	
	def __init__(self, images_path: str):
		
		image_paths = glob.glob(images_path + "*.jpg")
		random.shuffle(image_paths)
		self.size        = 256
		self.image_paths = image_paths
		print("Total training examples:", len(self.image_paths))
	
	def __getitem__(self, index: int):
		image_path = self.image_paths[index]
		image_lq   = Image.open(image_path)
		image_lq   = image_lq.resize((self.size, self.size), Image.ANTIALIAS)
		image_lq   = (np.asarray(image_lq) / 255.0)
		image_lq   = torch.from_numpy(image_lq).float()
		return image_lq.permute(2, 0, 1)
	
	def __len__(self):
		return len(self.image_paths)
