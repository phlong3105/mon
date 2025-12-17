#!/usr/bin/env python
# -*- coding: utf-8 -*-

import albumentations as A
import pandas as pd
from torch.utils.data import Dataset

from utils import cv_to_tensor, read_image

transform = A.Compose(
    [
        A.Flip(p=0.8),
        A.ShiftScaleRotate(p=1.0,rotate_limit=180, shift_limit = 0, scale_limit = 0),
        A.RandomBrightnessContrast(p=0.5),

    ],
    additional_targets={'wet': 'image'}
)


class imageDebandDataset(Dataset):

    def __init__(self, csv, root_dir="", test=False, transform=None):
        self.frame     = pd.read_csv(csv)
        self.root_dir  = root_dir
        self.test      = test
        self.transform = transform

    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, idx):
        dry = self.frame.loc[idx, "dry"]
        wet = self.frame.loc[idx, "banded"]
        d   = read_image(dry, to_tensor=False)
        w   = read_image(wet, to_tensor=False)
        if not self.test and transform:
            transformed = transform(image = d, wet = w)
            sample = {
                "dry": cv_to_tensor(transformed["image"]),
                "wet": cv_to_tensor(transformed["wet"])
            }
        else:
            sample = {
                "dry": cv_to_tensor(d),
                "wet": cv_to_tensor(w)
            }
        return sample
