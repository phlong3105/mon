#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from utils import is_png_file, load_img, Augment_RGB_torch

import mon

augment        = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith("_")] 


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ["jpeg", "JPEG", "jpg", "png", "JPG", "PNG", "gif"])


##################################################################################################

class DataLoaderTrain(Dataset):

    def __init__(self, input_dir, target_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()
        self.target_transform = target_transform
        
        input_dir    = mon.Path(input_dir)
        input_files  = sorted(list(input_dir.rglob("*")))
        input_files  = [str(x) for x in input_files]
                     
        target_dir   = mon.Path(target_dir)
        target_files = sorted(list(target_dir.rglob("*")))
        target_files = [str(x) for x in target_files]

        self.noisy_filenames = [os.path.join(input_dir,  x) for x in input_files  if is_png_file(x)]
        self.clean_filenames = [os.path.join(target_dir, x) for x in target_files if is_png_file(x)]
        self.img_options     = img_options
        self.tar_size        = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        clean     = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy     = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        
        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        # Crop Input and Target
        ps = self.img_options["patch_size"]
        H  = clean.shape[1]
        W  = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H - ps == 0:
            r = 0
            c = 0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]
        clean       = getattr(augment, apply_trans)(clean)
        noisy       = getattr(augment, apply_trans)(noisy)

        return clean, noisy, clean_filename, noisy_filename


##################################################################################################

class DataLoaderVal(Dataset):

    def __init__(self, input_dir, target_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()
        self.target_transform = target_transform
        input_dir    = mon.Path(input_dir)
        input_files  = sorted(list(input_dir.rglob("*")))
        input_files  = [str(x) for x in input_files]

        target_dir   = mon.Path(target_dir)
        target_files = sorted(list(target_dir.rglob("*")))
        target_files = [str(x) for x in target_files]

        self.noisy_filenames = [os.path.join(input_dir,  x) for x in input_files  if is_png_file(x)]
        self.clean_filenames = [os.path.join(target_dir, x) for x in target_files if is_png_file(x)]
        self.tar_size        = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index      = index % self.tar_size
        clean          = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy          = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        clean          = clean.permute(2, 0, 1)
        noisy          = noisy.permute(2, 0, 1)
        return clean, noisy, clean_filename, noisy_filename


##################################################################################################

class DataLoaderTest(Dataset):

    def __init__(self, input_dir, img_options):
        super(DataLoaderTest, self).__init__()
        input_dir          = mon.Path(input_dir)
        input_files        = sorted(list(input_dir.rglob("*")))
        input_files        = [str(x) for x in input_files]
        self.inp_filenames = [os.path.join(input_dir, x) for x in input_files if is_image_file(x)]
        self.inp_size      = len(self.inp_filenames)
        self.img_options   = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp      = Image.open(path_inp)
        inp      = TF.to_tensor(inp)
        return inp, filename


def get_training_data(input_dir, img_options):
    assert os.path.exists(input_dir)
    return DataLoaderTrain(input_dir, img_options, None)


def get_validation_data(input_dir):
    assert os.path.exists(input_dir)
    return DataLoaderVal(input_dir, None)


def get_test_data(input_dir, img_options=None):
    assert os.path.exists(input_dir)
    return DataLoaderTest(input_dir, img_options)
