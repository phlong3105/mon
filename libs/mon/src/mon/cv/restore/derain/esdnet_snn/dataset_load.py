#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
from numpy.random import RandomState
from skimage import io
from torch.utils.data import Dataset


# data_dir = 'D:/1Single_Image_Derain/data/Rain200H'

class Dataload(Dataset):
    
    def __init__(self, data_dir, patch_size):
        super().__init__()
        self.rand_state  = RandomState(66)
        self.root_dir    = data_dir
        # self.root_dir_rain   = os.path.join(self.root_dir, "image")
        # self.root_dir_label  = os.path.join(self.root_dir, "ref")
        # self.mat_files_rain  = sorted(os.listdir(self.root_dir_rain))
        # self.mat_files_label = sorted(os.listdir(self.root_dir_label))
        self.files_rain  = sorted(list(self.root_dir.rglob("*image/*")))
        self.files_label = sorted(list(self.root_dir.rglob("*ref/*")))
        self.files_rain  = [x for x in self.files_rain  if x.is_image_file()]
        self.files_label = [x for x in self.files_label if x.is_image_file()]
        self.patch_size  = patch_size
        self.file_num    = len(self.files_rain)

    def __len__(self):
        return self.file_num

    def __getitem__(self, index: int):
        # file_name_rain    = self.mat_files_rain[index % self.file_num]
        # file_name_label   = self.mat_files_label[index % self.file_num]
        # img_file_rain   = os.path.join(self.root_dir_rain,  file_name_rain)
        # img_file_label  = os.path.join(self.root_dir_label, file_name_label)
        img_file_rain  =  self.files_rain[index % self.file_num]
        img_file_label = self.files_label[index % self.file_num]
        img_rain  =  io.imread(img_file_rain).astype(np.float32) / 255
        img_label = io.imread(img_file_label).astype(np.float32) / 255
        O, B = self.crop(img_rain, img_label)
        O    = np.transpose(O, (2, 0, 1))
        B    = np.transpose(B, (2, 0, 1))
        return torch.Tensor(O), torch.Tensor(B)

    def crop(self, img_rain, img_label):
        h, w, c  = img_rain.shape
        h        = h - 1
        w        = w - 1
        p_h, p_w = self.patch_size, self.patch_size
        r        = self.rand_state.randint(0, h - p_h)
        c        = self.rand_state.randint(0, w - p_w)
        O        =  img_rain[r: r + p_h, c: c + p_w]
        B        = img_label[r: r + p_h, c: c + p_w]
        return O, B


class TrainValDataset(Dataset):
    
    def __init__(self, data_dir, name, patch_size):
        super().__init__()
        self.rand_state      = RandomState(66)
        self.name            = name
        self.root_dir        = os.path.join(data_dir, self.name)
        self.root_dir_rain   = os.path.join(self.root_dir, "image")
        self.root_dir_label  = os.path.join(self.root_dir, "ref")

        self.mat_files_rain  = sorted(os.listdir(self.root_dir_rain))
        self.mat_files_label = sorted(os.listdir(self.root_dir_label))
        self.patch_size      = patch_size
        self.file_num        = len(self.mat_files_label)

    def __len__(self):
        if self.name == "train":
            return self.file_num * 1
        else:
            return self.file_num

    def __getitem__(self, idx):
        file_name_rain  = self.mat_files_rain[idx % self.file_num]
        file_name_label = self.mat_files_label[idx % self.file_num]

        img_file_rain   = os.path.join(self.root_dir_rain, file_name_rain)
        img_file_label  = os.path.join(self.root_dir_label, file_name_label)

        img_rain  = io.imread(img_file_rain).astype(np.float32) / 255
        img_label = io.imread(img_file_label).astype(np.float32) / 255

        O, B = self.crop(img_rain, img_label)
        O    = np.transpose(O, (2, 0, 1))
        B    = np.transpose(B, (2, 0, 1))
        return torch.Tensor(O), torch.Tensor(B)

    def crop(self, img_rain, img_label):
        h, w, c  = img_rain.shape
        h        = h - 1
        w        = w - 1
        p_h, p_w = self.patch_size, self.patch_size
        r        = self.rand_state.randint(0, h - p_h)
        c        = self.rand_state.randint(0, w - p_w)
        O        =  img_rain[r: r + p_h, c: c + p_w]
        B        = img_label[r: r + p_h, c: c + p_w]
        return O, B
