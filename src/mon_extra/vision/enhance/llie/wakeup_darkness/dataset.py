import numpy as np
import torch
import torch.utils.data
import random
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import os


class ImageLowSemDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, sem_dir, depth_dir):
        self.low_img_dir = img_dir
        self.sem_dir = sem_dir
        self.depth_dir = depth_dir

        self.low_img_names = []
        self.sem_names = []
        self.depth_names = []
        for name in os.listdir(self.low_img_dir):
            if name.endswith('.jpg') or name.endswith('.png'):
                self.low_img_names.append(os.path.join(self.low_img_dir, name))
                self.sem_names.append(os.path.join(self.sem_dir, f'{os.path.splitext(name)[0]}_semantic.png'))
                self.depth_names.append(os.path.join(self.depth_dir, f'{os.path.splitext(name)[0]}_depth.png'))

        self.count = len(self.low_img_names)

        transform_list = []
        transform_list += [transforms.ToTensor()]  # ToTensor()包含将数据规范到(0,1)
        # transform_list += [transforms.Normalize((0, 0, 0), (255, 255, 255))]
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        img_norm = self.transform(im)
        return img_norm

    def __getitem__(self, index):

        low_img = self.load_images_transform(self.low_img_names[index])
        sem = self.load_images_transform(self.sem_names[index])
        depth = self.load_images_transform(self.depth_names[index])

        img_name = os.path.basename(self.low_img_names[index])
        sem_name = os.path.basename(self.sem_names[index])
        depth_name = os.path.basename(self.depth_names[index])

        return low_img, sem, depth, img_name, sem_name, depth_name

    def __len__(self):
        return self.count
    
    

class ImageLowSemDataset_Val(torch.utils.data.Dataset):
    def __init__(self, img_dir, sem_dir, depth_dir):
        self.low_img_dir = img_dir
        self.sem_dir = sem_dir
        self.depth_dir = depth_dir

        self.low_img_names = []
        self.sem_names = []
        self.depth_names = []
        for name in os.listdir(self.low_img_dir):
            if name.endswith('.jpg') or name.endswith('.png'):
                self.low_img_names.append(os.path.join(self.low_img_dir, name))
                self.sem_names.append(os.path.join(self.sem_dir, f'{os.path.splitext(name)[0]}_semantic.png'))
                self.depth_names.append(os.path.join(self.depth_dir, f'{os.path.splitext(name)[0]}_depth.png'))

        self.count = len(self.low_img_names)

        transform_list = []
        transform_list += [transforms.ToTensor()]  # ToTensor()包含将数据规范到(0,1)
        # transform_list += [transforms.Normalize((0, 0, 0), (255, 255, 255))]
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        img_norm = self.transform(im)
        return img_norm

    def __getitem__(self, index):

        low_img = self.load_images_transform(self.low_img_names[index])
        sem = self.load_images_transform(self.sem_names[index])
        depth = self.load_images_transform(self.depth_names[index])

        img_name = os.path.basename(self.low_img_names[index])
        sem_name = os.path.basename(self.sem_names[index])
        depth_name = os.path.basename(self.depth_names[index])

        return low_img, sem, depth, img_name, sem_name, depth_name

    def __len__(self):
        return self.count
