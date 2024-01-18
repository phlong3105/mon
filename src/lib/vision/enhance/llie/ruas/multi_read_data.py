import random
import glob

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from PIL.ImageFile import ImageFile
import mon
import config

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MemoryFriendlyLoader(torch.utils.data.Dataset):
    
    def __init__(self, img_dir, task):
        self.low_img_dir          = mon.Path(img_dir)
        self.task                 = task
        self.train_low_data_names = list(self.low_img_dir.rglob("*"))
        self.train_low_data_names = [name for name in self.train_low_data_names if name.is_image_file()]
        self.train_low_data_names.sort()
        self.count                = len(self.train_low_data_names)
        
        transform_list  = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        # im       = cv2.imread(str(file))  # BGR
        # im       = np.ascontiguousarray(im[::-1])
        im       = Image.open(file).convert("RGB")
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm
    
    def __getitem__(self, index):
        low = self.load_images_transform(self.train_low_data_names[index])

        h   = low.shape[0]
        w   = low.shape[1]

        h_offset = random.randint(0, max(0, h - config.h - 1))
        w_offset = random.randint(0, max(0, w - config.w - 1))

        if self.task != "test":
            low = low[h_offset:h_offset + config.h, w_offset:w_offset + config.w]

        low = np.asarray(low, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))

        if self.task == "test":
            # img_name = self.train_low_data_names[index].split('\\')[-1]
            img_name = self.train_low_data_names[index].name
            return torch.from_numpy(low), img_name

        return torch.from_numpy(low)

    def __len__(self):
        return self.count
