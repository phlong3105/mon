from torch.utils.data import Dataset
from pathlib import Path
import os

from misc import *

DATA_PATH = Path("/home/soom/data/LOL/our485")

class LOLDataset(Dataset):
    def __init__(self):
        self.size = (128, 128)
        self.filenames = sorted(os.listdir(DATA_PATH / 'low'))
        self.lq_imgs = [image_tensor(DATA_PATH / 'low' / filename, size=self.size) for filename in self.filenames]
        self.gt_imgs = [image_tensor(DATA_PATH / 'high' / filename, size=self.size) for filename in self.filenames]

    def __getitem__(self, index):
        lq = self.lq_imgs[index]
        gt = self.gt_imgs[index]
        return lq, gt

    def __len__(self):
        return len(self.filenames)