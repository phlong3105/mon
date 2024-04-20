from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor

from data.eval_sets import *
from data.lol_dataset import *
from data.sice_blur_sid import *


def transform1(size: int = 256) -> Compose:
    return Compose([
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])


def transform2() -> Compose:
    return Compose([ToTensor()])


def get_lol_v1_training_set(data_dir: str, size: int):
    return LOLDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_training_set(data_dir: str, size: int):
    return LOLv2DatasetFromFolder(data_dir, transform=transform1(size))


def get_training_set_blur(data_dir: str, size: int):
    return LOLBlurDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_synthetic_training_set(data_dir: str, size: int):
    return LOLv2SyntheticDatasetFromFolder(data_dir, transform=transform1(size))


def get_sid_training_set(data_dir: str, size: int):
    return SIDDatasetFromFolder(data_dir, transform=transform1(size))


def get_sice_training_set(data_dir: str, size: int):
    return SICEDatasetFromFolder(data_dir, transform=transform1(size))


def get_sice_eval_set(data_dir: str):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())


def get_eval_set(data_dir: str):
    return DatasetFromFolderEval(data_dir, transform=transform2())
