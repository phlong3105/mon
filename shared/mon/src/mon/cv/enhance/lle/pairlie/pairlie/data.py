from torchvision.transforms import Compose, RandomCrop, ToTensor

from dataset import DatasetFromFolder, DatasetFromFolderEval


def transform1():
    return Compose([
        RandomCrop((128, 128)),
        ToTensor(),
    ])


def transform2():
    return Compose([
        ToTensor(),
    ])


def get_training_set(data_dir):
    return DatasetFromFolder(data_dir, transform=transform1())


def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())
