#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIFAR datasets and datamodules.
"""

from __future__ import annotations

from urllib.error import URLError

from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torchvision.datasets.mnist import read_image_file
from torchvision.datasets.mnist import read_label_file
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.utils import download_and_extract_archive

from one.constants import *
from one.core import *
from one.data import Classification
from one.data import ClassLabels
from one.data import ClassLabels_
from one.data import DataModule
from one.data import Image
from one.data import ImageClassificationDataset
from one.plot import imshow_cls
from one.vision.acquisition import to_tensor
from one.vision.transformation import Resize

mnist_classlabels = [
    { "name": "0", "id": 0 },
    { "name": "1", "id": 1 },
    { "name": "2", "id": 2 },
    { "name": "3", "id": 3 },
    { "name": "4", "id": 4 },
    { "name": "5", "id": 5 },
    { "name": "6", "id": 6 },
    { "name": "7", "id": 7 },
    { "name": "8", "id": 8 },
    { "name": "9", "id": 9 }
]

fashionmnist_classlabels = [
    { "name": "T-shirt/top", "id": 0 },
    { "name": "Trouser",     "id": 1 },
    { "name": "Pullover",    "id": 2 },
    { "name": "Dress",       "id": 3 },
    { "name": "Coat",        "id": 4 },
    { "name": "Sandal",      "id": 5 },
    { "name": "Shirt",       "id": 6 },
    { "name": "Sneaker",     "id": 7 },
    { "name": "Bag",         "id": 8 },
    { "name": "Ankle boot",  "id": 9 }
]


# MARK: - Module ---------------------------------------------------------------

@DATASETS.register(name="mnist")
class MNIST(ImageClassificationDataset):
    """
    MNIST <http://yann.lecun.com/exdb/mnist/> Dataset.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    mirrors       = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]
    resources     = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz",  "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz",  "ec29112dd5afa0611ce80d1b7f02629c")
    ]
    training_file = "training.pt"
    test_file     = "test.pt"
    classes       = [
        "0 - zero", "1 - one", "2 - two", "3 - three", "4 - four",
        "5 - five", "6 - six", "7 - seven", "8 - eight", "9 - nine"
    ]
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_data      : bool                = False,
        cache_images    : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = root / "classlabels.json",
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "test"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train` or `test`. Get: {self.split}."
            )
        
        if not self._check_exists():
            self.download()

        image_file = f"{'train' if self.split == 'train' else 't10k'}-images-idx3-ubyte"
        data       = read_image_file(self.root / "raw" / image_file)
        data       = torch.unsqueeze(data, -1)
        data       = data.repeat(1, 1, 1, 3)
        
        self.images: list[Image] = [
            Image(
                image          = to_tensor(img, keepdim=False, normalize=True),
                keep_in_memory = True
            )
            for img in data
        ]
        
    def list_labels(self):
        """
        List label files.
        """
        label_file = f"{'train' if self.split == 'train' else 't10k'}-labels-idx1-ubyte"
        data       = read_label_file(self.root / "raw" / label_file)
        self.labels: list[Classification] = [Classification(id=l) for l in data]
        
    def filter(self):
        pass
    
    def _check_legacy_exist(self):
        processed_folder = self.root / self.__class__.classname / "processed"
        if not processed_folder.exists():
            return False

        return all(
            check_integrity(processed_folder / file)
            for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary,
        # but simply read from the raw data directly.
        processed_folder = self.root / "processed"
        data_file        = self.training_file if self.train else self.test_file
        return torch.load(self.processed_folder / data_file)
    
    def _check_exists(self) -> bool:
        return all(
            check_integrity(self.root / "raw" / Path(url).stem)
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""
        if self._check_exists():
            return

        raw_folder = self.root / self.__class__.classname / "raw"
        create_dirs([raw_folder])

        # Download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    console.log("Downloading {}".format(url))
                    download_and_extract_archive(
                        url,
                        download_root = raw_folder,
                        filename      = filename,
                        md5           = md5
                    )
                except URLError as error:
                    console.log("Failed to download (trying next):\n{}".format(error))
                    continue
                finally:
                    console.log()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


@DATASETS.register(name="fashionmnist")
class FashionMNIST(MNIST):
    """
    Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist> Dataset.
    """
    
    mirrors   = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    ]
    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz",  "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz",  "bb300cfdad3c16e7a12a480ee83cd310")
    ]
    classes   = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
        "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    

@DATAMODULES.register(name="mnist")
class MNISTDataModule(DataModule):
    """
    MNIST DataModule.
    """
    
    def __init__(
        self,
        root: Path_ = DATA_DIR / "mnist" / "mnist",
        name: str   = "mnist",
        *args, **kwargs
    ):
        super().__init__(root=root, name=name, *args, **kwargs)
    
    @property
    def num_workers(self) -> int:
        """
        Returns number of workers used in the data loading pipeline.
        """
        # Set `num_workers` = 4 * the number of gpus to avoid bottleneck
        return 1
    
    def prepare_data(self, *args, **kwargs):
        """
        Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
    def setup(self, phase: ModelPhase_ | None = None):
        """
        There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelPhase_ | None):
                Stage to use: [None, ModelPhase.TRAINING, ModelPhase.TESTING].
                Set to None to setup all train, val, and test data.
                Defaults to None.
        """
        console.log(f"Setup [red]MNIST[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = MNIST(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            train_size   = int(0.8 * len(full_dataset))
            val_size     = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn  = getattr(full_dataset, "collate_fn",  None)
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
            self.test = MNIST(
                root             = self.root,
                split            = "test",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """
        Load ClassLabels.
        """
        self.classlabels = ClassLabels(mnist_classlabels)


@DATAMODULES.register(name="fashionmnist")
class FashionMNISTDataModule(DataModule):
    """
    FashionMNIST DataModule.
    """
    
    def __init__(
        self,
        root: Path_ = DATA_DIR / "mnist" / "fashionmnist",
        name: str   = "fashionmnist",
        *args, **kwargs
    ):
        super().__init__(root=root, name=name, *args, **kwargs)
    
    @property
    def num_workers(self) -> int:
        """
        Returns number of workers used in the data loading pipeline.
        """
        # Set `num_workers` = 4 * the number of gpus to avoid bottleneck
        return 1
    
    def prepare_data(self, *args, **kwargs):
        """
        Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
    def setup(self, phase: ModelPhase_ | None = None):
        """
        There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelPhase_ | None):
                Stage to use: [None, ModelPhase.TRAINING, ModelPhase.TESTING].
                Set to None to setup all train, val, and test data.
                Defaults to None.
        """
        console.log(f"Setup [red]Fashion-MNIST[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = FashionMNIST(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            train_size   = int(0.8 * len(full_dataset))
            val_size     = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn  = getattr(full_dataset, "collate_fn",  None)
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
            self.test = FashionMNIST(
                root             = self.root,
                split            = "test",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """
        Load ClassLabels.
        """
        self.classlabels = ClassLabels(mnist_classlabels)


# MARK: - Test -----------------------------------------------------------------

def test_mnist():
    cfg = {
        "root": DATA_DIR / "mnist" / "mnist",
           # Root directory of dataset.
        "name": "mnist",
            # Dataset's name.
        "shape": [3, 32, 32],
            # Image shape as [H, W, C], [H, W], or [S, S].
        "transform": [
            Resize(size=[3, 32, 32])
        ],
            # Functions/transforms that takes in an input sample and returns a
            # transformed version.
        "target_transform": None,
            # Functions/transforms that takes in a target and returns a
            # transformed version.
        "transforms": None,
            # Functions/transforms that takes in an input and a target and
            # returns the transformed versions of both.
        "cache_data": True,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": VISION_BACKEND,
            # Vision backend to process image. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # Number of samples in one forward & backward pass. Defaults to 1.
        "devices" : 0,
            # The devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the data at every training epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = MNISTDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    imshow_cls(
        winname     = "image",
        image       = input,
        target      = target,
        classlabels = dm.classlabels
    )
    plt.show(block=True)


def test_fashionmnist():
    cfg = {
        "root": DATA_DIR / "mnist" / "fashionmnist",
           # Root directory of dataset.
        "name": "fashionmnist",
            # Dataset's name.
        "shape": [3, 32, 32],
            # Image shape as [H, W, C], [H, W], or [S, S].
        "transform": [
            Resize(size=[3, 32, 32])
        ],
            # Functions/transforms that takes in an input sample and returns a
            # transformed version.
        "target_transform": None,
            # Functions/transforms that takes in a target and returns a
            # transformed version.
        "transforms": None,
            # Functions/transforms that takes in an input and a target and
            # returns the transformed versions of both.
        "cache_data": True,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": VISION_BACKEND,
            # Vision backend to process image. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # Number of samples in one forward & backward pass. Defaults to 1.
        "devices" : 0,
            # The devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the data at every training epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = FashionMNISTDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    imshow_cls(
        winname     = "image",
        image       = input,
        target      = target,
        classlabels = dm.classlabels
    )
    plt.show(block=True)


# MARK: - Main -----------------------------------------------------------------

if __name__ == "__main__":
    test_fashionmnist()
