from dataclasses import dataclass
from typing import Dict


@dataclass
class DatasetSpec:
    name: str
    splits: Dict[str, int]
    version: str


DATASET_SPEC_MIDWEST = DatasetSpec(
    name="midwest-vehicle-detection",
    splits={"train": 172, "validation": 21, "test": 21},
    version="1.0.0",  # Remember to update "MIDWEST_VERSION" used in example-scripts accordingly
)

DATASET_SPEC_MIDWEST_TRAFFIC = DatasetSpec(
    name="midwest-detection-traffic",
    splits={"train": 172, "validation": 21, "test": 21},
    version="1.0.0",
)

DATASET_SPEC_TINY_DATASET = DatasetSpec(
    name="tiny-dataset",
    splits={"train": 6, "validation": 4, "test": 6},
    version="1.0.0",
)

DATASET_SPEC_MNIST = DatasetSpec(
    name="mnist",
    splits={"train": 176, "test": 18, "validation": 6},
    version="1.0.0",  # Remember to update "MNIST_VERSION" used in example-scripts accordingly
)

DATASET_SPEC_CALTECH_101 = DatasetSpec(
    name="caltech-101",
    splits={"train": 166, "validation": 21, "test": 13},
    version="1.0.0",
)

DATASET_SPEC_CALTECH_256 = DatasetSpec(
    name="caltech-256",
    splits={"train": 163, "validation": 17, "test": 20},
    version="1.0.0",
)

DATASET_SPEC_CIFAR10 = DatasetSpec(
    name="cifar10",
    splits={"train": 171, "validation": 4, "test": 25},
    version="1.0.0",
)
DATASET_SPEC_CIFAR100 = DatasetSpec(
    name="cifar100",
    splits={"train": 171, "validation": 4, "test": 25},
    version="1.0.0",
)
DATASET_SPEC_COCO_2017 = DatasetSpec(
    name="coco-2017",
    splits={"train": 189, "validation": 2, "test": 9},
    version="1.0.0",  # Remember to update "COCO_VERSION" used in example-scripts accordingly
)

DATASET_SPEC_COCO_2017_TINY = DatasetSpec(
    name="coco-2017-tiny",
    splits={"train": 3, "validation": 2, "test": 3},
    version="1.0.0",
)

SUPPORTED_DATASETS = [
    DATASET_SPEC_MIDWEST,
    DATASET_SPEC_MIDWEST_TRAFFIC,
    DATASET_SPEC_TINY_DATASET,
    DATASET_SPEC_MNIST,
    DATASET_SPEC_CALTECH_101,
    DATASET_SPEC_CALTECH_256,
    DATASET_SPEC_CIFAR10,
    DATASET_SPEC_CIFAR100,
    DATASET_SPEC_COCO_2017,
]
