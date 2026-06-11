from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from rich import print as rprint

from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.dataset.primitives import Bbox, Bitmask, Classification
from hafnia.dataset.primitives.polygon import Polygon

MNIST_VERSION = "1.0.0"
MIDWEST_VERSION = "1.0.0"
COCO_VERSION = "1.0.0"

# First ensure that you have the Hafnia CLI installed and configured.
# You can install it via pip:
#   pip install hafnia
# And configure it with your Hafnia account:
#   hafnia configure

# Load sample dataset
dataset = HafniaDataset.from_name("mnist", version=MNIST_VERSION)

# Use 'from_name' without version-argument to get available versions:
# dataset = HafniaDataset.from_name("mnist")
# >>> ValueError: Version must be specified. Available versions: ['1.0.0', '0.0.1']

# Dataset information is stored in 'dataset.info'
rprint(dataset.info)

# Annotations are stored in 'dataset.samples' as a Polars DataFrame
dataset.samples.head(2)

# Print dataset information
dataset.print_basic_stats()  # Print basic dataset statistics
dataset.print_sample_and_task_counts()
dataset.print_class_distribution()
dataset.print_stats()  # Print verbose dataset statistics

# Get dataset stats
dataset.calculate_class_counts()  # Get class counts for all tasks
dataset.calculate_task_class_counts(primitive=Classification)  # Get class counts for a specific task

# Create a dataset split for training
dataset_train = dataset.create_split_dataset("train")

# Checkout built-in transformations in 'operations/dataset_transformations' or 'HafniaDataset'
dataset_val = dataset.create_split_dataset(SplitName.VAL)  # Use 'SplitName' to avoid magic strings

small_dataset = dataset.select_samples(n_samples=10, seed=42)  # Selects 10 samples from the dataset
shuffled_dataset = dataset.shuffle(seed=42)  # Shuffle the dataset

# Create dataset splits by ratios
split_ratios = {SplitName.TRAIN: 0.8, SplitName.VAL: 0.1, SplitName.TEST: 0.1}
new_dataset_splits = dataset.splits_by_ratios(split_ratios)

# Get only samples with specific class names
dataset_ones = dataset.select_samples_by_class_name(name="1 - one", primitive=Classification)

# Get access to a few full and public dataset through Hafnia (no login required)
# Available datasets: "mnist", "caltech-101", "caltech-256", "cifar10", "cifar100"
public_dataset = HafniaDataset.from_name_public_dataset("mnist", n_samples=100)
public_dataset.print_stats()


# Rename class names with mapping
class_mapping_strict = {
    "0 - zero": "even",  # "0 - zero" will be renamed to "even". "even" appear first and get class index 0
    "1 - one": "odd",  # "1 - one" will be renamed to "odd". "odd" appear second and will get class index 1
    "2 - two": "even",
    "3 - three": "odd",
    "4 - four": "even",
    "5 - five": "odd",
    "6 - six": "even",
    "7 - seven": "odd",
    "8 - eight": "even",
    "9 - nine": "__REMOVE__",  # Remove all samples with class "9 - nine"
}
dataset_mapped = dataset.class_mapper(class_mapping=class_mapping_strict)
dataset_mapped.print_class_distribution()

# Support Chaining Operations (load, shuffle, select samples)
dataset = (
    HafniaDataset.from_name("midwest-vehicle-detection", version=MIDWEST_VERSION)
    .shuffle(seed=42)
    .select_samples(n_samples=10)
)


# Write dataset to disk
path_tmp = Path(".data/tmp")
path_dataset = path_tmp / "hafnia_dataset"
dataset.write(path_dataset)

# Load dataset from disk
dataset_again = HafniaDataset.from_path(path_dataset)

## Dataset importers and exporters ##
dataset_od = HafniaDataset.from_name("coco-2017", version=COCO_VERSION).select_samples(n_samples=5, seed=42)

# Export/import dataset to YOLO format
path_yolo_format = Path(".data/tmp/yolo_dataset")
dataset_od.to_yolo_format(path_output=path_yolo_format)  # Export to YOLO format
dataset_od_imported = HafniaDataset.from_yolo_format(path_yolo_format)  # Import dataset from YOLO format


# Export/import dataset to YOLO format
path_coco_format = Path(".data/tmp/coco_dataset")
dataset_od.to_coco_format(path_output=path_coco_format)  # Export to COCO format
dataset_od_imported = HafniaDataset.from_coco_format(path_coco_format)  # Import dataset from COCO format

## Custom dataset operations and statistics ##
# Want custom dataset transformations or statistics? Use the polars table (dataset.samples) directly
n_objects = dataset.samples["bboxes"].list.len().sum()
n_objects = dataset.samples[Bbox.column_name()].list.len().sum()  # Use Bbox.column_name() to avoid magic variables
n_classifications = dataset.samples[Classification.column_name()].list.len().sum()

class_counts = dataset.samples[Classification.column_name()].explode().struct.field("class_name").value_counts()
class_counts = dataset.samples[Bbox.column_name()].explode().struct.field("class_name").value_counts()
rprint(dict(class_counts.iter_rows()))

# Access the first sample in the training split - data is stored in a dictionary
sample_dict = dataset_train[0]

# Dataset can also be iterated to get sample data
for sample_dict in dataset_train:
    break

# Unpack dict into a Sample-object! Important for data validation, useability, IDE completion and mypy hints
sample: Sample = Sample(**sample_dict)

bboxes: List[Bbox] = sample.bboxes  # Use 'sample.bboxes' access bounding boxes as a list of Bbox objects
bitmasks: List[Bitmask] = sample.bitmasks  # Use 'sample.bitmasks' to access bitmasks as a list of Bitmask objects
polygons: List[Polygon] = sample.polygons  # Use 'sample.polygons' to access polygons as a list of Polygon objects
classifications: List[Classification] = sample.classifications  # As a list of Classification objects


# Read image using the sample object
image: np.ndarray = sample.read_image()

# Visualize sample and annotations
image_with_annotations = sample.draw_annotations()

# Save the image with annotations to a temporary directory
path_tmp.mkdir(parents=True, exist_ok=True)
Image.fromarray(image_with_annotations).save(path_tmp / "sample_with_annotations.png")

## Storing predictions: A hafnia dataset can also be used for storing predictions per sample
# set 'ground_truth=False' and add 'confidence'.
bboxes_predictions = [
    Bbox(top_left_x=0.1, top_left_y=0.2, width=0.3, height=0.4, class_name="car", ground_truth=False, confidence=0.9)
]

classifications_predictions = [Classification(class_name="vehicle", class_idx=0, ground_truth=False, confidence=0.95)]
