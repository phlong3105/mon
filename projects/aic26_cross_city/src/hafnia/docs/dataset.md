# The Hafnia Dataset Format

`HafniaDataset` is the in-memory representation of a computer-vision dataset
used throughout the Hafnia SDK. It is the same object whether you are working
with a small **sample dataset** locally or with a **full dataset** under
Training-aaS — a training script written against `HafniaDataset` does not
change between the two environments.

## What a dataset contains

A `HafniaDataset` is a pair of two fields:

- **`dataset.info`** — a `DatasetInfo` describing the dataset and its tasks.
  Each `TaskInfo` declares a primitive (`Bbox`, `Bitmask`, `Polygon`,
  `Classification`, ...) and the list of `ClassInfo` entries that define the
  label space for that task.
- **`dataset.samples`** — a [Polars](https://pola.rs/) `DataFrame` where each
  row is one image (or video frame). Primitive columns (`classifications`,
  `bboxes`, `bitmasks`, `polygons`) are stored as `List[Struct]` and mirror
  the fields of the corresponding `Sample` Pydantic model.

A row in `dataset.samples` unpacks into a `Sample` object that carries the
image path, dimensions, split, tags, optional camera/position/orientation
metadata, and the per-sample annotation primitives.

## Loading and writing

```python
from hafnia.dataset.hafnia_dataset import HafniaDataset

# Load by name — sample dataset locally, full dataset under Training-aaS
dataset = HafniaDataset.from_name("midwest-vehicle-detection", version="1.0.0")

# Or load from disk
dataset = HafniaDataset.from_path(".data/datasets/midwest-vehicle-detection")

# Persist to disk
dataset.write(".data/tmp/hafnia_dataset")
```

`from_name` downloads the dataset to `.data/datasets/` on first call and
caches it for subsequent calls.

## Iterating samples

```python
from hafnia.dataset.hafnia_dataset_types import Sample

for sample_dict in dataset:
    sample = Sample(**sample_dict)   # validate + IDE/mypy support
    image = sample.read_image()      # np.ndarray
    for bbox in sample.bboxes or []:
        print(bbox.class_name, bbox.top_left_x, bbox.top_left_y)
    break
```

`Sample` also exposes `draw_annotations()` for quick visualization.

## Common operations

`HafniaDataset` supports chainable operations that return a new dataset:

```python
dataset_train = dataset.create_split_dataset("train")
small        = dataset.select_samples(n_samples=10, seed=42)
shuffled     = dataset.shuffle(seed=42)
splits       = dataset.splits_by_ratios({"train": 0.8, "val": 0.1, "test": 0.1})
only_ones    = dataset.select_samples_by_class_name(name="1 - one", primitive="Classification")
remapped     = dataset.class_mapper(class_mapping={"car": "Vehicle", "person": "Person"})
merged       = HafniaDataset.from_merge(dataset0=a, dataset1=b)
```

For statistics, see `dataset.print_stats()`, `dataset.print_class_distribution()`,
`dataset.calculate_class_counts()` and friends. For anything custom, work
directly on `dataset.samples` with native Polars expressions.

## Annotation primitives

All annotations inherit from a common `Primitive` base and live in
[src/hafnia/dataset/primitives/](../src/hafnia/dataset/primitives/):

| Primitive        | Column            | Use                                    |
| ---------------- | ----------------- | -------------------------------------- |
| `Classification` | `classifications` | Image- or sample-level labels          |
| `Bbox`           | `bboxes`          | 2D bounding boxes (normalized coords)  |
| `Polygon`        | `polygons`        | Polygonal segmentations                |
| `Bitmask`        | `bitmasks`        | Pixel masks                            |

Each primitive carries `class_name`, `class_idx`, and a `ground_truth` flag.
Set `ground_truth=False` and add a `confidence` value to store predictions
alongside ground truth in the same dataset.

## Importers and exporters

Built-in converters under
[src/hafnia/dataset/format_conversions/](../src/hafnia/dataset/format_conversions/)
allow round-tripping with common formats:

```python
dataset.to_yolo_format(path_output=".data/tmp/yolo_dataset")
HafniaDataset.from_yolo_format(".data/tmp/yolo_dataset")

dataset.to_coco_format(path_output=".data/tmp/coco_dataset")
HafniaDataset.from_coco_format(".data/tmp/coco_dataset")
```

To bring data that is not yet on the platform into the Hafnia format, build
samples directly from `Sample`, `Bbox`, `DatasetInfo`, etc. See
[examples/example_custom_dataset.py](../examples/example_custom_dataset.py).

## See also

- [examples/example_hafnia_dataset.py](../examples/example_hafnia_dataset.py) — runnable walkthrough.
- [docs/dataset_recipe.md](dataset_recipe.md) — composing reproducible datasets from recipes.
