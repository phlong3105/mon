# Building a Custom `HafniaDataset`

When your data is not already on the Hafnia platform — for example a private
dataset on disk, or annotations in a non-standard format — you can construct
a `HafniaDataset` directly from images and annotation primitives. The result
is a regular `HafniaDataset` that supports all the usual operations,
exporters and (optionally) upload to the platform.

Before writing custom code, check whether a built-in importer already covers
your format:

```python
HafniaDataset.from_yolo_format(path)
HafniaDataset.from_coco_format(path)
```

See [src/hafnia/dataset/format_conversions/](../src/hafnia/dataset/format_conversions/)
for the full list. The custom path described below is the right tool only
when no importer fits.

## The three building blocks

To build a dataset by hand you need three things:

1. **Annotation primitives per image** — instances of `Bbox`, `Bitmask`,
   `Polygon` or `Classification`. Bounding-box coordinates are normalized
   to `[0, 1]` and use `top_left_x`, `top_left_y`, `width`, `height`.
2. **A `Sample` per image** — wraps the file path, image dimensions, split
   name and the list of primitives.
3. **A `DatasetInfo`** — declares the dataset name, version and the
   `TaskInfo`s (primitive + class names).

`HafniaDataset.from_samples_list(samples_list, info=...)` then turns these
into a dataset.

## End-to-end example

```python
from pathlib import Path
from PIL import Image
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, Sample, TaskInfo
from hafnia.dataset.primitives.bbox import Bbox

class_names = ["person", "car", "bicycle"]
samples = []
for path_image in Path("data/images").glob("*.jpg"):
    image = Image.open(path_image)
    width, height = image.size

    # Read your annotations however you like; build hafnia Bbox primitives.
    bboxes = [
        Bbox(
            top_left_x=cx - bw / 2,
            top_left_y=cy - bh / 2,
            width=bw,
            height=bh,
            class_idx=class_idx,
            class_name=class_names[class_idx],
        )
        for class_idx, cx, cy, bw, bh in read_yolo_txt(path_image.with_suffix(".txt"))
    ]
    samples.append(Sample(
        file_path=str(path_image),
        height=height,
        width=width,
        split="train",
        bboxes=bboxes,
    ))

info = DatasetInfo(
    dataset_name="my-custom-dataset",
    version="0.0.1",
    tasks=[TaskInfo(primitive=Bbox, class_names=class_names)],
)
dataset = HafniaDataset.from_samples_list(samples_list=samples, info=info)
```

Once built, the dataset behaves like any other:

```python
dataset.print_stats()
sample = Sample(**dataset[0])
Image.fromarray(sample.draw_annotations()).save("preview.png")
dataset.write(".data/datasets/my-custom-dataset")
```

## Tips and gotchas

- **Image dimensions are required.** Read them once per image and pass them
  to `Sample` — primitive coordinates are normalized against them.
- **Each `Sample.split` value matters.** Use `"train"`, `"val"`, `"test"` (or
  `SplitName.*`) consistently so `create_split_dataset(...)` works later.
- **`class_idx` should agree with `class_name`.** The `TaskInfo.class_names`
  order defines the index space; keep them in sync to avoid surprises in
  metrics and exports.
- **Predictions vs ground truth.** Set `ground_truth=False` and add
  `confidence=...` on a primitive to store predictions instead of labels.

## Uploading to the platform (optional)

If your custom dataset should become a named Hafnia dataset, use
`HafniaDataset.upload_to_platform`:

```python
dataset.upload_to_platform(
    interactive=False,
    allow_version_overwrite=True,
    gallery_samples=...,   # optional preview samples
)
```

## See also

- [examples/example_custom_dataset.py](../examples/example_custom_dataset.py) — runnable walkthrough that builds a dataset from a YOLO-formatted directory.
- [docs/dataset.md](dataset.md) — the resulting `HafniaDataset` format and operations.
- [docs/dataset_recipe.md](dataset_recipe.md) — composing your custom dataset with other sources via recipes.
