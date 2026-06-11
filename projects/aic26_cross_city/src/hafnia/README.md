# Hafnia

The `hafnia` python sdk and cli is a collection of tools to create and run model trainer packages on
the [Hafnia Platform](https://hafnia.milestonesys.com/).

The package includes the following interfaces:

- `cli`: A Command Line Interface (CLI) to 1) configure/connect to Hafnia's [Training-aaS](https://hafnia.readme.io/docs/training-as-a-service),
  2) manage datasets, dataset recipes and trainer packages, and 3) build and launch trainer packages locally.
- `hafnia`: A python package including `HafniaDataset` to manage datasets, `DatasetRecipe` to compose
  reproducible dataset transformations and `HafniaLogger` for experiment tracking.

## The Concept: Training as a Service (Training-aaS)

`Training-aaS` is the concept of training models on the Hafnia platform on large
and _hidden_ datasets. Hidden datasets refers to datasets that can be used for
training, but are not available for download or direct access.

This is a key for the Hafnia platform, as a hidden dataset ensures data
privacy, and allow models to be trained compliantly and ethically by third parties (you).

The Training-aaS concept involves packaging your custom training
project as a _trainer package_ and using the package to train models on the hidden datasets.

To support local development of a trainer package, we have introduced a **sample dataset**
for each dataset available in the Hafnia [data library](https://hafnia.milestonesys.com/training-aas/datasets). The sample dataset is a small
and an anonymized subset of the full dataset and available for download.

With the sample dataset, you can seamlessly switch between local development and Training-aaS.
Locally, you can create, validate and debug your trainer package. The trainer package is then
launched with Training-aaS, where the package runs on the full dataset and can be scaled to run on
multiple GPUs and instances if needed.

## Quick Start: No-Code Model Training

To demonstrate the concept of Training-aaS, we will first show how to launch model training using the Hafnia Training-aaS platform - without writing any code - using a pre-built public trainer package.

### Steps:

1. **Sign in**  
   To sign in to the [Hafnia Platform](https://hafnia.milestonesys.com/).
1. **Access the Dashboard**  
   Go do the dashboard, select [Training-aaS](https://hafnia.milestonesys.com/dashboard/training-aas/experiments) and click "Create Experiment"

1. **Select Dataset**  
   Choose your target dataset (e.g., `coco-2017` or `midwest-vehicle-detection`)

1. **Select Trainer Package**  
   Use the public trainer package provided by Hafnia. Select the "Public Trainers" tab and choose the "Object Detection Trainer" package. *You may also upload your own trainer package, but we will describe that later.*

1. **Configure Training**  
   - **Training command:** `python scripts/train.py`
   - **Configuration:** Select "Lite", "Pro" or "Scale" based on your needs

1. **Launch & Monitor**  
   Click "Create Experiment" and monitor progress in the dashboard

That's it! You have successfully launched an object detection model training experiment using the Hafnia Training-aaS platform.

For default training parameters, the trainer package converges in approximately 4 hours on the `midwest-vehicle-detection` dataset using the "Lite" configuration.

## Installation and Configuration

To use the CLI for managing datasets and trainer packages — and to load datasets locally with the
Python SDK — install `hafnia` and configure it with your API key.

1. Install `hafnia` with your favorite python package manager:

   ```bash
   # With uv package manager
   uv add hafnia
   ```

2. Sign in to the [Hafnia Platform](https://hafnia.milestonesys.com/).
3. Create an API KEY for Training aaS. For more instructions, follow this
   [guide](https://hafnia.readme.io/docs/create-an-api-key).
   Copy the key and save it for later use.
4. From terminal, configure your machine to access Hafnia:

   ```
   # Start configuration with
   hafnia configure

   # You are then prompted:
   Alias:   # Press [Enter] to skip — personal label only, not tied to your Hafnia account
   Hafnia API Key:  # Pass your HAFNIA API key
   Hafnia Platform URL [default https://api.hafnia.milestonesys.com]:  # Press [Enter] to use the default
   ```

5. Download `mnist` from terminal to verify that your configuration is working.

   ```bash
   hafnia dataset download mnist --force
   ```

### CLI command surface

Once configured, the `hafnia` CLI exposes the following command groups. Each group has a `--help`
flag that shows the available subcommands and options:

```bash
hafnia configure                       Interactive first-time setup (profile name, API key, URL)
hafnia clear                           Remove all stored configuration

hafnia profile     ls | active | use | rm | create               # Manage local profiles
hafnia dataset     ls | download | delete                        # Manage datasets on the platform
hafnia recipe      ls | create | rm                              # Manage dataset recipes on the platform
hafnia trainer     ls | create | update | create-zip | view-zip  # Manage trainer packages
hafnia experiment  ls | create | environments                    # Launch and inspect experiments
hafnia runc        build | build-local | launch-local            # Build and run trainer packages locally
```

Run `hafnia <group> --help` (and `hafnia <group> <subcommand> --help`) to see the full set of
options for any command.

## Detailed Documentation

For more information, go to our [documentation page](https://hafnia.readme.io/docs/welcome-to-hafnia)
or use the topic guides below. The rest of this README links into them from each relevant section.

- [CLI](docs/cli.md) — Detailed guide for the Hafnia command-line interface.
- [Hafnia Dataset Format](docs/dataset.md) — The `HafniaDataset` in-memory format, annotation primitives and operations.
- [Dataset Recipes](docs/dataset_recipe.md) — Composing reproducible datasets with `DatasetRecipe`.
- [Custom Datasets](docs/custom_dataset.md) — Building a `HafniaDataset` from your own images and annotations.
- [Benchmarking](docs/benchmark.md) — Running models against a dataset and computing metrics.
- [Release lifecycle](docs/release.md) — Details about the package release lifecycle.

## Trainer Packages: Bring Your Own Training Code

When the public trainer packages are not enough — for example because you want a different model
architecture, a custom training loop, or your own evaluation metrics — you can package your own
training script as a _trainer package_ and launch it on Training-aaS just like the public ones.

We provide two reference trainer packages that serve as templates for creating
and structuring your own trainers:

- [trainer-classification](https://github.com/milestone-hafnia/trainer-classification) — image classification.
- [trainer-object-detection](https://github.com/milestone-hafnia/trainer-object-detection) — object detection.

Each repository contains additional information on how to structure your trainer package, use the
`HafniaLogger`, load a dataset and launch the trainer on the Hafnia platform.

### Managing trainer packages from the CLI

Trainer packages can be created and updated on the platform directly from the CLI:

```bash
hafnia trainer ls                                                # list trainer packages on the platform
hafnia trainer create   ../trainer-classification                # upload a new trainer package
hafnia trainer update   <trainer-id>  ../trainer-classification  # push a new version
hafnia trainer view-zip trainer.zip                              # inspect the contents of a trainer.zip
```

To get a better understanding we advice to visit the trainer package repositories above

### Build and run a `trainer.zip` locally

To test trainer-package compatibility with Hafnia cloud before uploading, build and run the job locally:

```bash
# Create 'trainer.zip' from the root folder of your trainer project (e.g. '../trainer-classification')
hafnia trainer create-zip ../trainer-classification

# Build the docker image locally from a 'trainer.zip' file
hafnia runc build-local trainer.zip

# Execute the docker image locally with a desired dataset
hafnia runc launch-local --dataset mnist  "python scripts/train.py"
```

## The `hafnia` Python Package

The `hafnia` Python package provides everything needed to build trainer packages: dataset loading
and manipulation, custom-dataset construction, recipe-based dataset composition, torch integration,
experiment tracking and inference benchmarking. Each feature has a runnable script under
[examples/](examples/).

### Loading datasets with `HafniaDataset`

`HafniaDataset` is the main entry point for loading and manipulating datasets. The same
`HafniaDataset.from_name` call returns the **sample dataset** locally and the **full dataset** when
running under Training-aaS — so a training script does not need to change between the two
environments.

```python
from hafnia.dataset.hafnia_dataset import HafniaDataset, Sample

# Load by name (downloads to .data/datasets/ on first call)
dataset = HafniaDataset.from_name("midwest-vehicle-detection", version="1.0.0")

# Or load from a local path
# dataset = HafniaDataset.from_path(path_dataset)

dataset.print_stats()
dataset_train = dataset.create_split_dataset("train")

# Iterate samples
for sample_dict in dataset:
    sample = Sample(**sample_dict)
    image = sample.read_image()
    print(sample.sample_index, sample.bboxes)
    break
```

`dataset.info` carries the dataset metadata (`DatasetInfo` with `TaskInfo` per task) and
`dataset.samples` is a Polars DataFrame whose primitive columns (`classifications`, `bboxes`,
`bitmasks`, `polygons`) mirror the corresponding `Sample` fields.

Available datasets (and their sample variants) are listed in the
[data library](https://hafnia.milestonesys.com/training-aas/datasets) including metadata and a
description for each dataset.

For a deeper walkthrough of `HafniaDataset`, see the
[Hafnia Dataset Format](docs/dataset.md) guide and the runnable
[examples/example_hafnia_dataset.py](examples/example_hafnia_dataset.py).

### Composing datasets with `DatasetRecipe`

A `DatasetRecipe` is a serializable specification of a dataset and the operations applied to it
(shuffle, select, merge, split, filter, remove classes, ...). Recipes are not executed when defined
— call `.build()` to materialize a `HafniaDataset`. This makes recipes ideal for sharing a dataset
configuration across local development and Training-aaS, and for combining multiple sources into a
single training dataset.

```python
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe

recipe = DatasetRecipe.from_name(name="mnist", version="1.0.0").shuffle().select_samples(n_samples=20)
dataset = recipe.build()

# Use thus function to upload the recipe and make it available for Training-aaS experiments
recipe.as_platform_recipe(recipe_name="example-mnist-recipe", overwrite=True)
```

Recipes can also be managed on the platform via the CLI: `hafnia recipe ls | create | rm`.

For a deeper walkthrough — including the recipe lifecycle, supported operations, merging and
platform upload — see the [Dataset Recipes](docs/dataset_recipe.md) guide and the runnable
[examples/example_dataset_recipe.py](examples/example_dataset_recipe.py).

### Bringing your own data: custom `HafniaDataset`

If you have data that is not yet on the Hafnia platform, you can construct a `HafniaDataset`
directly from images and annotations using `Sample`, the annotation primitives (`Bbox`, `Bitmask`,
`Polygon`, `Classification`) and `DatasetInfo`. Built-in importers such as
`HafniaDataset.from_yolo_format` and `HafniaDataset.from_coco_format` are also available for
common formats.

For a walkthrough of the building blocks and gotchas, see the
[Custom Datasets](docs/custom_dataset.md) guide and the runnable
[examples/example_custom_dataset.py](examples/example_custom_dataset.py), which builds a
`HafniaDataset` from a YOLO-formatted directory end-to-end.

### Torch dataloader

For `torch`-based training scripts, a dataset is typically used together with a dataloader that
performs augmentation and batching.
[examples/example_torchvision_dataloader.py](examples/example_torchvision_dataloader.py) shows how
to load a dataset, apply augmentations using `torchvision.transforms.v2`, visualize samples with
`torch_helpers.draw_image_and_targets` and combine `TorchvisionDataset` with `TorchVisionCollateFn`
in a `torch.utils.data.DataLoader`:

```python
# Load HafniaDataset (sample dataset locally, full dataset under Training-aaS)
dataset = HafniaDataset.from_name("midwest-vehicle-detection", version="1.0.0")
dataset_train = dataset.create_split_dataset("train")
dataset_test = dataset.create_split_dataset("test")

train_transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = torch_helpers.TorchvisionDataset(dataset_train, transforms=train_transforms)
collate_fn = torch_helpers.TorchVisionCollateFn()
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
```

### Experiment tracking with `HafniaLogger`

The `HafniaLogger` is an important part of the trainer and enables you to track, log and
reproduce your experiments.

When integrated into your training script, the `HafniaLogger` is responsible for collecting:

- **Trained Model**: The model trained during the experiment
- **Model Checkpoints**: Intermediate model states saved during training
- **Experiment Configurations**: Hyperparameters and other settings used in your experiment
- **Training/Evaluation Metrics**: Performance data such as loss values, accuracy, and custom metrics

```python
from hafnia.experiment import HafniaLogger

logger = HafniaLogger(project_name="my_classification_project")
logger.log_configuration({"batch_size": 128, "learning_rate": 0.001})

ckpt_dir = logger.path_model_checkpoints()  # store checkpoints here
model_dir = logger.path_model()             # store the trained model here

logger.log_scalar("train/loss", value=0.1, step=100)
logger.log_metric("train/accuracy", value=0.98, step=100)
logger.log_scalar("validation/loss", value=0.1, step=100)
logger.log_metric("validation/accuracy", value=0.95, step=100)
```

The tracker behaves differently when running locally or in the cloud.
Locally, experiment data is stored in a local folder `.data/experiments/{DATE_TIME}`.
In the cloud, the experiment data will be available in the Hafnia platform under
[experiments](https://hafnia.milestonesys.com/training-aas/experiments).

See [examples/example_logger.py](examples/example_logger.py) for the runnable version.  
See also [trainer-classification](https://github.com/milestone-hafnia/trainer-classification) on how
to use it in a real training script.

### Benchmarking models on a `HafniaDataset`

The benchmark utilities run an `InferenceModel` over a dataset, store the predictions as a new
task on a copy of the dataset and compute metrics (e.g. object-detection mAP) against the ground
truth. For the `InferenceModel` interface, available metrics and the full flow, see the
[Benchmarking](docs/benchmark.md) guide and the runnable
[examples/example_benchmark.py](examples/example_benchmark.py), which wraps a torchvision
SSDLite detector.

## Development

For development, we are using an uv based virtual python environment.

Install uv (linux and macOS)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create virtual environment and install python dependencies

```bash
uv sync --dev
```

Run tests:

```bash
uv run pytest tests
```
