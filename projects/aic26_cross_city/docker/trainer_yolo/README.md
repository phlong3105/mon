# Trainer Package: Ultralytics YOLO Detection on Hafnia

This trainer package adapts Ultralytics YOLO detection training to Hafnia Training aaS.
It mirrors the same cloud/local flow as the classification template, but uses object detection datasets.

## What this trainer does

- Loads dataset from Hafnia cloud path when running in cloud.
- Loads a sample dataset by name/version for local development.
- Converts Hafnia samples into an Ultralytics-compatible folder layout:
  - images/train, images/val, images/test
  - labels/train, labels/val, labels/test
  - dataset.yaml
- Trains a YOLO model with Ultralytics.
- Copies resulting weights to Hafnia output locations:
  - best model: model/model.pt
  - checkpoints: checkpoints/best.pt and checkpoints/last.pt
- Logs final key metrics into Hafnia logger when results.csv is available.

## Project structure

```
trainer_yolo/
  scripts/
    train.py
    train.schema.json
  src/trainer_yolo/
  Dockerfile
  .hafniaignore
  pyproject.toml
```

## Local setup

```bash
cd trainer-yolo-detection
uv sync
uv pip install -e .
```

Configure Hafnia access (optional for local-only runs):

```bash
hafnia configure
```

## Run locally

Build docker
```bash
cd trainer-yolo-detection
hafnia trainer create-zip .
hafnia runc build-local trainer.zip
```


```bash
hafnia runc launch-local --dataset eccv-cross-city "python scripts/train.py --model yolo26l --epochs 1 --model_path ./pretrained_models/yolo26l.zip"
```

## Important CLI arguments

# Ultralytics YOLO Training

**Usage:** `train [ARGS]`
*Ultralytics YOLO Training for Hafnia object detection datasets*

## ── Commands ──

| Command | Description |
| :--- | :--- |
| `-h`, `--help` | Display this message and exit. |
| `--version` | Display application version. |

## ── Parameters ──

### Project & Dataset Settings
| Argument | Option / Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| `PROJECT-NAME` | `--project-name` | Project name for experiment tracking. | `YOLO Detection` |
| `DATASET-NAME` | `--dataset-name` | Local sample dataset name outside Hafnia. | `midwest-vehicle-detection` |
| `DATASET-VERSION` | `--dataset-version` | Dataset version used for local runs. | `1.0.1` |
| `TASK-NAME` | `--task-name` | Optional bounding box task name. | *None* |

### Model & Training Configuration
| Argument | Option / Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| `MODEL-PATH` | `--model-path` | Path to a compressed (`.zip`) pretrained model used as the training starting point.<br>Options:<br>• `pretrained_models/yolo26n.zip`<br>• `pretrained_models/yolo26s.zip`<br>• `pretrained_models/yolo26l.zip` | *None* |
| `PRETRAINED` | `--pretrained`<br>`--no-pretrained` | Initialize the model from pretrained weights. | `True` |
| `MODEL` | `--model` | Ultralytics model name (e.g. `yolo26l.pt`) or path to `.zip`. | `yolo26l.pt` |
| `EPOCHS` | `--epochs` | Number of training epochs. | `20` |
| `IMAGE-SIZE` | `--image-size` | Training image size. | `640` |
| `BATCH-SIZE` | `--batch-size` | Training batch size. Use `-1` for auto-batching. | `16` |
| `LEARNING-RATE` | `--learning-rate` | Initial learning rate. | `0.01` |
| `NUM-WORKERS` | `--num-workers` | Data loader workers. | `8` |
| `PATIENCE` | `--patience` | Early stopping patience. | `30` |
| `DEVICE` | `--device` | Device string: `cpu`, `0`, `0,1` etc. | `""` |
| `SEED` | `--seed` | Random seed. | `42` |
| `FORCE-REEXPORT` | `--force-reexport`<br>`--no-force-reexport` | Rebuild YOLO dataset. | `False` |               │

## Package and launch with Hafnia

Create zip package:

```bash
hafnia trainer create-zip .
```

Launch experiment from CLI:

```bash
hafnia experiment create --dataset midwest-vehicle-detection --trainer-path . --cmd "python scripts/train.py --model yolov8n.pt --epochs 50"
```

## Notes

- Validation split fallback: if dataset has no validation split, this trainer uses test split as validation.
- You can change dataset conversion behavior in scripts/train.py if you need a custom split policy.
