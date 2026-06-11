# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hafnia is a Python SDK and CLI for the Hafnia Platform — a Training-as-a-Service (TaaS) platform for training ML models on large, private datasets. It provides dataset management, experiment tracking (via MLflow), format conversions, and annotation primitives.

**Package manager:** `uv` (not pip). Always use `uv run` to execute commands.

## Common Commands

```bash
# Install dependencies
uv sync --dev

# Run all tests
uv run pytest tests

# Run tests excluding slow ones
uv run pytest tests -m "not slow"

# Run a single test file
uv run pytest tests/unit/dataset/test_hafnia_dataset.py

# Run a single test by name
uv run pytest tests/unit/dataset/test_hafnia_dataset.py -k "test_name"

# Linting and formatting
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type checking
uv run mypy src/ tests/

# Run all pre-commit hooks
pre-commit run --all-files

# Build package
uv build
```

## Architecture

Two packages live under `src/`:

### `hafnia` — Core SDK

- **`dataset/`** — Central module. `HafniaDataset` (in `hafnia_dataset.py`) is the main entry point for loading and manipulating datasets. Types live in `hafnia_dataset_types.py`, constants in `dataset_names.py`.
  - **`primitives/`** — Annotation types (Bbox, Polygon, Bitmask, Classification, Point, Segmentation), all Pydantic models inheriting from a base `Primitive`.
  - **`format_conversions/`** — Export to YOLO, COCO, Encord, ImageNet folder formats.
  - **`operations/`** — Dataset statistics, transformations, table operations, mask adjustment.
  - **`dataset_recipe/`** — Recipe system for composable dataset transformations.
- **`experiment/`** — `HafniaLogger` for experiment tracking via MLflow. `CommandBuilder` auto-generates CLI commands from docstrings.
- **`inference/`** — Model inference, benchmarking, and metrics (object detection mAP).
- **`platform/`** — API integration: dataset download, experiment creation, trainer packages, S3 utilities.
- **`visualizations/`** — Image annotation drawing utilities.

### `hafnia_cli` — CLI (Click-based)

Entry point: `hafnia_cli.__main__:main`. Configuration stored in `~/.hafnia/config.json` with keychain integration for API keys.

Command surface (invoke as `hafnia <group> <subcommand>`; use `uv run hafnia ...` during development):

```
hafnia configure                       Interactive first-time setup (profile name, API key, URL)
hafnia clear                           Remove all stored configuration

hafnia profile     ls | active | use | rm | create
hafnia dataset     ls | download | delete
hafnia recipe      ls | create | rm
hafnia trainer     ls | create | update | create-zip | view-zip
hafnia experiment  create | environments
hafnia runc        build | build-local | launch-local
```

Notes for working on the CLI:
- Each group lives in its own `*_cmds.py` module under `src/hafnia_cli/` and is registered in [src/hafnia_cli/__main__.py](src/hafnia_cli/__main__.py). The Click group name does not always match the module name — `recipe` is in [dataset_recipe_cmds.py](src/hafnia_cli/dataset_recipe_cmds.py), `trainer` in [trainer_package_cmds.py](src/hafnia_cli/trainer_package_cmds.py).
- Use `uv run hafnia <group> --help` (and `... <subcommand> --help`) to discover flags rather than grepping — Click renders the canonical surface.
- `experiment create` is generated from training-script docstrings via `hafnia.experiment.CommandBuilder` — adding a CLI flag for an experiment usually means editing the docstring/signature of the script, not the CLI module.
- Config is read/written through `hafnia_cli.config.Config` (active profile + `ConfigSchema`); avoid touching `~/.hafnia/config.json` directly in code or tests.
- API keys go through `hafnia_cli.keychain` when `use_keychain=True`; tests should not depend on a real keychain backend.
- New CLI commands must be exercised in [tests/integration/test_cli_integration.py](tests/integration/test_cli_integration.py). Each new subcommand should at minimum cover `--help`, missing-required-arg behaviour, and (where the platform is reachable) a real round-trip against the configured profile. The whole module is `@pytest.mark.slow` and skips when `is_hafnia_configured()` is False.

## Testing

- Framework: `pytest`. Marker: `slow` for long-running tests.
- Visual regression: The `compare_to_expected_image` fixture (in `tests/conftest.py`) auto-generates expected images on first run. If images don't match, a side-by-side debug image is saved to the pytest cache directory. Re-run the test after the expected image is created.
- Expected images stored in `tests/data/expected_images/`.
- Test data directories: `.data/datasets/` (downloaded datasets), `.data/experiments/` (local runs).

## Code Style

- **Logging:** Use loggers from `hafnia.log`, not `logging.getLogger()`. Two loggers: `user_logger` (clean output for end users) and `sys_logger` (detailed output with tracebacks/paths, level controlled by `HAFNIA_LOG` env var). Import as `from hafnia.log import user_logger`.
- **Ruff** rules: `I` (isort), `E`, `F`. Line length 120. `E501` is ignored.
- **MyPy** config is permissive (no strict flags).
- **Pre-commit hooks** run ruff-check, ruff-format, uv-lock, and mypy automatically. pytest is a manual-stage hook.
- Data models use **Pydantic v2** (`BaseModel`).
- DataFrames use **Polars** (not pandas).
- **Polars expressions:** When operating on the `samples` DataFrame, use native Polars expressions (`pl.col`, `pl.when`, `.struct.field`, `.list.eval`, etc.) as much as possible. Avoid `polars.Expr.apply` and `map_elements` unless there is no reasonable native alternative — native expressions are faster and parallelizable.

## HafniaDataset.samples DataFrame

The `samples` field on `HafniaDataset` is a `polars.DataFrame` built from a `List[Sample]` (see `HafniaDataset.from_samples_list`). Each `Sample` (Pydantic model in `hafnia_dataset_types.py`) is serialized to JSON and loaded into the DataFrame, so columns mirror `Sample` fields. Primitive columns (e.g. `classifications`, `bboxes`, `bitmasks`, `polygons`) are stored as `List[Struct]` — each struct mirrors the corresponding Pydantic primitive model. Use `pl.col("classifications").list.eval(...)` and `.struct.field("task_name")` to filter/extract within these nested columns.

## CI/CD

GitHub Actions on push/PR to `main`. Pipeline: lint → security scan (Trivy) → tests (Ubuntu + Windows matrix) → build → publish (PyPI + Docker to AWS ECR). Tests mount `HAFNIA_CONFIG` secret for integration tests.

## Default dependency group

`uv` default group is `test` (includes pytest, pre-commit, ruff). Use `--dev` flag for full dev dependencies (torch, torchvision, etc.).
