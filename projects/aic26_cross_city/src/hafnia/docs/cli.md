# Project Hafnia.

## Features

- **Platform Configuration**: Easy setup and management of Hafnia platform settings
A command-line interface tool for managing datasets, trainers, and experiments on the Hafnia Platform.

## Installation

Install the package and the CLI becomes available as `hafnia`:

```bash
pip install hafnia
```

## Configuration

Before using the CLI, configure it with your API key:

```bash
hafnia configure
```

You will be prompted for a profile name (defaults to `default`), your Hafnia API key, and the platform URL.
Verify your setup with:

```bash
hafnia profile active
```

## CLI Commands

### Core

| Command | Description |
|---|---|
| `hafnia configure` | Interactive setup of API key and platform URL |
| `hafnia clear` | Remove all stored configuration |
| `hafnia --version` | Show installed version |

---

### Profile Management

| Command | Description |
|---|---|
| `hafnia profile ls` | List all available profiles |
| `hafnia profile active` | Show details of the active profile |
| `hafnia profile use <name>` | Switch to a different profile |
| `hafnia profile create` | Create a new profile |
| `hafnia profile rm <name>` | Remove a profile |

---

### Dataset Management

**List datasets**
```bash
hafnia dataset ls [OPTIONS]
```
| Option | Default | Description |
|---|---|---|
| `-l, --limit` | `10` | Maximum number of results |
| `-s, --search` | | Filter by name |
| `-o, --ordering` | `name` | Sort order: `name`, `-name`, `created_at`, `-created_at`, `traceability_score`, `-traceability_score` |
| `-v, --visibility` | | Filter by visibility: `PUBLIC` or `ORGANIZATION` |

**Download a dataset**
```bash
hafnia dataset download <name> [OPTIONS]
```
| Option | Default | Description |
|---|---|---|
| `-v, --version` | | Dataset version to download |
| `--destination` | | Local path to download into |
| `--force` | | Re-download even if already present |

**Delete a dataset**
```bash
hafnia dataset delete [OPTIONS]
```
| Option | Default | Description |
|---|---|---|
| `--interactive / --no-interactive` | interactive | Prompt before deleting |

---

### Experiment Management

**List experiments**
```bash
hafnia experiment ls [OPTIONS]
```
| Option | Default | Description |
|---|---|---|
| `-l, --limit` | `1000` | Maximum number of results |
| `--ordering` | `-created_at` | Sort order: `created_at`, `-created_at`, `name`, `-name` |
| `-s, --search` | | Filter by name |

**List available training environments**
```bash
hafnia experiment environments
```

**Create and launch an experiment**
```bash
hafnia experiment create [OPTIONS]
```
| Option | Default | Description |
|---|---|---|
| `-n, --name` | `run-[DATETIME]` | Experiment name |
| `-c, --cmd` | `python scripts/train.py` | Command to run |
| `-e, --environment` | `Lite` | Training environment name (see `hafnia experiment environments`) |
| `-d, --dataset` | | Dataset name (e.g. `mnist`) |
| `-r, --recipe` | | Dataset recipe name |
| `--recipe-id` | | Dataset recipe ID |
| `-p, --trainer-path` | | Path to local trainer package directory |
| `-i, --trainer-id` | | Trainer package ID |

One dataset identifier (`--dataset`, `--recipe`, or `--recipe-id`) and one trainer identifier (`--trainer-path` or `--trainer-id`) are required.

---

### Trainer Package Management

**List trainer packages**
```bash
hafnia trainer ls [OPTIONS]
```
| Option | Default | Description |
|---|---|---|
| `-l, --limit` | `1000` | Maximum number of results |
| `--ordering` | `-created_at` | Sort order: `created_at`, `-created_at`, `name`, `-name`, `updated_at`, `-updated_at` |
| `--search` | | Filter by name |
| `-v, --visibility` | | Filter by visibility: `PUBLIC` or `ORGANIZATION` |

**Upload a trainer package**
```bash
hafnia trainer create <path> [OPTIONS]
```
| Option | Default | Description |
|---|---|---|
| `--name` | | Trainer package name |
| `--description` | | Trainer package description |

**Create a trainer zip (without uploading)**
```bash
hafnia trainer create-zip <path> [OPTIONS]
```
| Option | Default | Description |
|---|---|---|
| `--output` | `trainer.zip` | Output zip file path |

**Inspect a trainer zip**
```bash
hafnia trainer view-zip [OPTIONS]
```
| Option | Default | Description |
|---|---|---|
| `--path` | `trainer.zip` | Path to zip file |
| `--depth-limit` | | Max directory depth to display |

---

### Dataset Recipe Management

**List dataset recipes**
```bash
hafnia recipe ls [OPTIONS]
```
| Option | Default | Description |
|---|---|---|
| `-l, --limit` | `1000` | Maximum number of results |
| `-o, --ordering` | `-created_at` | Sort order: `created_at`, `-created_at`, `name`, `-name` |
| `-s, --search` | | Filter by name |

**Create a dataset recipe from a JSON file**
```bash
hafnia recipe create <path> [OPTIONS]
```
| Option | Default | Description |
|---|---|---|
| `-n, --name` | | Recipe name |

**Delete a dataset recipe**
```bash
hafnia recipe rm [OPTIONS]
```
| Option | Default | Description |
|---|---|---|
| `--id` | | Recipe ID |
| `--name` | | Recipe name |

---

### Run Container (runc)

| Command | Description |
|---|---|
| `hafnia runc launch <task>` | Launch a task within the container image |
| `hafnia runc build <recipe_url>` | Build a Docker image from a remote recipe |
| `hafnia runc build-local <recipe>` | Build a Docker image from a local recipe path |

---

## Example Usage

```bash
# Configure the CLI with a new profile
hafnia configure

# List all available profiles
hafnia profile ls

# Switch to a different profile
hafnia profile use profile_2

# List your experiments (newest first)
hafnia experiment ls

# List experiments matching a search term
hafnia experiment ls --search yolo

# See available training environments
hafnia experiment environments

# Launch an experiment with a local trainer and a dataset
hafnia experiment create --dataset mnist --trainer-path ../trainer-classification

# Launch an experiment with existing trainer and recipe IDs
hafnia experiment create \
  --recipe my-recipe \
  --trainer-id 5e454c0d-fdf1-4d1f-9732-771d7fecd28e \
  --name "My Experiment" \
  --environment "Lite"

# List available datasets
hafnia dataset ls --search coco --ordering -traceability_score

# Download a dataset
hafnia dataset download mnist --force

# List trainer packages
hafnia trainer ls

# Upload a trainer package
hafnia trainer create ../trainer-classification --name "My Trainer"

# List dataset recipes
hafnia recipe ls
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `HAFNIA_API_KEY` | API key (overrides config file and keychain) |
| `HAFNIA_PLATFORM_URL` | Platform URL (overrides config file) |
| `HAFNIA_PROFILE_NAME` | Active profile name (overrides config file) |
| `MDI_CONFIG_PATH` | Custom path to the configuration file |
| `HAFNIA_CLOUD` | Set to `true` to emulate cloud behaviour |
| `HAFNIA_LOG` | Log level for system messages (e.g. `DEBUG`, `INFO`, `WARNING`) |