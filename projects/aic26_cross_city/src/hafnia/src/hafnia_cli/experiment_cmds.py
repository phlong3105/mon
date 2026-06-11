from pathlib import Path
from typing import Dict, Optional

import click

from hafnia import utils
from hafnia.platform.dataset_recipe import (
    get_dataset_recipe_by_id,
    get_dataset_recipe_by_name,
    get_or_create_dataset_recipe_by_dataset_name,
)
from hafnia.platform.trainer_package import create_trainer_package
from hafnia_cli.config import Config


@click.group(name="experiment")
def experiment() -> None:
    """Experiment management commands"""
    pass


@experiment.command(name="ls")
@click.option("-l", "--limit", type=int, default=1000, help="Limit number of listed experiments.")
@click.option(
    "--ordering",
    type=click.Choice(["created_at", "-created_at", "name", "-name"], case_sensitive=False),
    default="-created_at",
    help="Ordering of listed experiments.",
)
@click.option("-s", "--search", type=str, default=None, help="Search term to filter experiments by name.")
@click.pass_obj
def cmd_list_experiments(cfg: Config, limit: int, ordering: str, search: Optional[str] = None) -> None:
    """List available experiments on the Hafnia platform."""
    from hafnia.platform.experiment import get_experiments, pretty_print_experiments

    experiments = get_experiments(cfg=cfg, limit=limit, ordering=ordering, search=search)
    pretty_print_experiments(experiments)


@experiment.command(name="environments")
@click.pass_obj
def cmd_view_environments(cfg: Config):
    """
    View available experiment training environments.
    """
    from hafnia.platform import get_environments, pretty_print_training_environments

    envs = get_environments(cfg=cfg)

    pretty_print_training_environments(envs)


def default_experiment_run_name():
    return f"run-{utils.now_as_str()}"


@experiment.command(name="create")
@click.option(
    "-n",
    "--name",
    type=str,
    default=default_experiment_run_name(),
    required=False,
    help=f"Name of the experiment. [default: run-[DATETIME] e.g. {default_experiment_run_name()}] ",
)
@click.option(
    "-c",
    "--cmd",
    type=str,
    default="python scripts/train.py",
    show_default=True,
    help="Command to run the experiment.",
)
@click.option(
    "-p",
    "--trainer-path",
    type=Path,
    default=None,
    help="Path to the trainer package directory. ",
)
@click.option(
    "-i",
    "--trainer-id",
    type=str,
    default=None,
    help="ID of the trainer package. View available trainers with 'hafnia trainer ls'",
)
@click.option(
    "-d",
    "--dataset",
    type=str,
    default=None,
    required=False,
    help="DatasetIdentifier: Name of the dataset. View Available datasets with 'hafnia dataset ls'",
)
@click.option(
    "-r",
    "--recipe",
    type=str,
    default=None,
    required=False,
    help="DatasetIdentifier: Name of the dataset recipe. View available dataset recipes with 'hafnia recipe ls'",
)
@click.option(
    "--recipe-id",
    type=str,
    default=None,
    required=False,
    help="DatasetIdentifier: ID of the dataset recipe. View dataset recipes with 'hafnia recipe ls'",
)
@click.option(
    "-e",
    "--environment",
    type=str,
    default=None,
    help=(
        "Experiment environment name. View available environments with 'hafnia experiment environments'. "
        "Defaults to the first environment returned by the platform (sorted by 'order')."
    ),
)
@click.pass_obj
def cmd_create_experiment(
    cfg: Config,
    name: str,
    cmd: str,
    trainer_path: Path,
    trainer_id: Optional[str],
    dataset: Optional[str],
    recipe: Optional[str],
    recipe_id: Optional[str],
    environment: Optional[str],
) -> None:
    """
    Create and launch a new experiment run

    Requires one dataset recipe and one trainer package:.
        - One dataset identifier is required either '--dataset', '--recipe' or '--recipe-id'.
        - One trainer identifier is required either '--trainer-path' or '--trainer-id'.

    \b
    Examples:
    # Launch an experiment with a dataset and a trainer package from local path
    hafnia experiment create --dataset mnist --trainer-path ../trainer-classification

    \b
    # Launch experiment with dataset recipe by name and trainer package by id
    hafnia experiment create --recipe mnist-recipe --trainer-id 5e454c0d-fdf1-4d1f-9732-771d7fecd28e

    \b
    # Show available options:
    hafnia experiment create --name "My Experiment" -d mnist --cmd "python scripts/train.py" -e "Lite" -p ../trainer-classification
    """
    from hafnia.platform import create_experiment, get_exp_environment_id_from_name
    from hafnia.platform.experiment import get_environments

    dataset_recipe_response = get_dataset_recipe_by_identifiers(
        cfg=cfg,
        dataset_name=dataset,
        recipe_name=recipe,
        recipe_id=recipe_id,
    )
    recipe_id = dataset_recipe_response["id"]

    trainer_id = get_trainer_package_by_identifiers(cfg=cfg, trainer_path=trainer_path, trainer_id=trainer_id)

    if environment is None:
        envs = get_environments(cfg=cfg)
        if not envs:
            raise click.ClickException("No experiment environments available on the platform.")
        # Pick the first environment (defined by the order-field provided by the platform)
        # as the default if no environment is specified
        env_id = envs[0]["id"]
    else:
        env_id = get_exp_environment_id_from_name(name=environment, cfg=cfg)

    experiment = create_experiment(
        experiment_name=name,
        dataset_recipe_id=recipe_id,
        trainer_id=trainer_id,
        exec_cmd=cmd,
        environment_id=env_id,
        cfg=cfg,
    )

    experiment_properties = {
        "ID": experiment.get("id", "N/A"),
        "Name": experiment.get("name", "N/A"),
        "State": experiment.get("state", "N/A"),
        "Trainer Package ID": experiment.get("trainer", "N/A"),
        "Dataset Recipe ID": experiment.get("dataset_recipe", "N/A"),
        "Dataset ID": experiment.get("dataset", "N/A"),
        "Created At": experiment.get("created_at", "N/A"),
    }
    print("Successfully created experiment: ")
    for key, value in experiment_properties.items():
        print(f"  {key}: {value}")


def get_dataset_recipe_by_identifiers(
    cfg: Config,
    dataset_name: Optional[str],
    recipe_name: Optional[str],
    recipe_id: Optional[str],
) -> Dict:
    dataset_identifiers = [dataset_name, recipe_name, recipe_id]
    n_dataset_identifies_defined = sum([bool(identifier) for identifier in dataset_identifiers])

    if n_dataset_identifies_defined > 1:
        raise click.ClickException(
            "Multiple dataset identifiers have been provided. Define only one dataset identifier."
        )

    if dataset_name:
        return get_or_create_dataset_recipe_by_dataset_name(dataset_name, cfg=cfg)

    if recipe_name:
        recipe = get_dataset_recipe_by_name(recipe_name, cfg=cfg)
        if recipe is None:
            raise click.ClickException(f"Dataset recipe '{recipe_name}' was not found in the dataset library.")
        return recipe

    if recipe_id:
        return get_dataset_recipe_by_id(recipe_id, cfg=cfg)

    raise click.MissingParameter(
        "At least one dataset identifier must be provided. Set one of the following:\n"
        "  --dataset <name>  -- E.g. '--dataset mnist'\n"
        "  --recipe <name>  -- E.g. '--recipe my-recipe'\n"
        "  --recipe-id <id>  -- E.g. '--recipe-id 5e454c0d-fdf1-4d1f-9732-771d7fecd28e'\n"
    )


def get_trainer_package_by_identifiers(
    cfg: Config,
    trainer_path: Optional[Path],
    trainer_id: Optional[str],
) -> str:
    from hafnia.platform import get_trainer_package_by_id

    if trainer_path is not None and trainer_id is not None:
        raise click.ClickException(
            "Multiple trainer identifiers (--trainer-path, --trainer-id) have been provided. Define only one."
        )

    if trainer_path is not None:
        trainer_path = Path(trainer_path)
        if not trainer_path.exists():
            raise click.ClickException(f"Trainer package path '{trainer_path}' does not exist.")
        response = create_trainer_package(
            source_dir=trainer_path,
            cfg=cfg,
        )
        return response["id"]

    if trainer_id:
        trainer_response = get_trainer_package_by_id(id=trainer_id, cfg=cfg)
        return trainer_response["id"]

    raise click.MissingParameter(
        "At least one trainer identifier must be provided. Set one of the following:\n"
        "  --trainer-path <path>  -- E.g. '--trainer-path .'\n"
        "  --trainer-id <id>  -- E.g. '--trainer-id 5e454c0d-fdf1-4d1f-9732-771d7fecd28e'\n"
    )
