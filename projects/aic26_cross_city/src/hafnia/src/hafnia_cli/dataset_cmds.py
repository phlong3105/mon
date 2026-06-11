from pathlib import Path
from typing import Optional

import click

import hafnia.dataset.hafnia_dataset
from hafnia import utils
from hafnia_cli.config import Config


@click.group()
def dataset():
    """Manage dataset interaction"""
    pass


ordering_options = ["name", "-name", "created_at", "-created_at", "traceability_score", "-traceability_score"]


@dataset.command("ls")
@click.option("--limit", "-l", type=int, default=10, help="Limit the number of datasets displayed.")
@click.option("--search", "-s", default=None, help="Search term to filter datasets.")
@click.option("--ordering", "-o", type=click.Choice(ordering_options), default="name", help="Ordering of the datasets.")
@click.option(
    "-v",
    "--visibility",
    default=None,
    type=click.Choice(["PUBLIC", "ORGANIZATION"], case_sensitive=False),
    help="Filter datasets by visibility.",
)
@click.pass_obj
def cmd_list_datasets(cfg: Config, limit: int, search: Optional[str], ordering: str, visibility: Optional[str]) -> None:
    """List available datasets on Hafnia platform"""
    from hafnia.platform.datasets import get_datasets, pretty_print_datasets

    datasets = get_datasets(cfg=cfg, limit=limit, search=search, ordering=ordering, visibility=visibility)
    pretty_print_datasets(datasets)


@dataset.command("download")
@click.argument("dataset_name")
@click.option(
    "--version",
    "-v",
    default="latest",
    required=False,
    help="Dataset version to download e.g. '0.0.1' or '1.0.1'. Defaults to the latest version.",
)
@click.option(
    "--destination",
    "-d",
    default=None,
    required=False,
    help=f"Destination folder to save the dataset. Defaults to '{utils.PATH_DATASETS}/<dataset_name>'",
)
@click.option("--force", "-f", is_flag=True, default=False, help="Flag to enable force redownload")
@click.pass_obj
def cmd_dataset_download(
    cfg: Config, dataset_name: str, version: Optional[str], destination: Optional[click.Path], force: bool
) -> Path:
    """Download dataset from Hafnia platform"""

    path_dataset = hafnia.dataset.hafnia_dataset.download_or_get_dataset_path(
        dataset_name=dataset_name,
        version=version,
        cfg=cfg,
        path_datasets_folder=destination,
        force_redownload=force,
    )
    return path_dataset


@dataset.command("delete")
@click.argument("dataset_name")
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Whether to ask for confirmation before deleting the dataset.",
)
@click.pass_obj
def cmd_dataset_delete(cfg: Config, dataset_name: str, interactive: bool) -> None:
    """Delete dataset from Hafnia platform"""
    from hafnia.platform import datasets

    datasets.delete_dataset_completely_by_name(
        dataset_name=dataset_name,
        interactive=interactive,
        cfg=cfg,
    )
