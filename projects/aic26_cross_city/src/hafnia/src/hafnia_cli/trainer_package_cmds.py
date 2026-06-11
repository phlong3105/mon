from pathlib import Path
from typing import Dict, Optional

import click

import hafnia_cli.consts as consts
from hafnia_cli.config import Config


@click.group(name="trainer")
def trainer_package() -> None:
    """Trainer package commands"""
    pass


@trainer_package.command(name="ls")
@click.pass_obj
@click.option("-l", "--limit", type=int, default=1000, help="Limit number of listed trainer packages.")
@click.option(
    "--ordering",
    type=click.Choice(["created_at", "-created_at", "name", "-name"], case_sensitive=False),
    default="-created_at",
    help="Ordering of listed trainer packages.",
)
@click.option("--search", type=str, default=None, help="Search term to filter trainer packages by name.")
@click.option(
    "-v",
    "--visibility",
    default=None,
    type=click.Choice(["PUBLIC", "ORGANIZATION"], case_sensitive=False),
    help="Filter trainer packages by visibility.",
)
def cmd_list_trainer_packages(
    cfg: Config,
    limit: int,
    ordering: str,
    search: Optional[str] = None,
    visibility: Optional[str] = None,
) -> None:
    """List available trainer packages on the platform"""

    from hafnia.platform.trainer_package import get_trainer_packages, pretty_print_trainer_packages

    trainers = get_trainer_packages(cfg=cfg, limit=limit, ordering=ordering, search=search, visibility=visibility)

    pretty_print_trainer_packages(trainers)


@trainer_package.command(name="create")
@click.pass_obj
@click.argument(
    "path",
    type=Path,
)
@click.option(
    "-n",
    "--name",
    type=str,
    default=None,
    help="Name of the trainer package.",
)
@click.option(
    "-d",
    "--description",
    type=str,
    default=None,
    help="Description of the trainer package.",
)
@click.option(
    "--cmd",
    type=Optional[str],
    default=None,
    show_default=True,
    help="Default command to run the trainer package.",
)
def cmd_create_trainer_package(
    cfg: Config,
    path: Path,
    cmd: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict:
    """Create a trainer package on the platform"""
    from hafnia.platform.trainer_package import create_trainer_package

    path_trainer = Path(path).resolve()
    trainer_response = create_trainer_package(
        source_dir=path_trainer,
        name=name,
        description=description,
        cmd=cmd,
        cfg=cfg,
    )

    return trainer_response


@trainer_package.command(name="update")
@click.pass_obj
@click.argument("id", type=str)
@click.option(
    "--path",
    type=Path,
    default=None,
    help="Path to a trainer package source directory. If omitted, only metadata is updated.",
)
@click.option("-n", "--name", type=str, default=None, help="New name for the trainer package.")
@click.option("-d", "--description", type=str, default=None, help="New description for the trainer package.")
@click.option("--cmd", type=str, default=None, help="New default command to run the trainer package.")
def cmd_update_trainer_package(
    cfg: Config,
    id: str,
    path: Optional[Path] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    cmd: Optional[str] = None,
) -> Dict:
    """Update an existing trainer package on the platform by id.

    All of `--path`, `--name`, `--description`, `--cmd` are optional, but at least one must be provided.
    """
    from hafnia.platform.trainer_package import update_trainer_package

    if path is None and name is None and description is None and cmd is None:
        raise click.UsageError("Provide at least one of --path, --name, --description, or --cmd.")

    source_dir = Path(path).resolve() if path is not None else None
    return update_trainer_package(
        id=id,
        source_dir=source_dir,
        name=name,
        description=description,
        cmd=cmd,
        cfg=cfg,
    )


@trainer_package.command(name="create-zip")
@click.argument("source")
@click.option(
    "--output",
    type=click.Path(writable=True),
    default="./trainer.zip",
    show_default=True,
    help="Output trainer package path.",
)
def cmd_create_trainer_package_zip(source: str, output: str) -> None:
    """Create Hafnia trainer package as zip-file from local path"""

    from hafnia.utils import archive_dir

    path_output_zip = Path(output)
    if path_output_zip.suffix != ".zip":
        raise click.ClickException(consts.ERROR_TRAINER_PACKAGE_FILE_FORMAT)

    path_source = Path(source)
    path_output_zip, _ = archive_dir(path_source, path_output_zip)


@trainer_package.command(name="view-zip")
@click.option("--path", type=str, default="./trainer.zip", show_default=True, help="Path of trainer.zip.")
@click.option("--depth-limit", type=int, default=3, help="Limit the depth of the tree view.", show_default=True)
def cmd_view_trainer_package_zip(path: str, depth_limit: int) -> None:
    """View the content of a trainer package zip file."""
    from hafnia.utils import show_trainer_package_content

    path_trainer_package = Path(path)
    if not path_trainer_package.exists():
        raise click.ClickException(
            f"Trainer package file '{path_trainer_package}' does not exist. Please provide a valid path. "
            f"To create a trainer package, use the 'hafnia trainer create-zip' command."
        )
    show_trainer_package_content(path_trainer_package, depth_limit=depth_limit)
