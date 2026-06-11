from pathlib import Path
from typing import Dict, Optional

import click
from rich import print as rprint

from hafnia_cli.config import Config


@click.group(name="recipe")
def dataset_recipe() -> None:
    """Dataset recipe commands"""
    pass


@dataset_recipe.command(name="create")
@click.argument("path_json_recipe", required=True)
@click.option(
    "-n",
    "--name",
    type=str,
    default=None,
    show_default=True,
    help="Name of the dataset recipe.",
)
@click.pass_obj
def cmd_get_or_create_dataset_recipe(cfg: Config, path_json_recipe: Path, name: Optional[str]) -> None:
    """Create Hafnia dataset recipe from dataset recipe JSON file"""
    from hafnia.platform.dataset_recipe import get_or_create_dataset_recipe_from_path

    recipe = get_or_create_dataset_recipe_from_path(path_json_recipe, name=name, cfg=cfg)

    if recipe is None:
        raise click.ClickException("Failed to create dataset recipe.")

    rprint(recipe)


@dataset_recipe.command(name="ls")
@click.pass_obj
@click.option("-l", "--limit", type=int, default=1000, help="Limit number of listed dataset recipes.")
@click.option(
    "-o",
    "--ordering",
    type=click.Choice(["created_at", "-created_at", "name", "-name"], case_sensitive=False),
    default="-created_at",
    help="Ordering of listed dataset recipes.",
)
@click.option("-s", "--search", type=str, default=None, help="Search term to filter dataset recipes by name.")
def cmd_list_dataset_recipes(
    cfg: Config,
    limit: int,
    ordering: str,
    search: Optional[str] = None,
) -> None:
    """List available dataset recipes"""
    from hafnia.platform.dataset_recipe import get_dataset_recipes, pretty_print_dataset_recipes

    recipes = get_dataset_recipes(cfg=cfg, limit=limit, ordering=ordering, search=search)
    pretty_print_dataset_recipes(recipes)


@dataset_recipe.command(name="rm")
@click.option("-i", "--id", type=str, help="Dataset recipe ID to delete.")
@click.option("-n", "--name", type=str, help="Dataset recipe name to delete.")
@click.pass_obj
def cmd_delete_dataset_recipe(cfg: Config, id: Optional[str], name: Optional[str]) -> Dict:
    """Delete a dataset recipe by ID or name"""
    from hafnia.platform.dataset_recipe import delete_dataset_recipe_by_id, delete_dataset_recipe_by_name

    if id is not None:
        return delete_dataset_recipe_by_id(id=id, cfg=cfg)
    if name is not None:
        dataset_recipe = delete_dataset_recipe_by_name(name=name, cfg=cfg)
        if dataset_recipe is None:
            raise click.ClickException(f"Dataset recipe with name '{name}' was not found.")

        return dataset_recipe

    raise click.MissingParameter(
        "No dataset recipe identifier have been given. Provide either --id or --name. "
        "Get available recipes with 'hafnia recipe ls'."
    )
