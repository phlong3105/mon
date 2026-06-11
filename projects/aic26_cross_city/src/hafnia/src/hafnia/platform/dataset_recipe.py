import json
from pathlib import Path
from typing import Dict, List, Optional

from flatten_dict import flatten

from hafnia import http
from hafnia.log import user_logger
from hafnia.utils import pretty_print_list_as_table, timed
from hafnia_cli.config import Config


@timed("Get or create dataset recipe")
def get_or_create_dataset_recipe(
    recipe: dict,
    name: Optional[str] = None,
    overwrite: bool = False,
    cfg: Optional[Config] = None,
) -> Optional[Dict]:
    cfg = cfg or Config()

    endpoint = cfg.get_platform_endpoint("dataset_recipes")
    headers = {"Authorization": cfg.api_key}
    data = {"template": {"body": recipe}, "overwrite": overwrite}
    if name is not None:
        data["name"] = name  # type: ignore[assignment]

    response = http.post(endpoint, headers=headers, data=data)
    return response


def get_or_create_dataset_recipe_by_dataset_name(dataset_name: str, cfg: Optional[Config] = None) -> Dict:
    return get_or_create_dataset_recipe(recipe=dataset_name, cfg=cfg)


def get_dataset_recipes(
    cfg: Optional[Config] = None,
    ordering: str = "-created_at",
    limit: int = 100,
    search: Optional[str] = None,
) -> List[Dict]:
    cfg = cfg or Config()
    endpoint = cfg.get_platform_endpoint("dataset_recipes")
    params = {"page_size": limit, "ordering": ordering, "is_direct_dataset_reference": False}
    if search:
        params["search"] = search
    headers = {"Authorization": cfg.api_key}
    response: Dict = http.fetch(endpoint, headers=headers, params=params)
    dataset_recipes = response.get("data", [])
    return dataset_recipes


def get_dataset_recipe_by_id(dataset_recipe_id: str, cfg: Optional[Config] = None) -> Dict:
    cfg = cfg or Config()
    endpoint = cfg.get_platform_endpoint("dataset_recipes")
    headers = {"Authorization": cfg.api_key}
    full_url = f"{endpoint}/{dataset_recipe_id}"
    dataset_recipe_info: Dict = http.fetch(full_url, headers=headers)  # type: ignore[assignment]
    if not dataset_recipe_info:
        raise ValueError(f"Dataset recipe with ID '{dataset_recipe_id}' was not found.")
    return dataset_recipe_info


def get_or_create_dataset_recipe_from_path(
    path_recipe_json: Path, name: Optional[str] = None, cfg: Optional[Config] = None
) -> Dict:
    path_recipe_json = Path(path_recipe_json)
    if not path_recipe_json.exists():
        raise FileNotFoundError(f"Dataset recipe file '{path_recipe_json}' does not exist.")
    json_dict = json.loads(path_recipe_json.read_text())
    return get_or_create_dataset_recipe(json_dict, name=name, cfg=cfg)


def delete_dataset_recipe_by_id(id: str, cfg: Optional[Config] = None) -> Dict:
    cfg = cfg or Config()
    endpoint = cfg.get_platform_endpoint("dataset_recipes")
    headers = {"Authorization": cfg.api_key}
    full_url = f"{endpoint}/{id}"
    response = http.delete(endpoint=full_url, headers=headers)
    return response


@timed("Get dataset recipe")
def get_dataset_recipe_by_name(name: str, cfg: Optional[Config] = None) -> Optional[Dict]:
    cfg = cfg or Config()

    endpoint = cfg.get_platform_endpoint("dataset_recipes")
    headers = {"Authorization": cfg.api_key}
    params = {"name__iexact": name}
    response: Dict = http.fetch(endpoint, headers=headers, params=params)
    dataset_recipes = response.get("data", [])
    if len(dataset_recipes) == 0:
        return None

    if len(dataset_recipes) > 1:
        user_logger.warning(f"Found {len(dataset_recipes)} dataset recipes called '{name}'. Using the first one.")

    dataset_recipe = dataset_recipes[0]
    return dataset_recipe


def delete_dataset_recipe_by_name(name: str, cfg: Optional[Config] = None) -> Optional[Dict]:
    recipe_response = get_dataset_recipe_by_name(name, cfg=cfg)

    if recipe_response:
        return delete_dataset_recipe_by_id(recipe_response["id"], cfg=cfg)
    return recipe_response


def pretty_print_dataset_recipes(recipes: List[Dict]) -> None:
    recipes = [flatten(recipe, reducer="dot", max_flatten_depth=2) for recipe in recipes]  # noqa: F821
    for recipe in recipes:
        datasets = recipe.get("template.datasets", [])
        recipe["datasets"] = ", ".join(dataset.get("name", "") for dataset in datasets)

    RECIPE_FIELDS = {
        "ID": "id",
        "Name": "name",
        "Datasets": "datasets",
        "Created": "created_at",
    }
    pretty_print_list_as_table(
        table_title="Available Dataset Recipes",
        dict_items=recipes,
        column_name_to_key_mapping=RECIPE_FIELDS,
    )
