from typing import Dict, List, Optional

from hafnia import http
from hafnia.utils import pretty_print_list_as_table, timed
from hafnia_cli.config import Config


@timed("Creating experiment.")
def create_experiment(
    experiment_name: str,
    dataset_recipe_id: str,
    trainer_id: str,
    exec_cmd: str,
    environment_id: str,
    cfg: Optional[Config] = None,
) -> Dict:
    cfg = cfg or Config()
    endpoint = cfg.get_platform_endpoint("experiments")
    headers = {"Authorization": cfg.api_key}
    response = http.post(
        endpoint,
        headers=headers,
        data={
            "name": experiment_name,
            "trainer": trainer_id,
            "dataset_recipe": dataset_recipe_id,
            "command": exec_cmd,
            "environment": environment_id,
        },
    )
    return response


@timed("Fetching environment info.")
def get_environments(cfg: Optional[Config] = None) -> List[Dict]:
    cfg = cfg or Config()
    endpoint = cfg.get_platform_endpoint("experiment_environments")
    headers = {"Authorization": cfg.api_key}
    envs: List[Dict] = http.fetch(endpoint, headers=headers)  # type: ignore

    if not isinstance(envs, list):
        raise ValueError(f"Expected list of environments from API, got: {envs}")

    # Endpoint does not support sorting; sort client-side by the platform-provided 'order' field
    # so callers can rely on a stable, intentional ordering (e.g. the first entry is the default).
    envs.sort(key=lambda env: env.get("order") if env.get("order") is not None else float("inf"))  # type: ignore
    return envs


def pretty_print_training_environments(envs: List[Dict]) -> None:
    ENV_FIELDS = {
        "Name": "name",
        "Instance": "instance",
        "GPU": "gpu",
        "GPU Count": "gpu_count",
        "GPU RAM": "vram",
        "CPU": "cpu",
        "CPU Count": "cpu_count",
        "RAM": "ram",
    }
    pretty_print_list_as_table(
        table_title="Available Training Environments",
        dict_items=envs,
        column_name_to_key_mapping=ENV_FIELDS,
    )


def get_exp_environment_id_from_name(name: str, cfg: Optional[Config] = None) -> str:
    envs = get_environments(cfg=cfg)

    for env in envs:
        if env["name"] == name:
            return env["id"]

    pretty_print_training_environments(envs)

    available_envs = [env["name"] for env in envs]

    raise ValueError(f"Environment '{name}' not found. Available environments are: {available_envs}")


@timed("Fetching experiment list.")
def get_experiments(
    cfg: Optional[Config] = None,
    ordering: str = "-created_at",
    limit: int = 1000,
    search: Optional[str] = None,
) -> List[Dict[str, str]]:
    """List available experiments on the Hafnia platform."""
    cfg = cfg or Config()
    endpoint = cfg.get_platform_endpoint("experiments")
    params = {"page_size": limit, "ordering": ordering}
    if search:
        params["search"] = search

    headers = {"Authorization": cfg.api_key}
    response: Dict = http.fetch(endpoint, headers=headers, params=params)
    experiments = response.get("data", [])
    return experiments


def pretty_print_experiments(experiments: List[Dict]) -> None:
    EXPERIMENT_FIELDS = {
        "ID": "id",
        "Name": "name",
        "State": "state",
        "Created At": "created_at",
        "Description": "description",
    }
    pretty_print_list_as_table(
        table_title="Available Experiments (most recent first)",
        dict_items=experiments,
        column_name_to_key_mapping=EXPERIMENT_FIELDS,
    )
