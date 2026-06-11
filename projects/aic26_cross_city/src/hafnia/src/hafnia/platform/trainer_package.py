import json
from pathlib import Path
from typing import Dict, List, Optional

from hafnia import http
from hafnia.experiment.command_builder import DEFAULT_ORDER
from hafnia.log import user_logger
from hafnia.utils import (
    archive_dir,
    get_trainer_package_path,
    pretty_print_list_as_table,
    timed,
)
from hafnia_cli.config import Config


@timed("Uploading trainer package.")
def create_trainer_package(
    source_dir: Path,
    name: Optional[str] = None,
    description: Optional[str] = None,
    cmd: Optional[str] = None,
    cfg: Optional[Config] = None,
) -> Dict:
    # Ensure the path is absolute to handle '.' paths are given an appropriate name.
    source_dir = Path(source_dir).resolve()
    cfg = cfg or Config()
    endpoint = cfg.get_platform_endpoint("trainers")

    path_trainer = get_trainer_package_path(trainer_name=source_dir.name)
    name = name or path_trainer.stem
    zip_path, package_files = archive_dir(source_dir, output_path=path_trainer)
    user_logger.info(f"Trainer package created and stored in '{path_trainer}'")

    cmd_builder_schemas = auto_discover_cmd_builder_schemas(package_files)
    cmd = cmd or "python scripts/train.py"
    description = description or f"Trainer package for '{name}'. Created with Hafnia SDK Cli."
    headers = {"Authorization": cfg.api_key, "accept": "application/json"}
    data = {
        "name": name,
        "description": description,
        "default_command": cmd,
        "file": (zip_path.name, Path(zip_path).read_bytes()),
    }
    if len(cmd_builder_schemas) > 0:
        data["command_builder_schemas"] = json.dumps(cmd_builder_schemas)
    user_logger.info(f"Uploading trainer package '{name}' to platform...")
    response = http.post(endpoint, headers=headers, data=data, multipart=True)
    user_logger.info(f"Trainer package uploaded successfully with id '{response['id']}'")
    return response


def auto_discover_cmd_builder_schemas(package_files: List[Path]) -> List[Dict]:
    """
    Auto-discover command builder schema files in the trainer package files.
    Looks for files ending with '.schema.json' and loads their content as JSON.

    Only files that are part of the trainer package are considered, so '.schema.json' files excluded by
    '.hafniaignore' (e.g. in '.venv' or other ignored directories) are not picked up.
    """
    cmd_builder_schema_files = [file for file in package_files if file.name.endswith(".schema.json")]
    cmd_builder_schemas = []
    for cmd_builder_schema_file in cmd_builder_schema_files:
        try:
            cmd_builder_schema = json.loads(cmd_builder_schema_file.read_text())
        except json.JSONDecodeError:
            user_logger.warning(f"Could not parse '{cmd_builder_schema_file}' as JSON. Skipping.")
            continue
        required_fields = ["cmd", "json_schema"]
        missing_field = next((field for field in required_fields if field not in cmd_builder_schema), None)
        if missing_field is not None:
            user_logger.warning(
                f"One of the discovered command builder schema '{cmd_builder_schema_file}' is not considered "
                f"a command builder schema because it is missing the '{missing_field}' field. "
                "Skipping this file.",
            )
            continue
        cmd_entrypoint = cmd_builder_schema["cmd"]
        user_logger.info(f"Found command builder schema file for entry point '{cmd_entrypoint}'")
        cmd_builder_schemas.append(cmd_builder_schema)

    # Sort the schemas by the 'order' field, with missing 'order' treated as DEFAULT_ORDER
    cmd_builder_schemas.sort(key=lambda s: s.get("order", DEFAULT_ORDER))
    return cmd_builder_schemas


@timed("Updating trainer package.")
def update_trainer_package(
    id: str,
    source_dir: Optional[Path] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    cmd: Optional[str] = None,
    cfg: Optional[Config] = None,
) -> Dict:
    if source_dir is None and name is None and description is None and cmd is None:
        raise ValueError("Provide at least one of source_dir, name, description, or cmd to update.")

    cfg = cfg or Config()
    endpoint = f"{cfg.get_platform_endpoint('trainers')}/{id}"
    headers = {"Authorization": cfg.api_key, "accept": "application/json"}

    fields: Dict = {}
    if name is not None:
        fields["name"] = name
    if description is not None:
        fields["description"] = description
    if cmd is not None:
        fields["default_command"] = cmd

    if source_dir is not None:
        source_dir = Path(source_dir).resolve()
        path_trainer = get_trainer_package_path(trainer_name=source_dir.name)
        zip_path, package_files = archive_dir(source_dir, output_path=path_trainer)
        user_logger.info(f"Trainer package created and stored in '{path_trainer}'")

        cmd_builder_schemas = auto_discover_cmd_builder_schemas(package_files)
        if len(cmd_builder_schemas) > 0:
            fields["command_builder_schemas"] = json.dumps(cmd_builder_schemas)
        fields["file"] = (zip_path.name, Path(zip_path).read_bytes())
        user_logger.info(f"Updating trainer package '{id}' on platform...")
        response = http.patch(endpoint, headers=headers, data=fields, multipart=True)
    else:
        user_logger.info(f"Updating trainer package '{id}' metadata on platform...")
        response = http.patch(endpoint, headers=headers, data=fields)

    user_logger.info(f"Trainer package '{response.get('id', id)}' updated successfully")
    return response


@timed("Get trainer package.")
def get_trainer_package_by_id(id: str, cfg: Optional[Config] = None) -> Dict:
    cfg = cfg or Config()
    endpoint = cfg.get_platform_endpoint("trainers")
    full_url = f"{endpoint}/{id}"
    headers = {"Authorization": cfg.api_key}
    response: Dict = http.fetch(full_url, headers=headers)  # type: ignore[assignment]

    return response


@timed("Get trainer packages")
def get_trainer_packages(
    cfg: Optional[Config] = None,
    limit: int = 1000,
    ordering: str = "-created_at",
    search: Optional[str] = None,
    visibility: Optional[str] = None,
) -> List[Dict]:
    cfg = cfg or Config()

    endpoint = cfg.get_platform_endpoint("trainers")
    params = {"page_size": limit, "ordering": ordering}
    if visibility:
        params["visibility"] = visibility
    if search:
        params["search"] = search
    headers = {"Authorization": cfg.api_key}
    response: Dict = http.fetch(endpoint, headers=headers, params=params)
    trainers = response.get("data", [])
    return trainers


def pretty_print_trainer_packages(trainers: List[Dict[str, str]]) -> None:
    mapping = {
        "ID": "id",
        "Name": "name",
        "Description": "description",
        "Created At": "created_at",
        "Visibility": "visibility",
    }
    description_max_length = 25
    display_trainers = []
    for trainer in trainers:
        trainer = trainer.copy()
        description = trainer.get("description", None)
        if description is None:
            description = ""
        if len(description) > description_max_length:
            trainer["description"] = description[:description_max_length] + "..."

        display_trainers.append(trainer)

    pretty_print_list_as_table(
        table_title="Available Trainer Packages (most recent first)",
        dict_items=display_trainers,
        column_name_to_key_mapping=mapping,
    )
