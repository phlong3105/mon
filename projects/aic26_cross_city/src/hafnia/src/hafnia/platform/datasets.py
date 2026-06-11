from typing import Any, Dict, List, Optional

import rich
from rich import print as rprint

from hafnia import http, utils
from hafnia.http import fetch, post
from hafnia.log import user_logger
from hafnia.platform.download import get_resource_credentials
from hafnia.platform.s5cmd_utils import ResourceCredentials
from hafnia.utils import timed
from hafnia_cli.config import Config


@timed("Fetching dataset by name.")
def get_dataset_by_name(dataset_name: str, cfg: Optional[Config] = None) -> Optional[Dict[str, Any]]:
    """Get dataset details by name from the Hafnia platform."""
    cfg = cfg or Config()
    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    header = {"Authorization": cfg.api_key}
    params = {"name__iexact": dataset_name}
    response: Dict = http.fetch(endpoint_dataset, headers=header, params=params)
    dataset_count = response.get("count", 0)
    if dataset_count == 0:
        return None

    if dataset_count > 1:
        raise ValueError(f"Multiple datasets found with the name '{dataset_name}'.")

    dataset = response["data"][0]
    return dataset


@timed("Fetching dataset by ID.")
def get_dataset_by_id(dataset_id: str, cfg: Optional[Config] = None) -> Optional[Dict[str, Any]]:
    """Get dataset details by ID from the Hafnia platform."""
    cfg = cfg or Config()
    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    header = {"Authorization": cfg.api_key}
    full_url = f"{endpoint_dataset}/{dataset_id}"
    dataset: Dict[str, Any] = http.fetch(full_url, headers=header)  # type: ignore[assignment]
    if not dataset:
        return None

    return dataset


def get_or_create_dataset(dataset_name: str = "", cfg: Optional[Config] = None) -> Dict[str, Any]:
    """Create a new dataset on the Hafnia platform."""
    cfg = cfg or Config()
    dataset = get_dataset_by_name(dataset_name, cfg)
    if dataset is not None:
        user_logger.info(f"Dataset '{dataset_name}' already exists on the Hafnia platform.")
        return dataset

    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    header = {"Authorization": cfg.api_key}
    dataset_title = dataset_name.replace("-", " ").title()  # convert dataset-name to title "Dataset Name"
    payload = {
        "title": dataset_title,
        "name": dataset_name,
        "overview": "No description provided.",
    }

    dataset = http.post(endpoint_dataset, headers=header, data=payload)  # type: ignore[assignment]

    # TODO: Handle issue when dataset creation fails because name is taken by another user from a different organization
    if not dataset:
        raise ValueError("Failed to create dataset on the Hafnia platform. ")

    return dataset


@timed("Fetching dataset list.")
def get_datasets(
    cfg: Optional[Config] = None,
    ordering: str = "name",
    limit: int = 1000,
    search: Optional[str] = None,
    visibility: Optional[str] = None,
) -> List[Dict[str, str]]:
    """List available datasets on the Hafnia platform."""
    cfg = cfg or Config()
    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    params = {"page_size": limit, "ordering": ordering}
    if search:
        params["search"] = search

    if visibility:
        params["visibility"] = visibility

    header = {"Authorization": cfg.api_key}
    response: Dict = fetch(endpoint_dataset, headers=header, params=params)
    datasets = response.get("data", [])
    if len(datasets) == 0:
        raise ValueError("No datasets found on the Hafnia platform.")
    return datasets


@timed("Fetching dataset info.")
def get_dataset_id(dataset_name: str, cfg: Optional[Config] = None) -> str:
    """Get dataset ID by name from the Hafnia platform."""
    cfg = cfg or Config()
    endpoint = cfg.get_platform_endpoint("datasets")
    headers = {"Authorization": cfg.api_key}
    params = {"name__iexact": dataset_name}
    response: Dict = http.fetch(endpoint, headers=headers, params=params)
    dataset_responses = response.get("data", [])
    if len(dataset_responses) == 0:
        raise ValueError(f"Dataset '{dataset_name}' was not found in the dataset library.")

    if len(dataset_responses) > 1:
        raise ValueError(f"Multiple datasets found with the name '{dataset_name}'.")

    dataset = dataset_responses[0]
    dataset_id = dataset.get("id", None)
    if dataset_id is None:
        raise ValueError(f"Dataset ID not found for dataset '{dataset_name}'.")
    return dataset_id


@timed("Get upload access credentials")
def get_upload_credentials(dataset_name: str, cfg: Optional[Config] = None) -> Optional[ResourceCredentials]:
    """Get dataset details by name from the Hafnia platform."""
    cfg = cfg or Config()
    dataset_response = get_dataset_by_name(dataset_name=dataset_name, cfg=cfg)
    if dataset_response is None:
        return None

    return get_upload_credentials_by_id(dataset_response["id"], cfg=cfg)


@timed("Get upload access credentials by ID")
def get_upload_credentials_by_id(dataset_id: str, cfg: Optional[Config] = None) -> Optional[ResourceCredentials]:
    """Get dataset details by ID from the Hafnia platform."""
    cfg = cfg or Config()

    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    header = {"Authorization": cfg.api_key}
    full_url = f"{endpoint_dataset}/{dataset_id}/temporary-credentials-upload"
    credentials_response: Dict = http.fetch(full_url, headers=header)  # type: ignore[assignment]

    return ResourceCredentials.fix_naming(credentials_response)


@timed("Get read access credentials by ID")
def get_read_credentials_by_id(dataset_id: str, cfg: Optional[Config] = None) -> Optional[ResourceCredentials]:
    """Get dataset read access credentials by ID from the Hafnia platform."""
    cfg = cfg or Config()
    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    if utils.is_hafnia_cloud_job():
        credentials_endpoint_suffix = "temporary-credentials-hidden"  # Access to hidden datasets
    else:
        credentials_endpoint_suffix = "temporary-credentials"  # Access to sample dataset
    access_dataset_endpoint = f"{endpoint_dataset}/{dataset_id}/{credentials_endpoint_suffix}"
    resource_credentials = get_resource_credentials(access_dataset_endpoint, cfg.api_key)
    return resource_credentials


@timed("Get read access credentials by name")
def get_read_credentials_by_name(dataset_name: str, cfg: Optional[Config] = None) -> Optional[ResourceCredentials]:
    """Get dataset read access credentials by name from the Hafnia platform."""
    cfg = cfg or Config()
    dataset_response = get_dataset_by_name(dataset_name=dataset_name, cfg=cfg)
    if dataset_response is None:
        return None

    return get_read_credentials_by_id(dataset_response["id"], cfg=cfg)


@timed("Delete dataset by id")
def delete_dataset_by_id(dataset_id: str, cfg: Optional[Config] = None) -> Dict:
    cfg = cfg or Config()
    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    header = {"Authorization": cfg.api_key}
    full_url = f"{endpoint_dataset}/{dataset_id}"
    return http.delete(full_url, headers=header)  # type: ignore


@timed("Delete dataset by name")
def delete_dataset_by_name(dataset_name: str, cfg: Optional[Config] = None) -> Dict:
    cfg = cfg or Config()
    dataset_response = get_dataset_by_name(dataset_name=dataset_name, cfg=cfg)
    if dataset_response is None:
        raise ValueError(f"Dataset '{dataset_name}' not found on the Hafnia platform.")

    dataset_id = dataset_response["id"]  # type: ignore[union-attr]
    response = delete_dataset_by_id(dataset_id=dataset_id, cfg=cfg)
    user_logger.info(f"Dataset '{dataset_name}' has been deleted from the Hafnia platform.")
    return response


def delete_dataset_completely_by_name(
    dataset_name: str,
    interactive: bool = True,
    cfg: Optional[Config] = None,
) -> None:
    from hafnia.dataset.operations.dataset_s3_storage import delete_hafnia_dataset_files_on_platform

    cfg = cfg or Config()

    is_deleted = delete_hafnia_dataset_files_on_platform(
        dataset_name=dataset_name,
        interactive=interactive,
        cfg=cfg,
    )
    if not is_deleted:
        return
    delete_dataset_by_name(dataset_name, cfg=cfg)


@timed("Import dataset details to platform")
def upload_dataset_details(data: dict, dataset_name: str, cfg: Optional[Config] = None) -> dict:
    cfg = cfg or Config()
    dataset_endpoint = cfg.get_platform_endpoint("datasets")
    dataset_id = get_dataset_id(dataset_name, cfg=cfg)

    import_endpoint = f"{dataset_endpoint}/{dataset_id}/import"
    headers = {"Authorization": cfg.api_key}

    user_logger.info("Exporting dataset details to platform. This may take up to 30 seconds...")
    response = post(endpoint=import_endpoint, headers=headers, data=data)  # type: ignore[assignment]
    return response  # type: ignore[return-value]


TABLE_FIELDS = {
    "ID": "id",
    "Hidden\nSamples": "hidden.samples",
    "Hidden\nSize": "hidden.size",
    "Sample\nSamples": "sample.samples",
    "Sample\nSize": "sample.size",
    "Name": "name",
    "Title": "title",
}


def pretty_print_datasets(datasets: List[Dict[str, str]]) -> None:
    datasets = extend_dataset_details(datasets)

    table = rich.table.Table(title="Available Datasets")
    for i_dataset, dataset in enumerate(datasets):
        if i_dataset == 0:
            for column_name, _ in TABLE_FIELDS.items():
                table.add_column(column_name, justify="left", style="cyan", no_wrap=True)
        row = [str(dataset.get(field, "")) for field in TABLE_FIELDS.values()]
        table.add_row(*row)

    rprint(table)


def extend_dataset_details(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extends dataset details with number of samples and size"""
    for dataset in datasets:
        for variant in dataset["dataset_variants"]:
            variant_type = variant["variant_type"]
            dataset[f"{variant_type}.samples"] = variant["number_of_data_items"]
            dataset[f"{variant_type}.size"] = utils.size_human_readable(variant["size_bytes"])
    return datasets
