from pathlib import Path
from typing import Dict, Optional

import boto3
from botocore.exceptions import ClientError
from rich.progress import Progress

from hafnia.http import fetch
from hafnia.log import sys_logger, user_logger
from hafnia.platform.s5cmd_utils import ResourceCredentials


def get_resource_credentials(endpoint: str, api_key: str) -> ResourceCredentials:
    """
    Retrieve credentials for accessing the recipe stored in S3 (or another resource)
    by calling a DIP endpoint with the API key.

    Args:
        endpoint (str): The endpoint URL to fetch credentials from.

    Returns:
        ResourceCredentials

    Raises:
        RuntimeError: If the call to fetch the credentials fails for any reason.
    """
    try:
        headers = {"Authorization": api_key, "accept": "application/json"}
        credentials_dict: Dict = fetch(endpoint, headers=headers)  # type: ignore[assignment]
        credentials = ResourceCredentials.fix_naming(credentials_dict)
        sys_logger.debug("Successfully retrieved credentials from DIP endpoint.")
        return credentials
    except Exception as e:
        sys_logger.error(f"Failed to fetch credentials from endpoint: {e}")
        raise RuntimeError(f"Failed to retrieve credentials: {e}") from e


def download_single_object(s3_client, bucket: str, object_key: str, output_dir: Path) -> Path:
    """
    Downloads a single object from S3 to a local path.

    Args:
        s3_client: The Boto3 S3 client.
        bucket (str): S3 bucket name.
        object_key (str): The S3 object key to download.
        output_dir (Path): The local directory in which to place the file.

    Returns:
        Path: The local path where the file was saved.
    """
    local_path = output_dir / object_key
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, object_key, local_path.as_posix())
    return local_path


def download_resource(resource_url: str, destination: str, api_key: str, prefix: Optional[str] = None) -> Dict:
    """
    Downloads either a single file from S3 or all objects under a prefix.

    Args:
        resource_url (str): The URL or identifier used to fetch S3 credentials.
        destination (str): Path to local directory where files will be stored.
        api_key (str): API key for authentication when fetching credentials.
        prefix (Optional[str]): If provided, only download objects under this prefix.

    Returns:
        Dict[str, Any]: A dictionary containing download info, e.g.:
            {
                "status": "success",
                "downloaded_files": ["/path/to/file", "/path/to/other"]
            }

    Raises:
        ValueError: If the S3 ARN is invalid or no objects found under prefix.
        RuntimeError: If S3 calls fail with an unexpected error.
    """
    res_credentials = get_resource_credentials(resource_url, api_key)

    bucket_name = res_credentials.bucket_name()
    prefix = prefix or res_credentials.object_key()

    output_path = Path(destination)
    output_path.mkdir(parents=True, exist_ok=True)
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=res_credentials.access_key,
        aws_secret_access_key=res_credentials.secret_key,
        aws_session_token=res_credentials.session_token,
    )
    downloaded_files = []
    try:
        s3_client.head_object(Bucket=bucket_name, Key=prefix)
        local_file = download_single_object(s3_client, bucket_name, prefix, output_path)
        downloaded_files.append(str(local_file))
        user_logger.info(f"Downloaded single file: {local_file}")

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "404":
            sys_logger.debug(f"Object '{prefix}' not found; trying as a prefix.")
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            contents = response.get("Contents", [])

            if not contents:
                raise ValueError(f"No objects found for prefix '{prefix}' in bucket '{bucket_name}'")

            with Progress() as progress:
                task = progress.add_task("Downloading files", total=len(contents))
                for obj in contents:
                    sub_key = obj["Key"]
                    size_mb = obj.get("Size", 0) / 1024 / 1024
                    progress.update(task, description=f"Downloading {sub_key} ({size_mb:.2f} MB)")
                    local_file = download_single_object(s3_client, bucket_name, sub_key, output_path)
                    downloaded_files.append(local_file.as_posix())
                    progress.advance(task)

            user_logger.info(f"Downloaded folder/prefix '{prefix}' with {len(downloaded_files)} object(s).")
        else:
            user_logger.error(f"Error checking object or prefix: {e}")
            raise RuntimeError(f"Failed to check or download S3 resource: {e}") from e

    return {"status": "success", "downloaded_files": downloaded_files}
