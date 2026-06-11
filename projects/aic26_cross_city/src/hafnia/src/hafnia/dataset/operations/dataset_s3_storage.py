import hashlib
import ipaddress
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import boto3
import polars as pl
from botocore.exceptions import ClientError

from hafnia.dataset.dataset_helpers import hash_file_xxhash
from hafnia.dataset.dataset_names import (
    DatasetVariant,
    SampleField,
)
from hafnia.dataset.hafnia_dataset import (
    HafniaDataset,
    download_meta_dataset_files_from_version,
)
from hafnia.log import user_logger
from hafnia.platform import s5cmd_utils
from hafnia.platform.datasets import (
    get_read_credentials_by_name,
    get_upload_credentials,
)
from hafnia.platform.s5cmd_utils import (
    AwsCredentials,
    ResourceCredentials,
    fast_copy_files,
)
from hafnia.utils import progress_bar
from hafnia_cli.config import Config


def delete_hafnia_dataset_files_on_platform(
    dataset_name: str,
    interactive: bool = True,
    cfg: Optional[Config] = None,
) -> bool:
    cfg = cfg or Config()
    resource_credentials = get_upload_credentials(dataset_name, cfg=cfg)

    if resource_credentials is None:
        raise RuntimeError("Failed to get upload credentials from the platform.")

    return delete_hafnia_dataset_files_from_resource_credentials(
        interactive=interactive,
        resource_credentials=resource_credentials,
    )


def delete_hafnia_dataset_files_from_resource_credentials(
    resource_credentials: ResourceCredentials,
    interactive: bool = True,
    remove_bucket: bool = True,
) -> bool:
    envs = resource_credentials.aws_credentials()
    bucket_name = resource_credentials.bucket_name()
    if interactive:
        confirmation = (
            input(
                f"WARNING THIS WILL delete all files stored in 's3://{bucket_name}'.\n"
                "Meaning that all previous versions of the dataset will be deleted. \n"
                "Normally this is not needed, but if you have changed the dataset structure or want to start from fresh, "
                "you can delete all files in the S3 bucket. "
                "\nDo you really want to delete all files? (yes/NO): "
            )
            .strip()
            .lower()
        )
        if confirmation != "yes":
            user_logger.info("Delete operation cancelled by the user.")
            return False
    user_logger.info(f"Deleting all files in S3 bucket '{bucket_name}'...")
    s5cmd_utils.delete_bucket_content(
        bucket_prefix=f"s3://{bucket_name}",
        remove_bucket=remove_bucket,
        append_envs=envs,
    )
    return True


_BUCKET_NAME_CHARSET = re.compile(r"^[a-z0-9.\-]+$")


def validate_bucket_name_by_s3_rules(bucket_name: str) -> None:
    """Validate ``bucket_name`` against AWS S3 general-purpose bucket naming rules.

    This is a pure string check — it cannot detect global-uniqueness conflicts
    or ownership issues, only names that S3 will reject outright. Raises
    ``ValueError`` describing the first violated rule.

    See: https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html

    # Thank you Claude!
    """
    if not isinstance(bucket_name, str):
        raise ValueError(f"Bucket name must be a string, got {type(bucket_name).__name__}.")

    n = len(bucket_name)
    if n < 3 or n > 63:
        raise ValueError(f"Bucket name must be 3-63 characters long, got {n}: {bucket_name!r}.")

    if not _BUCKET_NAME_CHARSET.match(bucket_name):
        raise ValueError(
            f"Bucket name {bucket_name!r} contains invalid characters. "
            "Only lowercase letters, numbers, dots (.) and hyphens (-) are allowed."
        )

    if not (bucket_name[0].isalnum() and bucket_name[-1].isalnum()):
        raise ValueError(f"Bucket name {bucket_name!r} must begin and end with a lowercase letter or number.")

    if ".." in bucket_name:
        raise ValueError(f"Bucket name {bucket_name!r} must not contain two adjacent periods.")

    try:
        ipaddress.IPv4Address(bucket_name)
    except ipaddress.AddressValueError:
        pass
    else:
        raise ValueError(f"Bucket name {bucket_name!r} must not be formatted as an IPv4 address.")

    forbidden_prefixes = ("xn--", "sthree-", "amzn-s3-demo-")
    for prefix in forbidden_prefixes:
        if bucket_name.startswith(prefix):
            raise ValueError(f"Bucket name {bucket_name!r} must not start with the reserved prefix {prefix!r}.")

    forbidden_suffixes = ("-s3alias", "--ol-s3", ".mrap", "--x-s3", "--table-s3")
    for suffix in forbidden_suffixes:
        if bucket_name.endswith(suffix):
            raise ValueError(f"Bucket name {bucket_name!r} must not end with the reserved suffix {suffix!r}.")


def _ensure_s3_bucket_exists(session: boto3.Session, bucket_name: str) -> None:
    """Create ``bucket_name`` if it does not already exist.

    A successful ``head_bucket`` call (200) is treated as "exists, nothing to
    do". A 404 triggers creation in the session's region. Any other error
    (e.g. 403 Forbidden — bucket exists but is owned by someone else) is
    re-raised so the caller does not silently overwrite or proceed.
    """
    validate_bucket_name_by_s3_rules(bucket_name)
    s3_client = session.client("s3")
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code not in ("404", "NoSuchBucket"):
            raise

    region = session.region_name
    create_kwargs: Dict[str, Any] = {"Bucket": bucket_name}
    # us-east-1 must NOT include LocationConstraint; every other region must.
    if region and region != "us-east-1":
        create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3_client.create_bucket(**create_kwargs)
    user_logger.info(f"Created S3 bucket '{bucket_name}' in region '{region or 'us-east-1'}'.")


def download_dataset_from_s3(
    bucket_prefix: str,
    path_local: Path,
    session: boto3.Session,
    version: Optional[str] = None,
    force_redownload: bool = False,
    download_files: bool = True,
) -> HafniaDataset:
    """Download a Hafnia dataset from a user-controlled S3 bucket.

    Downloads the metadata files for the requested ``version`` (or latest if
    ``None``), then syncs the data files referenced by the dataset to
    ``path_local``. The returned dataset has ``FILE_PATH`` columns rewritten
    to the local file paths and the annotation files on disk are updated to
    match.

    Args:
        bucket_prefix: S3 location without scheme, e.g. ``"my-bucket/sample"``.
        path_local: Local folder to download into.
        session: Boto3 session used to obtain credentials.
        version: Dataset version string (e.g. ``"0.1.0"``) or ``None`` for latest.
        force_redownload: If ``True``, re-download data files that already
            exist locally.
        download_files: If ``False``, to only download metadata files. ``True`` by default to also download
            dataset images/videos.
    """
    resource_credentials = AwsCredentials.from_session(session=session).to_resource_credentials(bucket_prefix)

    download_meta_dataset_files_from_version(
        resource_credentials=resource_credentials,
        version=version,
        path_dataset=path_local,
    )

    dataset = HafniaDataset.from_path(path_local, check_for_images=False)

    if not download_files:
        return dataset

    dataset = sync_dataset_files_from_s3(
        dataset,
        path_local,
        aws_credentials=resource_credentials,
        force_redownload=force_redownload,
    )
    dataset.write_annotations(path_folder=path_local)  # Overwrite annotations as files have been re-downloaded
    return dataset


def upload_dataset_to_s3(
    dataset: HafniaDataset,
    bucket_prefix: str,
    session: boto3.Session,
    allow_version_overwrite: bool = False,
    create_bucket_if_missing: bool = True,
    interactive: bool = True,
) -> None:
    """Upload a Hafnia dataset to a user-controlled S3 bucket.

    Args:
        dataset: Dataset to upload.
        bucket_prefix: S3 location without scheme, e.g. ``"my-bucket"`` or
            ``"my-bucket/sample"``. The leading segment is treated as the
            bucket name for the optional create step.
        session: Boto3 session used to obtain credentials and (optionally)
            create the bucket.
        allow_version_overwrite: If ``True``, overwrite metadata files for an
            existing version in S3. Otherwise the upload aborts when a
            version collision is detected.
        create_bucket_if_missing: If ``True`` and the target bucket does not
            exist, create it in the session's region before uploading.
        interactive: If ``True``, prompt for confirmation before uploading.
    """
    resource_credentials = AwsCredentials.from_session(session=session).to_resource_credentials(bucket_prefix)
    envs = resource_credentials.aws_credentials()

    if create_bucket_if_missing:
        _ensure_s3_bucket_exists(session=session, bucket_name=resource_credentials.bucket_name())

    sync_dataset_files_to_s3(
        dataset=dataset,
        bucket_prefix=resource_credentials.s3_path(),
        interactive=interactive,
        allow_version_overwrite=allow_version_overwrite,
        envs=envs,
    )


def sync_dataset_files_to_s3(
    dataset: HafniaDataset,
    bucket_prefix: str,
    allow_version_overwrite: bool = False,
    interactive: bool = True,
    envs: Optional[Dict[str, str]] = None,
) -> None:
    t0 = time.time()
    # bucket_prefix e.g. 'bucket-name/sample' (no scheme)
    s3_uri_prefix = f"s3://{bucket_prefix}"
    remote_paths = []
    for file_str in progress_bar(dataset.samples[SampleField.FILE_PATH], description="Hashing data files"):
        path_file = Path(file_str)
        file_hash = hash_file_xxhash(path_file)

        # Relative path in S3 bucket e.g. 'data/e2/b0/e2b000ac47b19a999bee5456a6addb88.png'
        relative_path = s3_prefix_from_hash(hash=file_hash, suffix=path_file.suffix)

        # Remote path in S3 bucket e.g. 's3://bucket-name/sample/data/e2/b0/e2b000ac47b19a999bee5456a6addb88.png'
        remote_path = f"{s3_uri_prefix}/{relative_path}"
        remote_paths.append(remote_path)

    dataset.samples = dataset.samples.with_columns(pl.Series(remote_paths).alias(SampleField.REMOTE_PATH))

    user_logger.info(f"Syncing dataset to S3 bucket '{s3_uri_prefix}'")
    files_in_s3 = set(s5cmd_utils.list_bucket(bucket_prefix=s3_uri_prefix, append_envs=envs))

    # Discover data files (images, videos, etc.) missing in s3
    data_files_missing = dataset.samples.filter(~pl.col(SampleField.REMOTE_PATH).is_in(files_in_s3))
    files_already_in_s3 = dataset.samples.filter(pl.col(SampleField.REMOTE_PATH).is_in(files_in_s3))

    with tempfile.TemporaryDirectory() as temp_dir:  # Temp folder to store metadata files
        path_temp = Path(temp_dir)
        # File paths are dropped when uploading to S3
        dataset = dataset.update_samples(dataset.samples.drop(SampleField.FILE_PATH))
        dataset.write_annotations(path_temp)

        # Discover versioned metadata files (e.g. "annotations.jsonl", "dataset_info.json") missing in s3
        metadata_files_local = []
        metadata_files_s3 = []
        for filename in path_temp.iterdir():
            metadata_files_s3.append(f"{s3_uri_prefix}/versions/{dataset.info.version}/{filename.name}")
            metadata_files_local.append(filename.as_posix())

        overwrite_metadata_files = files_in_s3.intersection(set(metadata_files_s3))
        will_overwrite_metadata_files = len(overwrite_metadata_files) > 0

        n_files_already_in_s3 = len(files_already_in_s3)
        user_logger.info(f"Sync dataset to {s3_uri_prefix}")
        user_logger.info(
            f"- Found that {n_files_already_in_s3} / {len(dataset.samples)} data files already exist. "
            f"Meaning {len(data_files_missing)} data files will be uploaded. \n"
            f"- Will upload {len(metadata_files_local)} metadata files. \n"
            f"- Total files to upload: {len(data_files_missing) + len(metadata_files_local)}"
        )
        if will_overwrite_metadata_files:
            msg = f"Metadata files for dataset version '{dataset.info.version}' already exist"
            if allow_version_overwrite:
                user_logger.warning(
                    f"- WARNING: {msg}. Version will be overwritten as 'allow_version_overwrite=True' is set."
                )
            else:
                raise ValueError(
                    f"Upload cancelled. {msg}. \nTo overwrite existing metadata files, "
                    "you will need to set 'allow_version_overwrite=True' explicitly."
                )

        has_missing_files = len(data_files_missing) > 0
        if interactive and (has_missing_files or will_overwrite_metadata_files):
            print("Please type 'yes' to upload files.")
            confirmation = input("Do you want to continue? (yes/NO): ").strip().lower()

            if confirmation != "yes":
                raise RuntimeError("Upload cancelled by user.")

        local_paths = metadata_files_local + data_files_missing[SampleField.FILE_PATH].to_list()
        s3_paths = metadata_files_s3 + data_files_missing[SampleField.REMOTE_PATH].to_list()
        s5cmd_utils.fast_copy_files(local_paths, s3_paths, append_envs=envs, description="Uploading files")
    user_logger.info(f"- Synced dataset in {time.time() - t0:.2f} seconds.")


def sync_dataset_files_to_platform(
    dataset: HafniaDataset,
    sample_dataset: Optional[HafniaDataset] = None,
    interactive: bool = True,
    allow_version_overwrite: bool = False,
    cfg: Optional[Config] = None,
) -> None:
    cfg = cfg or Config()
    resource_credentials = get_upload_credentials(dataset.info.dataset_name, cfg=cfg)

    if resource_credentials is None:
        raise RuntimeError("Failed to get upload credentials from the platform.")

    sync_dataset_files_to_platform_from_resource_credentials(
        dataset=dataset,
        sample_dataset=sample_dataset,
        interactive=interactive,
        allow_version_overwrite=allow_version_overwrite,
        resource_credentials=resource_credentials,
    )


def sync_dataset_files_to_platform_from_resource_credentials(
    dataset: HafniaDataset,
    sample_dataset: Optional[HafniaDataset],
    interactive: bool,
    allow_version_overwrite: bool,
    resource_credentials: ResourceCredentials,
):
    envs = resource_credentials.aws_credentials()
    bucket_name = resource_credentials.bucket_name()

    for dataset_variant_type in [DatasetVariant.SAMPLE, DatasetVariant.HIDDEN]:
        if dataset_variant_type == DatasetVariant.SAMPLE:
            if sample_dataset is None:
                dataset_variant = dataset.create_sample_dataset()
            else:
                dataset_variant = sample_dataset
        else:
            dataset_variant = dataset

        dataset_variant.samples = dataset_variant.samples.with_columns(
            pl.lit(dataset.info.dataset_name).alias(SampleField.DATASET_NAME)
        )

        sync_dataset_files_to_s3(
            dataset=dataset_variant,
            bucket_prefix=f"{bucket_name}/{dataset_variant_type.value}",
            interactive=interactive,
            allow_version_overwrite=allow_version_overwrite,
            envs=envs,
        )


MISSING_LOCALLY_COLUMN = "MissingLocally"


MAX_PATH_LENGTH = 255


def local_path_from_remote_path(
    remote_src_path: str,
    path_data: Path,
    extend_name_by_subfolder: Optional[int],
) -> str:
    """Build the local download path for a remote S3 source path.

    The remote path is flattened into a single filename under ``path_data`` by
    joining its parts with ``-``. If the resulting absolute path would exceed
    ``MAX_PATH_LENGTH``, the filename is replaced with a short sha1-derived
    hash of itself (keeping the original suffix) so the path fits on common
    filesystems.
    """
    if extend_name_by_subfolder is not None and extend_name_by_subfolder < 0:
        raise ValueError(f"'extend_name_by_subfolder' must be a non-negative integer. Got {extend_name_by_subfolder}.")

    path_parts_all = remote_src_path.split("/")
    if extend_name_by_subfolder is None:
        path_parts = path_parts_all[2:]  # Drops "s3://" prefix
    else:
        path_parts = path_parts_all[-(extend_name_by_subfolder + 1) :]

    filename = "-".join(path_parts)
    local_path_str = (path_data / filename).absolute().as_posix()
    if len(local_path_str) <= MAX_PATH_LENGTH:
        return local_path_str

    suffix = Path(filename).suffix
    hash_part = hashlib.sha1(filename.encode()).hexdigest()
    shortened_path_str = (path_data / hash_part).with_suffix(suffix).absolute().as_posix()

    user_logger.debug(
        f"Generated filename '{filename}' exceeds MAX_PATH_LENGTH={MAX_PATH_LENGTH}. "
        f"Shortened to '{shortened_path_str}'."
    )
    if len(shortened_path_str) > MAX_PATH_LENGTH:
        raise ValueError(
            f"Even after shortening, local path '{shortened_path_str}' exceeds MAX_PATH_LENGTH={MAX_PATH_LENGTH}. "
            "Consider storing the dataset in a folder with a shorter path."
        )
    return shortened_path_str


def _resolve_local_download_paths(
    dataset: HafniaDataset,
    path_output_folder: Path,
    extend_name_by_subfolder: Optional[int],
    subfolder_name: str,
) -> Tuple[HafniaDataset, pl.DataFrame]:
    path_data = path_output_folder / subfolder_name
    path_data.mkdir(parents=True, exist_ok=True)

    update_rows = []
    for remote_src_path in dataset.samples[SampleField.REMOTE_PATH].unique().to_list():
        local_path_str = local_path_from_remote_path(
            remote_src_path=remote_src_path,
            path_data=path_data,
            extend_name_by_subfolder=extend_name_by_subfolder,
        )
        update_rows.append(
            {
                SampleField.REMOTE_PATH: remote_src_path,
                SampleField.FILE_PATH: local_path_str,
                MISSING_LOCALLY_COLUMN: not Path(local_path_str).exists(),
            }
        )
    download_df = pl.DataFrame(update_rows)

    samples = dataset.samples.update(
        download_df.select([SampleField.REMOTE_PATH, SampleField.FILE_PATH]),
        on=[SampleField.REMOTE_PATH],
    )
    dataset = dataset.update_samples(samples)
    return dataset, download_df


def sync_files_from_platform(
    dataset: HafniaDataset,
    path_output_folder: Path,
    force_redownload: bool = False,
    extend_name_by_subfolder: Optional[int] = None,
    subfolder_name: str = "data",
    cfg: Optional[Config] = None,
) -> HafniaDataset:
    """Download data files referenced by ``dataset`` from the Hafnia platform.

    Resolves a local destination under ``path_output_folder / subfolder_name``,
    fetches read credentials per dataset name, and copies the files via s5cmd.
    The returned dataset has ``FILE_PATH`` rewritten to the local locations.

    Args:
        dataset: Dataset whose ``REMOTE_PATH`` column points at platform-managed S3 objects.
        path_output_folder: Local folder to download into.
        force_redownload: If ``True``, re-download files that already exist locally.
        extend_name_by_subfolder: Number of trailing remote path segments to fold
            into the local filename (useful when remote paths share a basename).
            Default ``None`` uses the full remote path minus the leading "s3://" prefix
            and replaces "/" with "-" to create the local filename.
            E.g for remote path "s3://bucket-name/sample/data/e2/b0/e2b000ac47b19a999bee5456a6addb88.png"
            -> filename "data-e2-b0-e2b000ac47b19a999bee5456a6addb88.png" with ``extend_name_by_subfolder=None``
        subfolder_name: Subfolder under ``path_output_folder`` to write data into.
        cfg: Optional Hafnia CLI config; defaults to a fresh ``Config()``.
    """
    dataset, download_df = _resolve_local_download_paths(
        dataset=dataset,
        path_output_folder=path_output_folder,
        extend_name_by_subfolder=extend_name_by_subfolder,
        subfolder_name=subfolder_name,
    )

    # Extend 'download_df' with dataset name - but keep only unique remote paths to avoid duplicate downloads
    download_samples = download_df.join(
        dataset.samples.select([SampleField.REMOTE_PATH, SampleField.DATASET_NAME]),
        on=SampleField.REMOTE_PATH,
        how="left",
    ).unique(subset=SampleField.REMOTE_PATH, keep="first")
    missing_samples = download_samples.filter(pl.col(MISSING_LOCALLY_COLUMN))
    existing_samples = download_samples.filter(pl.col(MISSING_LOCALLY_COLUMN) == False)  # noqa: E712

    if not force_redownload and len(missing_samples) == 0:
        user_logger.info(
            "All files already exist locally. Skipping download. Set 'force_redownload=True' to re-download."
        )
        return dataset

    if force_redownload:
        missing_samples = download_samples
    else:
        if len(existing_samples) > 0:
            user_logger.info(
                f"Found {len(existing_samples)}/{len(download_samples)} files already exists. "
                f"Downloading {len(missing_samples)} files."
            )

    cfg = cfg or Config()
    for (dataset_name,), dataset_group in missing_samples.group_by(SampleField.DATASET_NAME):
        if dataset_name is None:
            user_logger.warning("Dataset name is missing in the samples. Cannot download files without dataset name.")
            continue
        user_logger.info(f"Downloading files for dataset '{dataset_name}'...")
        remote_src_paths = dataset_group[SampleField.REMOTE_PATH].to_list()
        local_dst_paths = dataset_group[SampleField.FILE_PATH].to_list()
        resource_credentials: ResourceCredentials = get_read_credentials_by_name(dataset_name=dataset_name, cfg=cfg)
        environment_vars = resource_credentials.aws_credentials()
        fast_copy_files(
            src_paths=remote_src_paths,
            dst_paths=local_dst_paths,
            append_envs=environment_vars,
            description="Downloading images",
        )
    return dataset


def sync_dataset_files_from_s3(
    dataset: HafniaDataset,
    path_output_folder: Path,
    aws_credentials: AwsCredentials,
    force_redownload: bool = False,
    extend_name_by_subfolder: Optional[int] = None,
    subfolder_name: str = "data",
) -> HafniaDataset:
    dataset, download_df = _resolve_local_download_paths(
        dataset=dataset,
        path_output_folder=path_output_folder,
        extend_name_by_subfolder=extend_name_by_subfolder,
        subfolder_name=subfolder_name,
    )

    if force_redownload:
        download_samples = download_df
    else:
        download_samples = download_df.filter(pl.col(MISSING_LOCALLY_COLUMN))
        skip_files = len(download_df) - len(download_samples)
        if skip_files > 0:
            user_logger.info(
                f"Found {skip_files}/{len(download_df)} files already exists. "
                f"Downloading {len(download_samples)} files."
            )

    if len(download_samples) == 0:
        user_logger.info(
            "All files already exist locally. Skipping download. Set 'force_redownload=True' to re-download."
        )
        return dataset

    environment_vars = aws_credentials.aws_credentials()
    fast_copy_files(
        src_paths=download_samples[SampleField.REMOTE_PATH].to_list(),
        dst_paths=download_samples[SampleField.FILE_PATH].to_list(),
        append_envs=environment_vars,
        description="Downloading images",
    )

    return dataset


def s3_prefix_from_hash(hash: str, suffix: str) -> str:
    """
    Generate a relative S3 path from a hash value for objects stored in S3.

    This function deliberately uses a hierarchical directory layout based on the
    hash prefix to avoid putting too many objects in a single S3 prefix, which
    can run into AWS S3 rate limits and performance issues. For example, for
    hash "dfe8f3b1c2a4f5b6c7d8e9f0a1b2c3d4" and suffix ".png", the returned
    path will be:

        "data/df/e8/dfe8f3b1c2a4f5b6c7d8e9f0a1b2c3d4.png"

    Note: This intentionally differs from when images are stored to disk locally, where
    a flat path of the form ``data/<hash><suffix>`` is used.
    """
    s3_prefix = f"data/{hash[:2]}/{hash[2:4]}/{hash}{suffix}"
    return s3_prefix
