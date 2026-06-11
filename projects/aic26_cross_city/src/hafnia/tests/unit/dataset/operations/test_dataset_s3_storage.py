from pathlib import Path

import pytest

from hafnia.dataset.operations.dataset_s3_storage import (
    MAX_PATH_LENGTH,
    local_path_from_remote_path,
    validate_bucket_name_by_s3_rules,
)


@pytest.mark.parametrize(
    "bucket_name",
    [
        "abc",
        "my-bucket",
        "my.bucket.name",
        "bucket123",
        "a1b2c3",
        "my-very-long-but-still-valid-bucket-name-with-63-characters-abc",  # 63 chars
        "1bucket",  # may start with a digit
        "bucket1",
        "a-b",
        "192-168-5-4",  # IP-like but not an IP
    ],
)
def test_validate_bucket_name_by_s3_rules_valid(bucket_name: str):
    validate_bucket_name_by_s3_rules(bucket_name)


@pytest.mark.parametrize(
    ("bucket_name", "reason"),
    [
        ("ab", "too short"),
        ("a" * 64, "too long"),
        ("", "empty string"),
        ("My-Bucket", "uppercase letters"),
        ("my_bucket", "underscore not allowed"),
        ("my bucket", "space not allowed"),
        ("my/bucket", "slash not allowed"),
        ("bucket!", "exclamation not allowed"),
        ("-bucket", "starts with hyphen"),
        ("bucket-", "ends with hyphen"),
        (".bucket", "starts with period"),
        ("bucket.", "ends with period"),
        ("my..bucket", "adjacent periods"),
        ("192.168.5.4", "IPv4 address"),
        ("10.0.0.1", "IPv4 address"),
        ("xn--bucket", "reserved xn-- prefix"),
        ("sthree-bucket", "reserved sthree- prefix"),
        ("amzn-s3-demo-bucket", "reserved amzn-s3-demo- prefix"),
        ("bucket-s3alias", "reserved -s3alias suffix"),
        ("bucket--ol-s3", "reserved --ol-s3 suffix"),
        ("bucket.mrap", "reserved .mrap suffix"),
        ("bucket--x-s3", "reserved --x-s3 suffix"),
        ("bucket--table-s3", "reserved --table-s3 suffix"),
    ],
)
def test_validate_bucket_name_by_s3_rules_invalid(bucket_name: str, reason: str):
    with pytest.raises(ValueError):
        validate_bucket_name_by_s3_rules(bucket_name)


def test_validate_bucket_name_by_s3_rules_non_string():
    with pytest.raises(ValueError):
        validate_bucket_name_by_s3_rules(12345)  # type: ignore[arg-type]


LONG_NAME1 = "s3://mdi-pre-labelling/detection/eulm/outputs/goa_pedestrian_random_test_20260212_233106/genoa-1-split/5min/lan0281_-_via_avio_portici/2025/08/09/12/2025-08-09T121055+0100_2025-08-09T124055+0100_redacted_seg003/2025-08-09T121055+0100_2025-08-09T124055+0100_redacted_seg003/000000.jpg"
LONG_NAME2 = "s3://mdi-pre-labelling.s3.eu-west-1.amazonaws.com/detection/eulm/outputs/goa_pedestrian_random_test_20260212_233106/genoa-1-split/5min/lan0281_-_via_avio_portici/2025/08/09/12/2025-08-09T121055+0100_2025-08-09T124055+0100_redacted_seg003/2025-08-09T121055+0100_2025-08-09T124055+0100_redacted_seg003/000000.jpg"


@pytest.mark.parametrize(
    ("remote_src_path", "extend_name_by_subfolder", "expected_filename"),
    [
        # extend=None drops the "s3://" prefix and joins the remaining parts with "-".
        ("s3://my-bucket/folder/image.jpg", None, "my-bucket-folder-image.jpg"),
        ("s3://b/short.png", None, "b-short.png"),
        # extend_name_by_subfolder keeps the basename plus N trailing segments.
        ("s3://bucket/a/b/image.jpg", 0, "image.jpg"),
        ("s3://bucket/a/b/image.jpg", 1, "b-image.jpg"),
        ("s3://bucket/a/b/image.jpg", 2, "a-b-image.jpg"),
        ("s3://bucket/a/b/image.jpg", 3, "bucket-a-b-image.jpg"),
        # Long path → sha1-shortened filename (suffix preserved).
        (LONG_NAME1, None, "786aa96d0b6cc7ece649e4ee8f9dcebb989a33e7.jpg"),
        (LONG_NAME2, None, "9068e98d6a9d578fa4e8ae2aaa05376bec5905cd.jpg"),
        (f"s3://bucket1/{'y' * 300}/image.jpg", None, "e111fa29e9cf9422d6cdc3c91cb4a2284a43ed36.jpg"),
        (f"s3://bucket/{'x' * 300}/image.jpg", None, "5f41c2190123fc1d9c7c5467888ad41c34853fa1.jpg"),
        (f"s3://bucket2/{'y' * 300}/image.jpg", None, "eaf6adc827c7bbdd4405cad68a779e9d42b74431.jpg"),
        # Long path with no suffix → hash-only filename, no suffix.
        (f"s3://bucket/{'z' * 300}/no_extension_file", None, "43476be47179ca9b063282ac1ec55c2ea59b55fd"),
    ],
)
def test_local_path_from_remote_path(
    tmp_path: Path,
    remote_src_path: str,
    extend_name_by_subfolder: int | None,
    expected_filename: str,
):
    result = local_path_from_remote_path(
        remote_src_path=remote_src_path,
        path_data=tmp_path,
        extend_name_by_subfolder=extend_name_by_subfolder,
    )
    assert result == (tmp_path / expected_filename).absolute().as_posix()
    assert len(result) <= MAX_PATH_LENGTH


def test_local_path_from_remote_path_raises_conditions(tmp_path: Path):
    # 1) Build a path_data so long that even hash + suffix won't fit under MAX_PATH_LENGTH.
    deep = tmp_path
    while len(deep.absolute().as_posix()) < MAX_PATH_LENGTH:
        deep = deep / ("d" * 20)
    deep.mkdir(parents=True, exist_ok=True)

    long_remote = f"s3://bucket/{'q' * 300}/image.jpg"
    with pytest.raises(ValueError, match="exceeds MAX_PATH_LENGTH"):
        local_path_from_remote_path(long_remote, deep, extend_name_by_subfolder=None)

    # 2) Negative extend_name_by_subfolder should raise ValueError.
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        local_path_from_remote_path(
            remote_src_path="s3://bucket/a/b/image.jpg",
            path_data=tmp_path,
            extend_name_by_subfolder=-1,
        )
