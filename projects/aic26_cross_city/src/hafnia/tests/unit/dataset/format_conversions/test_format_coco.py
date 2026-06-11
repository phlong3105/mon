import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict

import polars as pl
import pytest
from pycocotools import mask as coco_utils

from hafnia.dataset.dataset_names import SampleField
from hafnia.dataset.format_conversions import format_coco
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.dataset.primitives import Bbox, Bitmask
from tests import helper_testing


def test_from_coco_format_visualized(compare_to_expected_image: Callable) -> None:
    path_coco_dataset = helper_testing.get_path_test_dataset_formats() / "format_coco_roboflow"
    hafnia_dataset = format_coco.from_coco_format(
        path_dataset=path_coco_dataset,
        max_samples=None,
        coco_format_type="roboflow",
    )

    sample_name = "000000000632.jpg"
    samples = hafnia_dataset.samples.filter(pl.col(SampleField.FILE_PATH).str.contains(sample_name))
    assert len(samples) == 1, f"Expected to find one sample with name {sample_name}"
    sample_dict = samples.row(0, named=True)
    sample = Sample(**sample_dict)

    sample_visualized = sample.draw_annotations()
    compare_to_expected_image(sample_visualized)


def test_to_coco_format_visualized(compare_to_expected_image: Callable, tmp_path: Path) -> None:
    path_coco_dataset = helper_testing.get_path_test_dataset_formats() / "format_coco_roboflow"
    max_samples = None
    coco_format_type = "roboflow"

    hafnia_dataset = format_coco.from_coco_format(
        path_dataset=path_coco_dataset,
        max_samples=max_samples,
        coco_format_type=coco_format_type,
    )

    path_exported_coco_dataset = tmp_path / "format_coco_roboflow_exported"
    format_coco.to_coco_format(hafnia_dataset, path_exported_coco_dataset)

    hafnia_dataset_reloaded = format_coco.from_coco_format(
        path_dataset=path_exported_coco_dataset,
        max_samples=max_samples,
        coco_format_type=coco_format_type,
    )

    # Still exists after conversion and are normalized
    bbox_table = hafnia_dataset_reloaded.create_primitive_table(Bbox)
    assert bbox_table["area"].dtype != pl.Null, "Expected area column to have no null values"
    assert bbox_table["area"].min() > 0, "Expected area values to be greater than 0"
    assert bbox_table["area"].max() < 1.0, "Expected area values to be normalized (<1.0)"

    bitmask_table = hafnia_dataset_reloaded.create_primitive_table(Bitmask)
    assert bitmask_table["area"].dtype != pl.Null, "Expected area column to have no null values"
    assert bitmask_table["area"].min() > 0, "Expected area values to be greater than 0"
    assert bitmask_table["area"].max() < 1.0, "Expected area values to be normalized (<1.0)"

    samples = hafnia_dataset_reloaded.samples.filter(pl.col(SampleField.FILE_PATH).str.contains("000000000632.jpg"))
    assert len(samples) == 1
    sample_dict = samples.row(0, named=True)
    sample = Sample(**sample_dict)

    sample_visualized = sample.draw_annotations()
    compare_to_expected_image(sample_visualized)


@pytest.mark.parametrize("micro_dataset_name", helper_testing.MICRO_DATASETS)
def test_to_and_from_coco_format(micro_dataset_name: str, tmp_path: Path) -> None:
    dataset = helper_testing.get_micro_hafnia_dataset(dataset_name=micro_dataset_name)
    n_expected_samples = len(dataset.samples)
    path_output = tmp_path / micro_dataset_name

    # To COCO format
    dataset.to_coco_format(path_output=path_output)

    # From COCO format
    dataset_reloaded = HafniaDataset.from_coco_format(path_dataset=path_output, dataset_name=micro_dataset_name)
    assert len(dataset_reloaded.samples) == n_expected_samples, (
        "The number of samples before and after COCO format conversion should be the same"
    )


@pytest.mark.parametrize(("bitmask_type"), ["polygon", "rle_as_ints", "rle_compressed_str", "rle_compressed_bytes"])
def test_convert_segmentation_to_rle_list(bitmask_type: str, compare_to_expected_image: Callable) -> None:
    segmentations_types = get_rle_bitmask_encoding_examples()
    seg_obj, sizes = segmentations_types[bitmask_type]

    rle_list = format_coco.convert_segmentation_to_rle_list(seg_obj, height=sizes[0], width=sizes[1])
    assert isinstance(rle_list, list)
    for rle in rle_list:
        assert "counts" in rle
        assert "size" in rle

    mask = coco_utils.decode(rle_list).squeeze() > 0

    compare_to_expected_image(mask)


@lru_cache()
def get_rle_bitmask_encoding_examples() -> Dict[str, Any]:
    path_coco_dataset = helper_testing.get_path_test_dataset_formats() / "format_coco_roboflow"
    image_name = "000000000632.jpg"
    hafnia_dataset = format_coco.from_coco_format(path_dataset=path_coco_dataset, coco_format_type="roboflow")
    samples = hafnia_dataset.samples.filter(pl.col(SampleField.FILE_PATH).str.contains(image_name))
    assert len(samples) == 1

    sample = Sample(**samples.row(0, named=True))

    assert sample.bitmasks is not None and len(sample.bitmasks) > 0, (
        "Sample should have at least one Bitmask annotation"
    )
    bitmask = sample.bitmasks[-1]

    path_json = path_coco_dataset / "train" / "_annotations.coco.json"

    annotations = json.loads(path_json.read_text())
    images = [img for img in annotations["images"] if img["file_name"] == image_name]
    assert len(images) == 1
    image_info = images[0]
    img_annotations = [ann for ann in annotations["annotations"] if ann["image_id"] == image_info["id"]]

    polygon_segments = img_annotations[1]["segmentation"]
    assert isinstance(polygon_segments, list) and isinstance(polygon_segments[0], list), (
        "Expected polygon segmentation to be a list of lists"
    )

    rle_as_int_counts = img_annotations[-1]["segmentation"]
    assert isinstance(rle_as_int_counts, dict) and isinstance(rle_as_int_counts["counts"], list), (
        "Expected RLE segmentation to be a dict with 'counts' as a list of ints"
    )
    sizes = rle_as_int_counts["size"]
    rle_types = {
        "rle_compressed_bytes": (
            bitmask.to_coco_rle(img_height=sample.height, img_width=sample.width, as_bytes=True),
            sizes,
        ),
        "rle_compressed_str": (
            bitmask.to_coco_rle(img_height=sample.height, img_width=sample.width, as_bytes=False),
            sizes,
        ),
        "rle_as_ints": (rle_as_int_counts, sizes),
        "polygon": (polygon_segments, sizes),
    }
    return rle_types
