from typing import Callable

import numpy as np
import pytest

from hafnia.dataset import image_visualizations
from hafnia.dataset.primitives import Bbox, Bitmask, Polygon
from tests import helper_testing


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_mask_region(compare_to_expected_image: Callable, dataset_name: str):
    sample = helper_testing.get_sample_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    image = sample.read_image()
    if dataset_name == "micro-coco-2017":
        annotations = sample.get_primitives([Bitmask])
    else:
        annotations = sample.get_primitives()
    masked_image = image_visualizations.draw_masks(image, annotations)
    compare_to_expected_image(masked_image)


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_draw_annotations(compare_to_expected_image: Callable, dataset_name: str):
    sample = helper_testing.get_sample_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    image = sample.read_image()
    annotations = sample.get_primitives()
    masked_image = image_visualizations.draw_annotations(image, annotations)
    compare_to_expected_image(masked_image)


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_blur_anonymization(compare_to_expected_image: Callable, dataset_name: str):
    sample = helper_testing.get_sample_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    image = sample.read_image()
    if dataset_name == "micro-coco-2017":
        annotations = sample.get_primitives([Bitmask])
    else:
        annotations = sample.get_primitives([Bitmask, Bbox, Polygon])

    masked_image = image_visualizations.draw_anonymize_by_blurring(image, annotations)
    compare_to_expected_image(masked_image)


def test_polygon_to_bitmask_conversion(compare_to_expected_image: Callable):
    sample = helper_testing.get_sample_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)
    image = sample.read_image()
    annotations = sample.get_primitives()
    polygons = [a for a in annotations if isinstance(a, Polygon)]

    bitmasks = []
    assert len(polygons) > 0, "There should be at least one Polygon annotation in the sample to test mask conversion."
    for polygon in polygons:
        bitmask = polygon.to_bitmask(img_height=image.shape[0], img_width=image.shape[1])
        bitmasks.append(bitmask)
        mask_from_polygon = polygon.to_mask(img_height=image.shape[0], img_width=image.shape[1], use_coco_utils=True)
        mask_from_bitmask = bitmask.to_mask(img_height=image.shape[0], img_width=image.shape[1])
        assert np.array_equal(mask_from_polygon, mask_from_bitmask), "Masks from Polygon and Bitmask should match."

    masked_image = image_visualizations.draw_annotations(image, bitmasks)
    compare_to_expected_image(masked_image)
