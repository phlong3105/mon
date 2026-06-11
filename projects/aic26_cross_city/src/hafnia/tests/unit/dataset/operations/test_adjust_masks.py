from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
import pytest

from hafnia.dataset import image_visualizations
from hafnia.dataset.dataset_names import SampleField
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.dataset.operations.adjust_mask import _adjust_bboxes_from_polygon_masks
from hafnia.dataset.primitives import Bbox, Polygon
from tests import helper_testing


@dataclass
class TestCaseAdjustBbox:
    key: str
    boxes: List[Bbox]
    image_height: int = 1080
    image_width: int = 1920
    polygons: List[List[List[int]]] = field(default_factory=lambda: DEFAULT_POLYGON_MASKS)

    @classmethod
    def from_x1_y1_x2_y2(
        cls,
        key: str,
        box_coords: List[List[int]],
    ) -> "TestCaseAdjustBbox":
        test_case = cls(key=key, boxes=[])
        for coords in box_coords:
            x1, y1, x2, y2 = coords
            coco_box = [x1, y1, x2 - x1, y2 - y1]
            bbox = Bbox.from_coco(coco_box, height=test_case.image_height, width=test_case.image_width)
            bbox.class_name = "original"  # Set class name for visualization
            test_case.boxes.append(bbox)
        return test_case

    def polygons_as_primitives(self) -> List[Polygon]:
        polygons: List[Polygon] = []
        for idx, polygon_points in enumerate(self.polygons):
            polygons.append(
                Polygon.from_list_of_pixel_points(
                    points=polygon_points,
                    image_height=self.image_height,
                    image_width=self.image_width,
                    class_name=f"P{idx}",
                )
            )
        return polygons


DEFAULT_POLYGON_MASKS: List[List[List[int]]] = [
    [  # original polygon
        [529, 83],
        [614, 223],
        [1600, 400],
        [1450, 122],
        [1500, 20],
        [1450, 20],
        [1420, 80],
        [1450, 122],
    ],
    [  # second polygon: square
        [550, 600],
        [850, 600],
        [850, 900],
        [550, 900],
    ],
    [  # third polygon: square
        [614, 323],
        [1600, 500],
        [1600, 600],
        [561, 535],
    ],
]


TEST_CASES: List[TestCaseAdjustBbox] = [
    TestCaseAdjustBbox.from_x1_y1_x2_y2("AdjustBottom", box_coords=[[750, 20, 1000, 200]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("AdjustBottomAndLeftSide", box_coords=[[1400, 150, 1480, 200]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("AdjustBottomAndRightSide", box_coords=[[1435, 30, 1460, 60]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("AdjustLeftSide", box_coords=[[1400, 150, 1600, 200]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("AdjustRightSide", box_coords=[[500, 150, 750, 200]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("AdjustTopAndLeftSide", box_coords=[[1460, 30, 1490, 60]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("AdjustTopAndRightSide", box_coords=[[580, 150, 750, 200]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("DropBboxInsidePolygon", box_coords=[[1000, 180, 1200, 220]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("DropEmptyBboxInsidePolygon", box_coords=[[1000, 180, 1000, 180]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("NotAdjustedAdjacentCornersInPolygon1", box_coords=[[1400, 150, 1500, 370]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("NotAdjustedAdjacentCornersInPolygon2", box_coords=[[1435, 30, 1490, 60]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("NotAdjustedBboxOutsidePolygon", box_coords=[[100, 180, 200, 230]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("NotAdjustedEmptyBboxOutsidePolygon", box_coords=[[100, 180, 100, 180]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("NotAdjustedOneCornerInPolygon1", box_coords=[[500, 50, 750, 200]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("NotAdjustedOneCornerInPolygon2", box_coords=[[1400, 50, 1600, 200]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("NotAdjustedTwoCornerInTwoPolygons", box_coords=[[500, 200, 750, 400]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("SqueezeBetweenTwoPolygons1", box_coords=[[600, 500, 800, 750]]),
    TestCaseAdjustBbox.from_x1_y1_x2_y2("SqueezeBetweenTwoPolygons2", box_coords=[[750, 200, 1000, 400]]),
]


@pytest.mark.parametrize("case_key", TEST_CASES, ids=[case.key for case in TEST_CASES])
def test_cases_adjust_bbox_from_polygon(case_key: TestCaseAdjustBbox, compare_to_expected_image: Callable):
    polygons = case_key.polygons_as_primitives()
    adjust_bboxes = _adjust_bboxes_from_polygon_masks(
        boxes=case_key.boxes,
        polygons=polygons,
        image_width=case_key.image_width,
        image_height=case_key.image_height,
    )

    for adj in adjust_bboxes:
        adj.class_name = "adjusted"

    # Draws polygons and adjusted bboxes on the image for visualization
    img = np.zeros((case_key.image_height, case_key.image_width, 3), dtype=np.uint8)
    annotations = polygons + case_key.boxes + adjust_bboxes
    img_annotated = image_visualizations.draw_annotations(image=img, primitives=annotations)
    compare_to_expected_image(img_annotated)


def test_adjust_bbox_from_polygon(compare_to_expected_image: Callable):
    dataset: HafniaDataset = helper_testing.get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")
    polygon_masks = ["Annotator Marking Polygon.Mask"]
    dataset_adjusted = dataset.adjust_bboxes_from_polygon_masks(polygon_class_names=polygon_masks)
    dataset_adjusted.check_dataset(check_splits=False)  # Check that the dataset is still valid after adjustments

    person_bbox_adjusted_sample = 1  # Sample index showing an adjusted bbox
    sample = Sample(**dataset[person_bbox_adjusted_sample])
    sample_adjusted = Sample(**dataset_adjusted[person_bbox_adjusted_sample])

    image = sample.draw_annotations()
    image_roi = image[45:130, 600:800, :]
    image_bbox_adjusted = sample_adjusted.draw_annotations()
    image_roi_adjusted = image_bbox_adjusted[45:130, 600:800, :]

    images = np.concatenate([image_roi, image_roi_adjusted], axis=1)
    compare_to_expected_image(images)

    is_adjusted_sample = dataset_adjusted.samples[SampleField.BBOXES] != dataset.samples[SampleField.BBOXES]
    assert len(is_adjusted_sample) == 3
    assert sum(is_adjusted_sample) == 2, "Only one sample should have adjusted bboxes"


def test_adjust_bbox_from_polygon_assertions(compare_to_expected_image: Callable):
    dataset: HafniaDataset = helper_testing.get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")

    # Test case 1: Polygon class names not present in dataset tasks
    with pytest.raises(ValueError, match="not present in the dataset tasks"):
        dataset.adjust_bboxes_from_polygon_masks(polygon_class_names=["Doesn't exist"])

    # Test case 2: Polygon Primitive is not even present in the dataset tasks
    with pytest.raises(ValueError, match="No Polygon tasks found in the dataset"):
        # Drop polygon tasks to trigger error
        dataset.info.tasks = [t for t in dataset.info.tasks if t.primitive != Polygon]

        dataset.adjust_bboxes_from_polygon_masks(polygon_class_names=["Annotator Marking Polygon.Mask"])
