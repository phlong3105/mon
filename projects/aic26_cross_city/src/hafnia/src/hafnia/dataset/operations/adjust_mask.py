from typing import TYPE_CHECKING, List, Optional, Tuple

import more_itertools
import polars as pl

from hafnia.dataset.dataset_names import PrimitiveField, SampleField
from hafnia.dataset.primitives import Bbox, Point, Polygon
from hafnia.log import user_logger
from hafnia.utils import progress_bar

if TYPE_CHECKING:
    from hafnia.dataset.hafnia_dataset import HafniaDataset


def adjust_bboxes_from_polygon_masks_dataset(
    dataset: "HafniaDataset",
    polygon_class_names: List[str],
    run_checks: bool = True,
) -> "HafniaDataset":
    """
    Adjust bounding boxes to avoid overlapping with polygon masks for all samples in the dataset.

    Functions use pydantic Bbox and Polygon primitives to carry information instead of
    python native types such as tuple, list and dict.

    We use Bbox and Polygon primitives for better maintainability, but it adds computational overhead.
    We may consider using python native types in the future for better performance

    Performance: Running bbox adjustment for all samples:
    - coco-2017:
        - 26.3s: Python native types
        - 38.5s: Pydantic primitives (current implementation)

    - midwest-vehicle-detection:
        - 6.1s: Python native types
        - 9.0s: Pydantic primitives (current implementation)

    To revert ask Agent to use these types instead of pydantic primitives:
        _PolyPoints = List[Tuple[float, float]]  # normalized (x, y) points defining a polygon
        _BboxDict = Dict[str, Any]  # dict with keys: top_left_x, top_left_y, width, height (plus metadata)

    """
    if run_checks:
        # Check tasks for 'polygon_class_names'
        polygon_tasks = dataset.info.get_tasks_by_primitive(Polygon)
        if len(polygon_tasks) == 0:
            raise ValueError("No Polygon tasks found in the dataset, cannot adjust bboxes from polygon masks")
        classes_from_tasks = set(more_itertools.flatten([t.get_class_names() or [] for t in polygon_tasks]))
        has_existing_polygon_class = set(polygon_class_names).issubset(classes_from_tasks)
        if not has_existing_polygon_class:
            raise ValueError(
                f"Polygon class names {polygon_class_names} are not present in the dataset tasks. "
                f"Available polygon class names from tasks: {classes_from_tasks}"
            )

        # Check samples for 'polygon_class_names'
        classes_in_samples_df = dataset.create_primitive_table(Polygon)
        if classes_in_samples_df is None:
            raise ValueError(
                "No polygon primitives found in the dataset samples, cannot adjust bboxes from polygon masks"
            )
        classes_in_samples = set(classes_in_samples_df[PrimitiveField.CLASS_NAME].drop_nulls().to_list())
        has_existing_polygon_class_in_samples = len(classes_in_samples.intersection(polygon_class_names)) > 0
        if not has_existing_polygon_class_in_samples:
            raise ValueError(
                f"Polygon class names {polygon_class_names} are not present in the dataset samples. "
                f"Available polygon class names in samples: {classes_in_samples}"
            )

    adjusted_bboxes_per_sample = []
    for sample in progress_bar(dataset, description="Adjusting bboxes"):
        bboxes_dict = sample.get(SampleField.BBOXES, []) or []  # Returns list if missing or is None
        boxes = [Bbox(**bbox) for bbox in bboxes_dict]
        polygons_dict = sample.get(SampleField.POLYGONS, []) or []  # Returns list if missing or is None
        polygons = [Polygon(**poly) for poly in polygons_dict if poly[PrimitiveField.CLASS_NAME] in polygon_class_names]

        adjusted_boxes = _adjust_bboxes_from_polygon_masks(
            boxes=boxes,
            polygons=polygons,
            image_width=sample[SampleField.WIDTH],
            image_height=sample[SampleField.HEIGHT],
        )
        adjusted_boxes_dicts = {SampleField.BBOXES: [box.model_dump(mode="json") for box in adjusted_boxes]}
        adjusted_bboxes_per_sample.append(adjusted_boxes_dicts)  # Convert to list of dicts for JSON serialization
    samples_adjusted_bboxes = dataset.samples.with_columns(pl.from_records(adjusted_bboxes_per_sample))

    if run_checks:
        adjusted_samples = samples_adjusted_bboxes[SampleField.BBOXES] != dataset.samples[SampleField.BBOXES]
        num_adjusted = adjusted_samples.sum()
        user_logger.info(f"Adjusted bboxes for '{num_adjusted}' out of '{len(adjusted_samples)}' samples ")
    dataset_updated = dataset.update_samples(samples_adjusted_bboxes)
    return dataset_updated


def _adjust_bboxes_from_polygon_masks(
    boxes: List[Bbox],
    polygons: List[Polygon],
    image_width: int,
    image_height: int,
) -> List[Bbox]:
    """Adjust bounding boxes to avoid overlapping with polygon masks."""
    bboxes_adjusted: List[Bbox] = []
    for bbox in boxes:
        bbox_adjusted = bbox
        for polygon in polygons:
            bbox_adjusted = _adjust_bbox_with_polygon_mask(  # type: ignore[assignment]
                bbox=bbox_adjusted,
                polygon=polygon,
                W=image_width,
                H=image_height,
            )

            if bbox_adjusted is None:  # None == stop and remove the bbox
                break
        if bbox_adjusted is None:  # None == stop and remove the bbox
            continue
        bboxes_adjusted.append(bbox_adjusted)
    return bboxes_adjusted


def _adjust_bbox_with_polygon_mask(bbox: Bbox, polygon: Polygon, W: int, H: int) -> Optional[Bbox]:
    if len(polygon.points) == 0:
        return bbox

    iTL, iTR, iBR, iBL = _corners_states(bbox, polygon)

    # Drop bbox if all corners are inside the polygon
    all_corners_inside_polygon = iTL and iTR and iBR and iBL
    if all_corners_inside_polygon:
        return None

    for _ in range(8):
        side = _first_adjacent_pair_inside(bbox, polygon)
        if side is None:
            break
        bbox_adjusted0 = _adjust_side_minimal(bbox=bbox, side=side, polygon=polygon, W=W, H=H)
        if bbox_adjusted0 != bbox:
            bbox = bbox_adjusted0
            continue
        opp = {"top": "bottom", "right": "left", "bottom": "top", "left": "right"}[side]
        bbox_adjusted1 = _adjust_side_minimal(bbox, opp, polygon, W, H)
        if bbox_adjusted1 != bbox:
            bbox = bbox_adjusted1
            continue
        break

    # Clamp to valid normalized range, ensuring non-negative width and height
    top_left_x = _clamp(bbox.top_left_x, 0.0, 1.0)
    top_left_y = _clamp(bbox.top_left_y, 0.0, 1.0)
    width = _clamp(bbox.width, 0.0, 1.0 - top_left_x)
    height = _clamp(bbox.height, 0.0, 1.0 - top_left_y)
    return bbox.model_copy(
        update={"top_left_x": top_left_x, "top_left_y": top_left_y, "width": width, "height": height}
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _point_on_segment(p: Point, a: Point, b: Point) -> bool:
    cross = (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x)
    if abs(cross) > 1e-12:
        return False
    min_x, max_x = min(a.x, b.x), max(a.x, b.x)
    min_y, max_y = min(a.y, b.y), max(a.y, b.y)
    return min_x <= p.x <= max_x and min_y <= p.y <= max_y


def _point_in_poly_inclusive(point: Point, polygon: Polygon) -> bool:
    n = len(polygon.points)
    if n < 3:
        return False
    inside = False
    for i in range(n):
        p0 = polygon.points[i]
        p1 = polygon.points[(i + 1) % n]
        if _point_on_segment(point, p0, p1):
            return True
        if (p0.y > point.y) != (p1.y > point.y):
            xinters = p0.x + (point.y - p0.y) * (p1.x - p0.x) / (p1.y - p0.y)
            if point.x <= xinters:
                inside = not inside
    return inside


def _bbox_corners(bbox: Bbox) -> Tuple[Point, Point, Point, Point]:
    x1, y1 = bbox.top_left_x, bbox.top_left_y
    x2, y2 = x1 + bbox.width, y1 + bbox.height
    return (
        Point(x=x1, y=y1),  # TL
        Point(x=x2, y=y1),  # TR
        Point(x=x2, y=y2),  # BR
        Point(x=x1, y=y2),  # BL
    )


def _corners_states(bbox: Bbox, polygon: Polygon) -> Tuple[bool, bool, bool, bool]:
    tl, tr, br, bl = _bbox_corners(bbox)
    return (
        _point_in_poly_inclusive(tl, polygon),  # TL
        _point_in_poly_inclusive(tr, polygon),  # TR
        _point_in_poly_inclusive(br, polygon),  # BR
        _point_in_poly_inclusive(bl, polygon),  # BL
    )


def _inside_any(p: Point, polygons: List[Polygon]) -> bool:
    return any(_point_in_poly_inclusive(p, polygon) for polygon in polygons)


def _first_adjacent_pair_inside(bbox: Bbox, polygon: Polygon) -> Optional[str]:
    iTL, iTR, iBR, iBL = _corners_states(bbox, polygon)
    if iTL and iTR:
        return "top"
    if iTR and iBR:
        return "right"
    if iBR and iBL:
        return "bottom"
    if iBL and iTL:
        return "left"
    return None


def _valid_bbox(bbox: Bbox) -> bool:
    return bbox.width > 0 and bbox.height > 0


def _adjust_side_minimal(bbox: Bbox, side: str, polygon: Polygon, W: int, H: int) -> Bbox:
    """
    Minimally adjusts one side so that the pair of adjacent corners is no longer inside.
    RULE: always REDUCE the bbox (never expand it).
      - top:    y1 += delta
      - bottom: y2 -= delta
      - left:   x1 += delta
      - right:  x2 -= delta
    """
    dx = 1.0 / W
    dy = 1.0 / H

    def both_inside_for_side(bbox: Bbox) -> bool:
        iTL, iTR, iBR, iBL = _corners_states(bbox, polygon)
        return {"top": iTL and iTR, "right": iTR and iBR, "bottom": iBR and iBL, "left": iBL and iTL}[side]

    if not both_inside_for_side(bbox):
        return bbox

    x1 = bbox.top_left_x
    y1 = bbox.top_left_y
    x2 = x1 + bbox.width
    y2 = y1 + bbox.height

    for step in range(1, max(W, H) + 1):
        if side == "top":
            ny1 = _clamp(y1 + step * dy, 0.0, 1.0)
            cand = bbox.model_copy(update={"top_left_y": ny1, "height": y2 - ny1})

        elif side == "bottom":
            ny2 = _clamp(y2 - step * dy, 0.0, 1.0)
            cand = bbox.model_copy(update={"height": ny2 - y1})

        elif side == "left":
            nx1 = _clamp(x1 + step * dx, 0.0, 1.0)
            cand = bbox.model_copy(update={"top_left_x": nx1, "width": x2 - nx1})

        elif side == "right":
            nx2 = _clamp(x2 - step * dx, 0.0, 1.0)
            cand = bbox.model_copy(update={"width": nx2 - x1})

        else:
            raise ValueError(f"Unknown side: {side}")

        # Keep bbox valid and check that the adjacent pair is no longer inside
        if _valid_bbox(cand) and not both_inside_for_side(cand):
            return cand

    return bbox
