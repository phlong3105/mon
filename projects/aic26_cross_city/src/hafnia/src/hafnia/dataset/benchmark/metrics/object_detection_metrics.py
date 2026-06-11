from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Type

import numpy as np
from pydantic import BaseModel, Field

from hafnia.dataset.benchmark.metrics.metric_helpers import validate_matching_tasks
from hafnia.dataset.primitives import Bbox, Bitmask, Primitive
from hafnia.utils import progress_bar

if TYPE_CHECKING:
    from hafnia.dataset.hafnia_dataset import HafniaDataset


class ObjectDetectionMetrics(BaseModel):
    """Standard COCO mAP metrics (12 metrics from COCOeval.summarize)."""

    mAP: float = Field(description="AP at IoU=0.50:0.05:0.95 (primary challenge metric)")
    mAP_50: float = Field(description="AP at IoU=0.50 (PASCAL VOC metric)")
    mAP_75: float = Field(description="AP at IoU=0.75 (strict metric)")
    mAP_s: float = Field(description="AP for small objects (area < 32² px)")
    mAP_m: float = Field(description="AP for medium objects (32² < area < 96² px)")
    mAP_l: float = Field(description="AP for large objects (area > 96² px)")
    AR_1: float = Field(description="AR given 1 detection per image")
    AR_10: float = Field(description="AR given 10 detections per image")
    AR_100: float = Field(description="AR given 100 detections per image")
    AR_s: float = Field(description="AR for small objects (area < 32² px)")
    AR_m: float = Field(description="AR for medium objects (32² < area < 96² px)")
    AR_l: float = Field(description="AR for large objects (area > 96² px)")

    @staticmethod
    def from_coco_stats(stats: np.ndarray) -> "ObjectDetectionMetrics":
        stats_as_list = stats.tolist()  # Converts from numpy.float64 to native float
        return ObjectDetectionMetrics(
            mAP=stats_as_list[0],
            mAP_50=stats_as_list[1],
            mAP_75=stats_as_list[2],
            mAP_s=stats_as_list[3],
            mAP_m=stats_as_list[4],
            mAP_l=stats_as_list[5],
            AR_1=stats_as_list[6],
            AR_10=stats_as_list[7],
            AR_100=stats_as_list[8],
            AR_s=stats_as_list[9],
            AR_m=stats_as_list[10],
            AR_l=stats_as_list[11],
        )

    def as_dict(self) -> Dict[str, float]:
        """Return metrics as a dictionary."""
        metrics_dict = self.model_dump()
        return metrics_dict

    def report(self) -> str:
        """Return a formatted, human-readable report of the metrics."""
        header = "Object Detection Metrics (COCO mAP)"
        separator = "=" * len(header)
        fields = type(self).model_fields
        name_width = max(len(name) for name in fields)
        lines = [header, separator]
        for name, field_info in fields.items():
            value = getattr(self, name)
            description = field_info.description or ""
            lines.append(f"  {name:<{name_width}} = {value:6.3f}   {description}")
        return "\n".join(lines)

    def print_report(self) -> None:
        """Print the formatted metrics report to stdout."""
        print(self.report())


def _build_coco_data(
    dataset: "HafniaDataset",
    primitive: Type[Primitive],
    task_name_predictions: str,
    task_name_ground_truth: str,
    category_mapping: Dict[str, int],
    categories: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Iterate samples to build COCO GT dict and predictions list."""
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    ann_id = 1  # pycocotools treats id=0 as "no match" in boolean logic; IDs must start at 1
    column_name = primitive.column_name()  # "bboxes" or "bitmasks"

    for row in progress_bar(
        dataset.samples.iter_rows(named=True),
        description="Convert to 'pycocotools' format",
        total=len(dataset),
    ):
        sample_index = int(row["sample_index"])
        img_height = int(row["height"])
        img_width = int(row["width"])

        images.append(
            {
                "id": sample_index,
                "width": img_width,
                "height": img_height,
                "file_name": row.get("file_path") or "",
            }
        )

        for ann in row.get(column_name) or []:
            if ann is None:
                continue

            ann_task_name = ann.get("task_name")
            class_name = ann.get("class_name")
            if class_name not in category_mapping:
                continue
            cat_id = category_mapping[class_name]

            if ann_task_name == task_name_ground_truth:
                iscrowd = int((ann.get("meta") or {}).get("iscrowd") or 0)

                if primitive is Bbox:
                    bbox = [
                        ann["top_left_x"] * img_width,
                        ann["top_left_y"] * img_height,
                        ann["width"] * img_width,
                        ann["height"] * img_height,
                    ]
                    area = float(ann["height"] * ann["width"] * img_height * img_width)
                    segmentation: Any = []
                else:  # Bitmask
                    bbox = [
                        float(ann["left"]),
                        float(ann["top"]),
                        float(ann["width"]),
                        float(ann["height"]),
                    ]
                    area = float((ann.get("area") or 0.0) * img_height * img_width)
                    rle_counts = ann["rle_string"]
                    if isinstance(rle_counts, str):
                        rle_counts = rle_counts.encode("utf-8")
                    segmentation = {"counts": rle_counts, "size": [img_height, img_width]}

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": sample_index,
                        "category_id": cat_id,
                        "bbox": bbox,
                        "area": area,
                        "segmentation": segmentation,
                        "iscrowd": iscrowd,
                    }
                )
                ann_id += 1

            elif ann_task_name == task_name_predictions:
                score = float(ann.get("confidence") or 1.0)

                if primitive is Bbox:
                    pred: Dict[str, Any] = {
                        "image_id": sample_index,
                        "category_id": cat_id,
                        "bbox": [
                            ann["top_left_x"] * img_width,
                            ann["top_left_y"] * img_height,
                            ann["width"] * img_width,
                            ann["height"] * img_height,
                        ],
                        "score": score,
                        "area": float(ann["height"] * ann["width"] * img_height * img_width),
                    }
                else:  # Bitmask
                    rle_counts = ann["rle_string"]
                    if isinstance(rle_counts, str):
                        rle_counts = rle_counts.encode("utf-8")
                    pred = {
                        "image_id": sample_index,
                        "category_id": cat_id,
                        "segmentation": {"counts": rle_counts, "size": [img_height, img_width]},
                        "score": score,
                        "area": float((ann.get("area") or 0.0) * img_height * img_width),
                    }

                predictions.append(pred)

    gt_dict = {"images": images, "annotations": annotations, "categories": categories}
    return gt_dict, predictions


def calculate_mean_average_precision(
    dataset: "HafniaDataset",
    task_name_predictions: str,
    task_name_ground_truth: str,
) -> ObjectDetectionMetrics:
    """Calculate mean average precision (mAP) using the COCO evaluation protocol.

    Both ground-truth and prediction annotations must be stored in the same
    dataset, distinguished by their ``task_name`` field.  The primitive type
    (Bbox → ``"bbox"`` eval, Bitmask → ``"segm"`` eval) is inferred from the
    dataset's :class:`~hafnia.dataset.hafnia_dataset_types.TaskInfo` that
    matches *task_name_ground_truth*.

    Typical usage::

        metrics = dataset_with_predictions.calculate_mean_average_precision(
            task_name_predictions="predictions",
            task_name_ground_truth=Bbox.default_task_name(),
        )

    Args:
        dataset: Dataset containing both ground-truth and prediction annotations.
        task_name_predictions: Task name that identifies prediction annotations.
        task_name_ground_truth: Task name that identifies ground-truth annotations.

    Returns:
        A :class:`ObjectDetectionMetrics` dataclass with the 12 standard COCO metrics.
    """
    # Consider adding faster_coco_eval for faster evaluation. It is a drop in-replacement.
    # import faster_coco_eval
    # faster_coco_eval.init_as_pycocotools()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_task_info, _, gt_class_names = validate_matching_tasks(
        dataset=dataset,
        task_name_ground_truth=task_name_ground_truth,
        task_name_predictions=task_name_predictions,
        expected_primitives=(Bbox, Bitmask),
    )
    primitive = gt_task_info.primitive
    eval_type = "bbox" if primitive is Bbox else "segm"

    id_offset = 1  # COCO doesn't work nicely with 0-indexing so we start at 1.
    categories = [
        {"id": i, "name": name, "supercategory": "object"} for i, name in enumerate(gt_class_names, start=id_offset)
    ]
    category_mapping = {str(cat["name"]): cat["id"] for cat in categories}

    gt_dict, pred_list = _build_coco_data(
        dataset=dataset,
        primitive=primitive,
        task_name_predictions=task_name_predictions,
        task_name_ground_truth=task_name_ground_truth,
        category_mapping=category_mapping,  # type: ignore[arg-type]
        categories=categories,
    )
    if not pred_list:
        raise ValueError(
            f"No predictions found for task '{task_name_predictions}'. "
            "Ensure predictions are stored in the dataset with the correct task_name."
        )

    gt_dict.update({"info": {}})
    gt_coco = COCO()
    gt_coco.dataset = gt_dict

    gt_coco.createIndex()

    pred_coco = gt_coco.loadRes(pred_list)

    coco_eval = COCOeval(gt_coco, pred_coco, eval_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return ObjectDetectionMetrics.from_coco_stats(coco_eval.stats)
