from typing import Dict, List, Optional, Tuple, Type, Union

import cv2
import numpy as np
import polars as pl
import torch
import torchvision
from flatten_dict import flatten, unflatten
from PIL import Image, ImageDraw, ImageFont
from torchvision import tv_tensors
from torchvision import utils as tv_utils
from torchvision.transforms import v2

from hafnia.dataset.dataset_names import PrimitiveField, SampleField
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.dataset.primitives import (
    PRIMITIVE_COLUMN_NAMES,
    Bbox,
    Bitmask,
    Classification,
    Polygon,
    Segmentation,
    class_color_by_name,
)
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.log import user_logger


def get_primitives_per_task_name_for_primitive(
    sample: Sample, PrimitiveType: Type[Primitive], split_by_task_name: bool = True
) -> Dict[str, List[Primitive]]:
    if not hasattr(sample, PrimitiveType.column_name()):
        return {}

    primitives = getattr(sample, PrimitiveType.column_name())
    if primitives is None:
        return {}

    primitives_by_task_name: Dict[str, List[Primitive]] = {}
    for primitive in primitives:
        if primitive.task_name not in primitives_by_task_name:
            primitives_by_task_name[primitive.task_name] = []
        primitives_by_task_name[primitive.task_name].append(primitive)
    return primitives_by_task_name


class TorchvisionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: HafniaDataset,
        transforms=None,
        keep_metadata: bool = False,
    ):
        self.dataset = dataset

        self.max_points_in_polygon = 0

        if self.dataset.has_primitive(Polygon):
            self.max_points_in_polygon = (
                self.dataset.samples[SampleField.POLYGONS]
                .list.eval(pl.element().struct.field("points").list.len())
                .explode()
                .max()
            )

        self.transforms = transforms
        self.keep_metadata = keep_metadata

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        sample_dict = self.dataset[idx]
        sample = Sample(**sample_dict)
        image = tv_tensors.Image(sample.read_image_pillow())
        h, w = image.shape[-2:]
        target_flat = {}
        mask_tasks: Dict[str, List[Segmentation]] = get_primitives_per_task_name_for_primitive(sample, Segmentation)
        for task_name, masks in mask_tasks.items():
            raise NotImplementedError("Segmentation tasks are not yet implemented")
            # target[f"{mask.task_name}.mask"] = tv_tensors.Mask(mask.mask)

        class_tasks: Dict[str, List] = get_primitives_per_task_name_for_primitive(sample, Classification)
        for task_name, classifications in class_tasks.items():
            assert len(classifications) == 1, "Expected exactly one classification task per sample"
            target_flat[f"{Classification.column_name()}.{task_name}"] = {
                PrimitiveField.CLASS_IDX: classifications[0].class_idx,
                PrimitiveField.CLASS_NAME: classifications[0].class_name,
            }

        bbox_tasks: Dict[str, List[Bbox]] = get_primitives_per_task_name_for_primitive(sample, Bbox)
        for task_name, bboxes in bbox_tasks.items():
            bboxes_list = [bbox.to_coco_ints(image_height=h, image_width=w) for bbox in bboxes]
            bboxes_tensor = torch.as_tensor(bboxes_list).reshape(-1, 4)
            target_flat[f"{Bbox.column_name()}.{task_name}"] = {
                PrimitiveField.CLASS_IDX: [bbox.class_idx for bbox in bboxes],
                PrimitiveField.CLASS_NAME: [bbox.class_name for bbox in bboxes],
                "bbox": tv_tensors.BoundingBoxes(bboxes_tensor, format="XYWH", canvas_size=(h, w)),
            }

        bitmask_tasks: Dict[str, List[Bitmask]] = get_primitives_per_task_name_for_primitive(sample, Bitmask)
        for task_name, bitmasks in bitmask_tasks.items():
            bitmasks_np = np.array([bitmask.to_mask(img_height=h, img_width=w) for bitmask in bitmasks])
            target_flat[f"{Bitmask.column_name()}.{task_name}"] = {
                PrimitiveField.CLASS_IDX: [bitmask.class_idx for bitmask in bitmasks],
                PrimitiveField.CLASS_NAME: [bitmask.class_name for bitmask in bitmasks],
                "mask": tv_tensors.Mask(bitmasks_np),
            }

        polygon_tasks: Dict[str, List[Polygon]] = get_primitives_per_task_name_for_primitive(sample, Polygon)
        for task_name, polygons in polygon_tasks.items():
            polygon_tensors = [
                torch.tensor(pg.to_pixel_coordinates(image_shape=(h, w), as_int=False)) for pg in polygons
            ]
            n_polygons = len(polygons)
            polygons_matrix = torch.full((n_polygons, self.max_points_in_polygon, 2), fill_value=torch.nan)

            for i, polygon_tensor in enumerate(polygon_tensors):
                polygons_matrix[i, : polygon_tensor.shape[0], :] = polygon_tensor

            target_flat[f"{Polygon.column_name()}.{task_name}"] = {
                PrimitiveField.CLASS_IDX: [polygon.class_idx for polygon in polygons],
                PrimitiveField.CLASS_NAME: [polygon.class_name for polygon in polygons],
                "polygon": tv_tensors.KeyPoints(polygons_matrix, canvas_size=(h, w)),
            }
        if self.transforms:
            image, target_flat = self.transforms(image, target_flat)

        if self.keep_metadata:
            sample_dict = sample_dict.copy()
            drop_columns = PRIMITIVE_COLUMN_NAMES
            for column in drop_columns:
                if column in sample_dict:
                    sample_dict.pop(column)

        target = flatten(target_flat, reducer="dot")
        return image, target

    def __len__(self):
        return len(self.dataset)


def draw_image_classification(visualize_image: torch.Tensor, text_labels: Union[str, List[str]]) -> torch.Tensor:
    if isinstance(text_labels, str):
        text_labels = [text_labels]
    text = "\n".join(text_labels)
    max_dim = max(visualize_image.shape[-2:])
    font_size = max(int(max_dim * 0.06), 10)  # Minimum font size of 10
    txt_font = ImageFont.load_default(font_size)
    dummie_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    _, _, w, h = dummie_draw.textbbox((0, 0), text=text, font=txt_font)  # type: ignore[arg-type]

    text_image = Image.new("RGB", (int(w), int(h)))
    draw = ImageDraw.Draw(text_image)
    draw.text((0, 0), text=text, font=txt_font)  # type: ignore[arg-type]
    text_tensor = v2.functional.to_image(text_image)

    height = text_tensor.shape[-2] + visualize_image.shape[-2]
    width = max(text_tensor.shape[-1], visualize_image.shape[-1])
    visualize_image_new = torch.zeros((3, height, width), dtype=visualize_image.dtype)
    shift_w = (width - visualize_image.shape[-1]) // 2
    visualize_image_new[:, : visualize_image.shape[-2], shift_w : shift_w + visualize_image.shape[-1]] = visualize_image
    shift_w = (width - text_tensor.shape[-1]) // 2
    shift_h = visualize_image.shape[-2]
    visualize_image_new[:, shift_h : shift_h + text_tensor.shape[-2], shift_w : shift_w + text_tensor.shape[-1]] = (
        text_tensor
    )
    visualize_image = visualize_image_new
    return visualize_image


def draw_image_and_targets(
    image: torch.Tensor,
    targets,
) -> torch.Tensor:
    visualize_image = image.clone()
    if visualize_image.is_floating_point():
        visualize_image = image - torch.min(image)
        visualize_image = visualize_image / visualize_image.max()

    visualize_image = v2.functional.to_dtype(visualize_image, torch.uint8, scale=True)
    targets = unflatten(targets, splitter="dot")  # Nested dictionary format
    # NOTE: Order of drawing is important so visualizations are not overlapping in an undesired way
    if Segmentation.column_name() in targets:
        primitive_annotations = targets[Segmentation.column_name()]
        for task_name, task_annotations in primitive_annotations.items():
            raise NotImplementedError("Segmentation tasks are not yet implemented")
            # mask = targets[mask_field].squeeze(0)
            # masks_list = [mask == value for value in mask.unique()]
            # masks = torch.stack(masks_list, dim=0).to(torch.bool)
            # visualize_image = tv_utils.draw_segmentation_masks(visualize_image, masks=masks, alpha=0.5)

    if Bitmask.column_name() in targets:
        primitive_annotations = targets[Bitmask.column_name()]
        for task_name, task_annotations in primitive_annotations.items():
            colors = [class_color_by_name(class_name) for class_name in task_annotations[PrimitiveField.CLASS_NAME]]
            visualize_image = tv_utils.draw_segmentation_masks(
                image=visualize_image,
                masks=task_annotations["mask"],
                colors=colors,
            )

    if Bbox.column_name() in targets:
        primitive_annotations = targets[Bbox.column_name()]
        for task_name, task_annotations in primitive_annotations.items():
            bboxes = torchvision.ops.box_convert(task_annotations["bbox"], in_fmt="xywh", out_fmt="xyxy")
            colors = [class_color_by_name(class_name) for class_name in task_annotations[PrimitiveField.CLASS_NAME]]
            visualize_image = tv_utils.draw_bounding_boxes(
                image=visualize_image,
                boxes=bboxes,
                labels=task_annotations[PrimitiveField.CLASS_NAME],
                width=2,
                colors=colors,
            )

    if Polygon.column_name() in targets:
        primitive_annotations = targets[Polygon.column_name()]
        np_image = visualize_image.permute(1, 2, 0).numpy()
        for task_name, task_annotations in primitive_annotations.items():
            task_annotations["polygon"]
            colors = [class_color_by_name(class_name) for class_name in task_annotations[PrimitiveField.CLASS_NAME]]
            for color, polygon in zip(colors, task_annotations["polygon"], strict=True):
                single_polygon = np.array(polygon[~torch.isnan(polygon[:, 0]), :][None, :, :]).astype(int)

                np_image = cv2.polylines(np_image, [single_polygon], isClosed=False, color=color, thickness=2)
        visualize_image = torch.from_numpy(np_image).permute(2, 0, 1)

    # Important that classification is drawn last as it will change image dimensions
    if Classification.column_name() in targets:
        primitive_annotations = targets[Classification.column_name()]
        text_labels = []
        for task_name, task_annotations in primitive_annotations.items():
            if task_name == Classification.default_task_name():
                text_label = task_annotations[PrimitiveField.CLASS_NAME]
            else:
                text_label = f"{task_name}: {task_annotations[PrimitiveField.CLASS_NAME]}"
            text_labels.append(text_label)
        visualize_image = draw_image_classification(visualize_image, text_labels)
    return visualize_image


class TorchVisionCollateFn:
    def __init__(self, skip_stacking: Optional[List] = None):
        if skip_stacking is None:
            skip_stacking = [f"{Bbox.column_name()}.*", f"{Bitmask.column_name()}.*"]

        self.wild_card_skip_stacking = []
        self.skip_stacking_list = []
        for skip_name in skip_stacking:
            if skip_name.endswith("*"):
                self.wild_card_skip_stacking.append(skip_name[:-1])  # Remove the trailing '*'
            else:
                self.skip_stacking_list.append(skip_name)

    def skip_key_name(self, key_name: str) -> bool:
        if key_name in self.skip_stacking_list:
            return True
        if any(key_name.startswith(wild_card) for wild_card in self.wild_card_skip_stacking):
            return True
        return False

    def __call__(self, batch):
        images, targets = tuple(zip(*batch, strict=False))
        if "image" not in self.skip_stacking_list:
            images = torch.stack(images)
        height, width = images.shape[-2:]
        keys_min = set(targets[0])
        keys_max = set(targets[0])
        for target in targets:
            keys_min = keys_min.intersection(target)
            keys_max = keys_max.union(target)

        if keys_min != keys_max:
            user_logger.warning(
                "Not all images in the batch contain the same targets. To solve for missing targets "
                f"the following keys {keys_max - keys_min} are dropped from the batch "
            )

        targets_modified = {k: [d[k] for d in targets] for k in keys_min}
        for key_name, item_values in targets_modified.items():
            if self.skip_key_name(key_name):
                continue
            first_element = item_values[0]
            if isinstance(first_element, torch.Tensor):
                item_values = torch.stack(item_values)
            elif isinstance(first_element, (int, float)):
                item_values = torch.tensor(item_values)
            elif isinstance(first_element, (str, list)):
                # Skip stacking for certain types such as strings and lists
                pass
            if isinstance(first_element, tv_tensors.Mask):
                item_values = tv_tensors.Mask(item_values)
            elif isinstance(first_element, tv_tensors.Image):
                item_values = tv_tensors.Image(item_values)
            elif isinstance(first_element, tv_tensors.BoundingBoxes):
                item_values = tv_tensors.BoundingBoxes(item_values)
            elif isinstance(first_element, tv_tensors.KeyPoints):
                item_values = tv_tensors.KeyPoints(item_values, canvas_size=(height, width))
            targets_modified[key_name] = item_values

        return images, targets_modified
