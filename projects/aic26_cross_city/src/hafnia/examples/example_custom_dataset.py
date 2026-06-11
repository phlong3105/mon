import uuid
from pathlib import Path
from typing import List

from PIL import Image

from hafnia.dataset.dataset_names import SampleField
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, Sample, TaskInfo
from hafnia.dataset.primitives.bbox import Bbox

"""
This script demonstrates how to create a HafniaDataset from custom data.
Note: We already have a function for converting a yolo formatted dataset to a HafniaDataset using
HafniaDataset.from_yolo_format(). But the purpose is to demonstrate how to create a HafniaDataset 
from a custom dataset format. 
"""

# Define paths to the custom dataset in YOLO format
path_tmp = Path(".data/tmp")
path_yolo_dataset = Path("tests/data/dataset_formats/format_yolo/train")
path_class_names = path_yolo_dataset.parent / "obj.names"

# Get class names
class_names = [line.strip() for line in path_class_names.read_text().splitlines() if line.strip()]

# Get image file paths
path_images_file = path_yolo_dataset / "images.txt"
image_files = [line.strip() for line in path_images_file.read_text().splitlines() if line.strip()]

# Iterate over image files and corresponding bounding box files to create Sample objects for the HafniaDataset
fake_samples = []
for image_file in image_files:
    path_image = path_yolo_dataset / image_file
    path_bboxes = path_yolo_dataset / image_file.replace(".jpg", ".txt")
    bboxes: List[Bbox] = []
    for bboxes_line in path_bboxes.read_text().splitlines():
        str_parts = bboxes_line.strip().split()
        class_idx = int(str_parts[0])
        x_center, y_center, bbox_width, bbox_height = (float(value) for value in str_parts[1:5])
        bbox = Bbox(
            top_left_x=x_center - bbox_width / 2,
            top_left_y=y_center - bbox_height / 2,
            width=bbox_width,
            height=bbox_height,
            class_idx=class_idx,
            class_name=class_names[class_idx],
        )
        bboxes.append(bbox)
    image = Image.open(path_image)
    height, width = image.size[1], image.size[0]
    sample = Sample(file_path=str(path_image), height=height, width=width, split="train", bboxes=bboxes)
    fake_samples.append(sample)


unique_dataset_name = f"custom-dataset-{str(uuid.uuid4())[:8]}"  # Unique name to avoid conflicts with existing datasets

# Create dataset info
dataset_info = DatasetInfo(
    dataset_name=unique_dataset_name,
    version="0.0.1",
    tasks=[TaskInfo(primitive=Bbox, class_names=class_names)],
)
custom_dataset = HafniaDataset.from_samples_list(samples_list=fake_samples, info=dataset_info)

sample = Sample(**custom_dataset[0])

# To visualize and verify dataset is formatted correctly store image with annotations
image_with_annotations = sample.draw_annotations()
Image.fromarray(image_with_annotations).save(path_tmp / "custom_dataset_sample.png")  # Save visualization to TMP


# Upload dataset to Hafnia platform (optional)
gallery_image_names = [custom_dataset.samples[SampleField.FILE_PATH].str.split("/").list.last().sort()[0]]

# custom_dataset.upload_to_platform(
#     interactive=False,
#     allow_version_overwrite=True,
#     gallery_images=gallery_image_names,
# )
