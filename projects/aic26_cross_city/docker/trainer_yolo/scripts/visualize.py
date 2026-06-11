from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.utils import get_dataset_path_in_hafnia_cloud, is_hafnia_cloud_job, progress_bar
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

app = App(name="predict", help="Run YOLO sliced prediction and save visualizations")


@app.default
def main(
    model_path: Annotated[str, Parameter(help="Path to YOLO checkpoint")] = "checkpoints/model-yolo26l_v2/model.pt",
    output_path: Annotated[
        str, Parameter(help="Directory where prediction visualizations are saved")
    ] = ".data/predictions-yolo",
    split_name: Annotated[str, Parameter(help="Dataset split to run prediction on")] = SplitName.TEST,
    samples: Annotated[int, Parameter(help="Number of samples to predict and visualize")] = 300,
    imgsz: Annotated[int, Parameter(help="YOLO inference image size (px)")] = 640,
    threshold: Annotated[float, Parameter(help="Confidence threshold for predictions")] = 0.3,
    slice_height: Annotated[int, Parameter(help="SAHI slice height (px)")] = 640,
    slice_width: Annotated[int, Parameter(help="SAHI slice width (px)")] = 640,
    overlap_height_ratio: Annotated[float, Parameter(help="SAHI slice overlap ratio in height")] = 0.2,
    overlap_width_ratio: Annotated[float, Parameter(help="SAHI slice overlap ratio in width")] = 0.2,
    postprocess_match_threshold: Annotated[float, Parameter(help="NMS IoU threshold for slice merging")] = 0.3,
    postprocess_class_agnostic: Annotated[
        bool, Parameter(help="Use class-agnostic NMS across categories")
    ] = False,
):
    path_prediction_visualization = Path(output_path)
    path_prediction_visualization.mkdir(parents=True, exist_ok=True)
    print(f"Saving prediction visualizations to: {path_prediction_visualization.resolve()}")

    if is_hafnia_cloud_job():
        path_dataset = get_dataset_path_in_hafnia_cloud()
        dataset = HafniaDataset.from_path(path_dataset)
    else:
        dataset = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda:0"
    except Exception:
        pass

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=threshold,
        device=device,
        image_size=imgsz,
    )

    dataset_split = dataset.create_split_dataset(split_name=split_name)

    test_subset = dataset_split.select_samples(n_samples=samples, seed=42)
    for i_sample, dict_sample in enumerate(progress_bar(test_subset)):
        sample = Sample(**dict_sample)
        image = sample.read_image()

        prediction_result = get_sliced_prediction(
            image=image,
            detection_model=detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=postprocess_match_threshold,
            postprocess_class_agnostic=postprocess_class_agnostic,
            confidence_threshold=threshold,
        )

        file_name = f"prediction_visualization_{i_sample}"
        prediction_result.export_visuals(
            export_dir=path_prediction_visualization.as_posix(),
            file_name=file_name,
        )
        print(f"Saved -> {path_prediction_visualization / f'{file_name}.png'}")


if __name__ == "__main__":
    app()
