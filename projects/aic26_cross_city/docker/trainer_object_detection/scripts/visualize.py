from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from hafnia.dataset import image_visualizations
from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.utils import is_hafnia_cloud_job, progress_bar
from PIL import Image

from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

app = App(name="predict", help="Run prediction and save visualizations")

default_inference_config = InferenceConfig(compile=True, batch_size=1, threshold=0.35)


@app.default
def main(
    model_path: Annotated[
        str, Parameter(help="Path to the trained model archive (.zip)")
    ] = "./pretrained_models/RFDETRNano.zip",
    inference: Annotated[
        InferenceConfig, Parameter(help="Inference configuration for the model")
    ] = default_inference_config,
    output_path: Annotated[
        str, Parameter(help="Directory where prediction visualizations are saved")
    ] = ".data/predictions",
    dataset: Annotated[
        str,
        Parameter(help="Name of a public Hafnia sample dataset, or path to a local dataset on disk"),
    ] = "midwest-vehicle-detection",
    split_name: Annotated[str, Parameter(help="Dataset split to run prediction on")] = SplitName.TEST,
    samples: Annotated[int, Parameter(help="Number of samples to predict and visualize")] = 10,
):
    """Run model prediction on a small subset of a Hafnia dataset and save the rendered visualizations.

    This script is intended for **local execution only** as a quick visual sanity-check of a model.

    (It is deliberately restricted to local execution and to the small public sample datasets or a
    user-supplied local dataset. Users should not extract images from hidden platform datasets.)

    The ``dataset`` argument is resolved as a local path when it points at an existing directory on
    disk (loaded via ``HafniaDataset.from_path``); otherwise it is treated as the name of a public
    sample dataset (loaded via ``HafniaDataset.from_name``). The script then selects a deterministic
    random subset of the chosen split, runs the model on each sample, draws the predicted bounding
    boxes onto the image and writes the resulting PNGs to ``output_path``.
    """
    if is_hafnia_cloud_job():
        raise RuntimeError(
            "visualize.py is intended for local execution only and must not be run on the Hafnia platform. "
            "Extracting images from the platform's hidden datasets is not permitted - use this script "
            "locally against a public sample dataset instead."
        )

    path_prediction_visualization = Path(output_path)
    path_prediction_visualization.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(dataset)
    if dataset_path.exists():
        hafnia_dataset = HafniaDataset.from_path(dataset_path)
    else:
        hafnia_dataset = HafniaDataset.from_name(dataset, version="latest")

    model = WrappedModel.load_model(model_path, inference_config=inference)
    model.optimize_for_inference()

    dataset_split = hafnia_dataset.create_split_dataset(split_name=split_name)

    test_subset = dataset_split.select_samples(n_samples=samples, seed=42)
    for i_sample, dict_sample in enumerate(progress_bar(test_subset, description="Visualizing predictions")):
        sample = Sample(**dict_sample)

        image = sample.read_image()
        predictions = model.predict(image)
        annotations_visualized = image_visualizations.draw_annotations(image=image, primitives=predictions)
        path_visualization = path_prediction_visualization / f"prediction_visualization_{i_sample}.png"
        Image.fromarray(annotations_visualized).save(path_visualization)


if __name__ == "__main__":
    app()
