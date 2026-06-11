import inspect
import os
import shutil
import tempfile
import textwrap
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from torchvision import datasets as tv_datasets
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, extract_archive

from hafnia import utils
from hafnia.dataset.dataset_helpers import save_pil_image_with_hash_name
from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.format_conversions.format_image_classification_folder import (
    from_image_classification_split_folder,
)
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, Sample, TaskInfo
from hafnia.dataset.primitives import Classification


def torchvision_to_hafnia_converters() -> Dict[str, Callable]:
    return {
        "mnist": mnist_as_hafnia_dataset,
        "cifar10": cifar10_as_hafnia_dataset,
        "cifar100": cifar100_as_hafnia_dataset,
        "caltech-101": caltech_101_as_hafnia_dataset,
        "caltech-256": caltech_256_as_hafnia_dataset,
    }


def mnist_as_hafnia_dataset(force_redownload=False, n_samples: Optional[int] = None) -> HafniaDataset:
    samples, tasks = torchvision_basic_image_classification_dataset_as_hafnia_dataset(
        dataset_loader=tv_datasets.MNIST,
        force_redownload=force_redownload,
        n_samples=n_samples,
    )

    dataset_info = DatasetInfo(
        dataset_name="mnist",
        version="1.0.0",
        tasks=tasks,
        reference_bibtex=textwrap.dedent("""\
            @article{lecun2010mnist,
              title={MNIST handwritten digit database},
              author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
              journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
              volume={2},
              year={2010}
            }"""),
        reference_paper_url=None,
        reference_dataset_page="http://yann.lecun.com/exdb/mnist",
    )
    return HafniaDataset.from_samples_list(samples_list=samples, info=dataset_info)


def cifar10_as_hafnia_dataset(force_redownload: bool = False, n_samples: Optional[int] = None) -> HafniaDataset:
    return cifar_as_hafnia_dataset(dataset_name="cifar10", force_redownload=force_redownload, n_samples=n_samples)


def cifar100_as_hafnia_dataset(force_redownload: bool = False, n_samples: Optional[int] = None) -> HafniaDataset:
    return cifar_as_hafnia_dataset(dataset_name="cifar100", force_redownload=force_redownload, n_samples=n_samples)


def caltech_101_as_hafnia_dataset(
    force_redownload: bool = False,
    n_samples: Optional[int] = None,
) -> HafniaDataset:
    dataset_name = "caltech-101"
    path_image_classification_folder = _download_and_extract_caltech_dataset(
        dataset_name, force_redownload=force_redownload
    )
    hafnia_dataset = from_image_classification_split_folder(
        path_image_classification_folder,
        split=SplitName.TRAIN,
        n_samples=n_samples,
        dataset_name=dataset_name,
    )
    hafnia_dataset.info.version = "1.0.0"
    hafnia_dataset.info.reference_bibtex = textwrap.dedent("""\
        @article{FeiFei2004LearningGV,
            title={Learning Generative Visual Models from Few Training Examples: An Incremental Bayesian
                   Approach Tested on 101 Object Categories},
            author={Li Fei-Fei and Rob Fergus and Pietro Perona},
            journal={Computer Vision and Pattern Recognition Workshop},
            year={2004},
        }
        """)
    hafnia_dataset.info.reference_dataset_page = "https://data.caltech.edu/records/mzrjq-6wc02"

    return hafnia_dataset


def caltech_256_as_hafnia_dataset(
    force_redownload: bool = False,
    n_samples: Optional[int] = None,
) -> HafniaDataset:
    dataset_name = "caltech-256"

    path_image_classification_folder = _download_and_extract_caltech_dataset(
        dataset_name, force_redownload=force_redownload
    )
    hafnia_dataset = from_image_classification_split_folder(
        path_image_classification_folder,
        split=SplitName.TRAIN,
        n_samples=n_samples,
        dataset_name=dataset_name,
    )
    hafnia_dataset.info.version = "1.0.0"
    hafnia_dataset.info.reference_bibtex = textwrap.dedent("""\
        @misc{griffin_2023_5sv1j-ytw97,
            author       = {Griffin, Gregory and
                            Holub, Alex and
                            Perona, Pietro},
            title        = {Caltech-256 Object Category Dataset},
            month        = aug,
            year         = 2023,
            publisher    = {California Institute of Technology},
            version      = {public},
        }""")
    hafnia_dataset.info.reference_dataset_page = "https://data.caltech.edu/records/nyy15-4j048"

    task = hafnia_dataset.info.get_task_by_primitive(Classification)

    # Class Mapping: To remove numeric prefixes from class names
    # E.g. "001.ak47 --> ak47", "002.american-flag --> american-flag", ...
    class_mapping = {name: name.split(".")[-1] for name in task.get_class_names() or []}
    hafnia_dataset = hafnia_dataset.class_mapper(class_mapping=class_mapping, task_name=task.name)
    return hafnia_dataset


def cifar_as_hafnia_dataset(
    dataset_name: str,
    force_redownload: bool = False,
    n_samples: Optional[int] = None,
) -> HafniaDataset:
    if dataset_name == "cifar10":
        dataset_loader = tv_datasets.CIFAR10
    elif dataset_name == "cifar100":
        dataset_loader = tv_datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Supported: cifar10, cifar100")
    samples, tasks = torchvision_basic_image_classification_dataset_as_hafnia_dataset(
        dataset_loader=dataset_loader,
        force_redownload=force_redownload,
        n_samples=n_samples,
    )

    dataset_info = DatasetInfo(
        dataset_name=dataset_name,
        version="1.0.0",
        tasks=tasks,
        reference_bibtex=textwrap.dedent("""\
        @@TECHREPORT{Krizhevsky09learningmultiple,
            author = {Alex Krizhevsky},
            title = {Learning multiple layers of features from tiny images},
            institution = {},
            year = {2009}
        }"""),
        reference_paper_url="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf",
        reference_dataset_page="https://www.cs.toronto.edu/~kriz/cifar.html",
    )

    return HafniaDataset.from_samples_list(samples_list=samples, info=dataset_info)


def torchvision_basic_image_classification_dataset_as_hafnia_dataset(
    dataset_loader: VisionDataset,
    force_redownload: bool = False,
    n_samples: Optional[int] = None,
) -> Tuple[List[Sample], List[TaskInfo]]:
    """
    Converts a certain group of torchvision-based image classification datasets to a Hafnia Dataset.

    This conversion only works for certain group of image classification VisionDataset by torchvision.
    Common for these datasets is:
    1) They provide a 'class_to_idx' mapping,
    2) A "train" boolean parameter in the init function to separate training and test data - thus no validation split
       is available for these datasets,
    3) Datasets are in-memory and not on disk
    4) Samples consist of a PIL image and a class index.

    """
    torchvision_dataset_name = dataset_loader.__name__

    # Check if loader has train-parameter using inspect module
    params = inspect.signature(dataset_loader).parameters

    has_train_param = ("train" in params) and (params["train"].annotation is bool)
    if not has_train_param:
        raise ValueError(
            f"The dataset loader '{dataset_loader.__name__}' does not have a 'train: bool' parameter in the init "
            "function. This is a sign that the wrong dataset loader is being used. This conversion function only "
            "works for certain image classification datasets provided by torchvision that are similar to e.g. "
            "MNIST, CIFAR-10, CIFAR-100"
        )

    path_torchvision_dataset = utils.get_path_torchvision_downloads() / torchvision_dataset_name
    path_hafnia_conversions = utils.get_path_hafnia_conversions() / torchvision_dataset_name

    if force_redownload:
        shutil.rmtree(path_torchvision_dataset, ignore_errors=True)
        shutil.rmtree(path_hafnia_conversions, ignore_errors=True)

    splits = {
        SplitName.TRAIN: dataset_loader(root=path_torchvision_dataset, train=True, download=True),
        SplitName.TEST: dataset_loader(root=path_torchvision_dataset, train=False, download=True),
    }

    samples = []
    n_samples_per_split = n_samples // len(splits) if n_samples is not None else None
    for split_name, torchvision_dataset in splits.items():
        class_name_to_index = torchvision_dataset.class_to_idx
        class_index_to_name = {v: k for k, v in class_name_to_index.items()}
        description = f"Convert '{torchvision_dataset_name}' ({split_name} split) to Hafnia Dataset "
        samples_in_split = []
        for image, class_idx in utils.progress_bar(
            torchvision_dataset, total=n_samples_per_split, description=description
        ):
            (width, height) = image.size
            path_image = save_pil_image_with_hash_name(image, path_hafnia_conversions / "data")
            sample = Sample(
                file_path=str(path_image),
                height=height,
                width=width,
                split=split_name,
                classifications=[
                    Classification(
                        class_name=class_index_to_name[class_idx],
                        class_idx=class_idx,
                    )
                ],
            )
            samples_in_split.append(sample)

            if n_samples_per_split is not None and len(samples_in_split) >= n_samples_per_split:
                break

        samples.extend(samples_in_split)
    class_names = list(class_name_to_index.keys())
    tasks = [TaskInfo.from_class_names(primitive=Classification, class_names=class_names)]

    return samples, tasks


def _download_and_extract_caltech_dataset(dataset_name: str, force_redownload: bool) -> Path:
    path_torchvision_dataset = utils.get_path_torchvision_downloads() / dataset_name

    if force_redownload:
        shutil.rmtree(path_torchvision_dataset, ignore_errors=True)

    if path_torchvision_dataset.exists():
        return path_torchvision_dataset

    with tempfile.TemporaryDirectory() as tmpdirname:
        path_tmp_output = Path(tmpdirname)
        path_tmp_output.mkdir(parents=True, exist_ok=True)

        if dataset_name == "caltech-101":
            download_and_extract_archive(
                "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip",
                download_root=path_tmp_output,
                filename="caltech-101.zip",
                md5="3138e1922a9193bfa496528edbbc45d0",
            )
            path_output_extracted = path_tmp_output / "caltech-101"
            for gzip_file in os.listdir(path_output_extracted):
                if gzip_file.endswith(".gz"):
                    extract_archive(
                        from_path=os.path.join(path_output_extracted, gzip_file),
                        to_path=path_output_extracted,
                    )
            path_org = path_output_extracted / "101_ObjectCategories"

        elif dataset_name == "caltech-256":
            org_dataset_name = "256_ObjectCategories"
            path_org = path_tmp_output / org_dataset_name
            download_and_extract_archive(
                url=f"https://data.caltech.edu/records/nyy15-4j048/files/{org_dataset_name}.tar",
                download_root=path_tmp_output,
                md5="67b4f42ca05d46448c6bb8ecd2220f6d",
                remove_finished=True,
            )

        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}. Supported: caltech-101, caltech-256")

        shutil.rmtree(path_torchvision_dataset, ignore_errors=True)
        shutil.move(path_org, path_torchvision_dataset)
    return path_torchvision_dataset
