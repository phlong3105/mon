from pathlib import Path

import torch
import torchvision
import torchvision.transforms.functional
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from hafnia.dataset import torch_helpers
from hafnia.dataset.hafnia_dataset import HafniaDataset

if __name__ == "__main__":
    torch.manual_seed(1)
    # Load Hugging Face dataset
    MIDWEST_VERSION = "1.0.0"
    dataset = HafniaDataset.from_name("midwest-vehicle-detection", version=MIDWEST_VERSION)

    # Define transforms
    train_transforms = v2.Compose(
        [
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transforms = v2.Compose(
        [
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset_train = dataset.create_split_dataset("train")
    dataset_test = dataset.create_split_dataset("test")

    keep_metadata = True
    train_dataset = torch_helpers.TorchvisionDataset(
        dataset_train, transforms=train_transforms, keep_metadata=keep_metadata
    )
    test_dataset = torch_helpers.TorchvisionDataset(
        dataset_test, transforms=test_transforms, keep_metadata=keep_metadata
    )

    # Visualize sample
    image, targets = train_dataset[0]
    visualize_image = torch_helpers.draw_image_and_targets(image=image, targets=targets)
    pil_image = torchvision.transforms.functional.to_pil_image(visualize_image)

    path_tmp = Path(".data/tmp")
    path_tmp.mkdir(parents=True, exist_ok=True)
    pil_image.save(path_tmp / "visualized_labels.png")

    # Create DataLoaders - using TorchVisionCollateFn
    collate_fn = torch_helpers.TorchVisionCollateFn()
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)

    for images, targets in train_loader:
        print(f"Batch of images: {len(images)}")
        print(f"Batch of targets: {len(targets)}")
        break
