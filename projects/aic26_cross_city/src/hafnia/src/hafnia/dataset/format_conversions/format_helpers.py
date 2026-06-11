from dataclasses import dataclass
from pathlib import Path
from typing import List

from hafnia.dataset.dataset_names import SplitName
from hafnia.log import user_logger


@dataclass
class SplitNameAndPath:
    name: str
    path: Path

    def check(self) -> None:
        if not self.path.is_dir():
            raise ValueError(f"Path '{self.path}' is not a valid directory.")

        if self.name not in SplitName.valid_splits():
            raise ValueError(f"Split name '{self.name}' is not a valid split name.")


def get_splits_from_folder(path_folder: Path) -> List[SplitNameAndPath]:
    split_name_and_paths = []
    # Sort to ensure deterministic order across different filesystems and OS
    for path_sub_folder in sorted(path_folder.iterdir()):
        if not path_sub_folder.is_dir():
            continue
        folder_split_name = path_sub_folder.name
        split_name = SplitName.map_split_name(folder_split_name, strict=False)
        if split_name == SplitName.UNDEFINED:
            user_logger.warning(f"Skipping sub-folder with name '{folder_split_name}'")
            continue
        split_name_and_paths.append(SplitNameAndPath(name=split_name, path=path_sub_folder))
    return split_name_and_paths
