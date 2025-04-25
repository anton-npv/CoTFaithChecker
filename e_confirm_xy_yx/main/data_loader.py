import json
from pathlib import Path
from typing import Dict, List
from e_confirm_xy_yx.main.logging_utils import init_logger

__all__ = [
    "get_dataset_files",
    "load_dataset",
]

logger = init_logger(log_dir=Path.cwd() / "logs", script_name="data_loader")


def get_dataset_files(
    questions_root: Path,
    dataset_folders: List[str],
) -> List[Path]:
    """
    Return every *.json file inside the requested folder names.
    Example: dataset_folders = ["gt_NO_1", "gt_YES_1"].
    """
    files: List[Path] = []
    for folder in dataset_folders:
        folder_path = questions_root / folder
        current_files = sorted(folder_path.glob("*.json"))
        logger.info(f"Found {len(current_files)} files in {folder_path}")
        files.extend(current_files)
    logger.info(f"Total files collected: {len(files)}")
    return files


def load_dataset(json_path: Path) -> Dict:
    """
    Load a single JSON dataset file and return it as a dict.
    """
    logger.debug(f"Loading {json_path}")
    with json_path.open() as f:
        data = json.load(f)
    return data
