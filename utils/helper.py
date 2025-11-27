# utils/helper.py
import yaml
import torch
import shutil
import logging
import random
import numpy as np

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


def set_seed(seed: int) -> None:
    """
    Set random seed for Python, NumPy and PyTorch to improve reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Optional, but usually helpful for reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_device(cuda: bool, device: int) -> torch.device:
    """
    Select computation device.

    Args:
        cuda: Whether to use CUDA if available.
        device: GPU index when cuda is True.

    Returns:
        torch.device: The selected device.
    """
    if cuda and torch.cuda.is_available():
        torch.cuda.set_device(device=device)
        return torch.device("cuda", device)
    return torch.device("cpu")


def load_config(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load a YAML config file and merge it into the args dict.

    CLI args stay in 'args', and top-level keys from the YAML
    are copied into args, possibly overriding existing keys
    with the same name.
    """
    config_path = args["config"]
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream) or {}

    # Shallow merge: top-level keys from YAML overwrite/extend args
    for k, v in config.items():
        args[k] = v
    return args


def save_config(args: Dict[str, Any], saving_path: str) -> None:
    """
    Save the full args dict (including merged config) as config.yaml.
    """
    saving_dir = Path(saving_path)
    saving_dir.mkdir(parents=True, exist_ok=True)
    config_file = saving_dir / "config.yaml"

    with config_file.open("w") as f:
        yaml.safe_dump(args, f, sort_keys=False)


def get_dir_path(args: Dict[str, Any], create_dir: bool = True) -> Tuple[str, str]:
    """
    Build a directory path for the current run.

    Directory structure:
        log_dir / dataset / mm_dd / (model + "_HH_MM_SS")

    Returns:
        dir_path (str): full directory path to create/use.
        dir_name (str): short name used for logging / run id.
    """
    model = args["model"]["name"]
    dataset = args["data"]["name"]
    base_path = args["log"]["log_dir"]

    now = datetime.now()
    date_str = now.strftime("%m_%d")
    time_str = now.strftime("_%H_%M_%S")

    dir_name = f"{date_str}_{model}{time_str}"
    dir_path = Path(base_path) / dataset / date_str / f"{model}{time_str}"

    if create_dir:
        dir_path.mkdir(parents=True, exist_ok=True)

    return str(dir_path), dir_name


def set_up_logger(args: Dict[str, Any]) -> Tuple[str, str]:
    """
    Initialize logging to both a file (train.log) and the console.

    Returns:
        log_dir (str): directory where logs are saved.
        dir_name (str): short name for this run.
    """
    log_dir, dir_name = get_dir_path(args)
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / "train.log"

    # Get root logger and reset handlers to avoid duplicate logs
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Saving logs in: %s", log_dir)

    return log_dir, dir_name


def save_code(
    module: Any,
    saving_path: str,
    with_dir: bool = False,
    with_path: bool = False,
) -> None:
    """
    Save source code of a given module (file or directory) into saving_path/code.

    Args:
        module: Python module object OR string path. When with_path=False,
                module is expected to be a module with a __file__ attribute.
        saving_path: Base directory where 'code' folder will be created.
        with_dir: If True, copy the entire directory containing module.__file__;
                  if False, copy only module.__file__.
        with_path: If True, 'module' is treated as a direct path (string or Path).
    """
    code_root = Path(saving_path) / "code"
    code_root.mkdir(parents=True, exist_ok=True)

    # Resolve source path
    if with_path:
        src = Path(str(module))
    else:
        if not hasattr(module, "__file__"):
            raise ValueError("module must have __file__ attribute when with_path=False.")
        src = Path(module.__file__)
        if with_dir:
            src = src.parent

    dst = code_root / src.name

    if src.is_dir():
        # If destination already exists, remove it to avoid copytree errors
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
