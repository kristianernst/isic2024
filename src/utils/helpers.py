import os
from typing import Tuple
import logging

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  

def get_data(input_path: str, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch data from the given path"""
    train_path = os.path.join(input_path, f"{dataset}.csv")
    train = pd.read_csv(train_path)
    test = None  # TODO: add test data.
    return train, test


def get_device() -> str:
    """
    Determine the best available device for PyTorch operations.

    Returns:
        str: A string indicating the device to be used:
            - "mps" for Apple Silicon GPUs (if available)
            - "cuda" for NVIDIA GPUs (if available)
            - "cpu" if no GPU is available

    Note:
        This function checks for MPS (Metal Performance Shaders) availability first,
        which is specific to Apple Silicon. If MPS is not available, it checks for CUDA.
        If neither GPU option is available, it defaults to CPU.
    """
    if torch.backends.mps.is_available():
        logger.info("Using device: mps"); return "mps"
    elif torch.cuda.is_available():
        logger.info("Using device: cuda"); return "cuda"
    else:
        logger.info("Using device: cpu"); return "cpu"
