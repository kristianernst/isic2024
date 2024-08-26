import numpy as np
import torch
import torch.backends
import torch.backends.mps


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across different components.

    This function sets the random seed for NumPy, PyTorch CPU, CUDA (if available),
    and MPS (if available) to ensure consistent random number generation.

    Args:
        seed (int): The random seed to set. Default is 42.

    Note:
        - For CUDA, it also sets the `cudnn.deterministic` flag to True for full reproducibility.
        - The function prints a confirmation message after setting the seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    print(f"Seed set successfully to {seed}")
