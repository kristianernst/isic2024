import os

import torch


def check_file_exists(file_path: str, file_name: str) -> bool:
    """
    Check if the file exists in the given path.
    """
    files = os.listdir(file_path)
    files = [f.split(".")[0] for f in files]
    return file_name in files


def save_model(modelDict: torch.nn.Module.state_dict, file_path: str, file_name: str):
    """
    Save the model to the given path with the given name.
    If the file already exists, create a new file with a different name.
    """

    # Create the directory if it does not exist
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)

    vinc = 0

    # Remove the .pt extension if it exists in the file_name
    file_name = file_name.rstrip(".pt")

    while check_file_exists(file_path, file_name):
        file_name = f"{file_name}_v{vinc}"
        vinc += 1

    file_path = os.path.join(file_path, f"{file_name}.pt")
    torch.save(modelDict, file_path)
