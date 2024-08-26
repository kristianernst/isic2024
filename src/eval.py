import argparse
import os
import textwrap
from typing import List, Type, Union

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from src.data.dataset import ISICDataset, transform_images
from src.model.baseline import AlwaysBenign, IdiotPredictor
from src.model.cnn_predictor import BaseCNNPredictor
from src.model.vit import LargeFormer, MegaFormer, SmallFormer
from src.utils.helpers import get_data, get_device
from src.utils.seed import set_seed


def load_model(model_path: str, model_class: Type[torch.nn.Module], device: str) -> torch.nn.Module:
    """
    Load a trained model from a file and prepare it for evaluation.

    Args:
        model_path (str): Path to the saved model file.
        model_class (Type[torch.nn.Module]): The class of the model to be loaded.
        device (str): The device to load the model onto ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded model, ready for evaluation.
    """
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_validation_data(fold: int, data_tuple: tuple) -> pd.DataFrame:
    """
    Extract validation data for a specific fold from the training data tuple.

    Args:
        fold (int): The fold number to use for validation.
        data_tuple (tuple): A tuple containing (pd_train, pd_test) DataFrames.

    Returns:
        pd.DataFrame: A DataFrame containing only the validation data for the specified fold.
    """
    print(f"type of fold: {type(fold)}")
    pd_train, _ = data_tuple  # Unpack the tuple, ignoring pd_test
    return pd_train[pd_train["fold"] == fold]


def plot_predictions(model: torch.nn.Module, val_loader: DataLoader, device: str, output_dir: str) -> None:
    """
    Plot model predictions on validation data and save individual images.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (str): Device to run the model on ('cpu' or 'cuda').
        output_dir (str): Directory to save the output images.

    Returns:
        None
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except IOError:
        font = ImageFont.load_default()

    with torch.no_grad():
        for batch_idx, (images, labels, img_paths) in enumerate(val_loader):
            images = images.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()

            for i, (img, label, pred, img_path) in enumerate(zip(images.cpu(), labels, preds.cpu(), img_paths)):
                img_pil = Image.fromarray((img.permute(1, 2, 0).numpy()).astype("uint8"))

                # Calculate the height needed for text
                true_label = "Malignant" if label.item() == 1 else "Benign"
                pred_label = "Malignant" if pred.item() == 1 else "Benign"
                text = f"True: {true_label}\nPred: {pred_label}"  # \nPath: {img_path}"

                # Wrap long lines
                wrapped_text = textwrap.fill(text, width=50)
                text_lines = wrapped_text.count("\n") + 1
                text_height = text_lines * 15  # Approximate height per line

                img_with_text = Image.new("RGB", (img_pil.width, img_pil.height + text_height + 10), color="white")
                img_with_text.paste(img_pil, (0, 0))

                draw = ImageDraw.Draw(img_with_text)

                # Add text to the image
                draw.text((10, img_pil.height + 5), wrapped_text, font=font, fill=(0, 0, 0))

                # Draw a line to separate the image from the text
                draw.line([(0, img_pil.height), (img_pil.width, img_pil.height)], fill=(0, 0, 0), width=1)

                output_filename = f"img_{batch_idx}_{i}_{os.path.basename(img_path)}"
                output_path = os.path.join(output_dir, output_filename)
                img_with_text.save(output_path)

    print(f"Saved {len(val_loader.dataset)} images to {output_dir}")


def debug_dataset(dataset: ISICDataset, num_samples: int = 5, output_dir: str = "debug_output"):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(min(num_samples, len(dataset))):
        image, label, image_path = dataset[i]

        # Save original image (before any transformations)
        original_img = cv2.imread(image_path)
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        Image.fromarray(original_img_rgb).save(os.path.join(output_dir, f"debug_original_{i}.png"))

        # Save transformed image
        if isinstance(image, torch.Tensor):
            transformed_img = Image.fromarray((image.permute(1, 2, 0).numpy()).astype("uint8"))
        else:
            transformed_img = Image.fromarray((image).astype("uint8"))
        transformed_img.save(os.path.join(output_dir, f"debug_transformed_{i}.png"))

        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(original_img_rgb)
        ax1.set_title("Original")
        ax1.axis("off")
        ax2.imshow(transformed_img)
        ax2.set_title("Transformed")
        ax2.axis("off")
        plt.suptitle(f"Sample {i}, Label: {label}")  # Path: {image_path}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"debug_comparison_{i}.png"))
        plt.close()

    print(f"Debug images saved to {output_dir}")


def evaluate(
    model_class: Union[Type[BaseCNNPredictor], Type[SmallFormer], Type[LargeFormer], Type[MegaFormer], Type[IdiotPredictor], Type[AlwaysBenign]],
    data_tuple: tuple,
    args: argparse.Namespace,
) -> None:
    """
    Evaluate a trained model on the validation set and save prediction images.

    Args:
        model_class (Union[Type[BaseCNNPredictor], Type[SmallFormer], Type[LargeFormer], Type[MegaFormer], Type[IdiotPredictor], Type[AlwaysBenign]]): Class of the model to be evaluated.
        data_tuple (tuple): A tuple containing (pd_train, pd_test) DataFrames.
        args (argparse.Namespace): Namespace containing runtime arguments.

    Returns:
        None
    """
    device = get_device()
    model = load_model(args.model_path, model_class, device)
    val_data = get_validation_data(args.fold, data_tuple)

    _, transform_valid = transform_images(args.img_size)
    val_dataset: ISICDataset = ISICDataset(val_data, transform=transform_valid)

    # Debug the dataset
    debug_dataset(val_dataset, num_samples=10, output_dir="eval_debug_output")

    val_loader: DataLoader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    output_dir: str = f"out_images/validation_predictions_fold_{args.fold}"
    plot_predictions(model, val_loader, device, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--model_path", type=str, default="./output/best_acc_0.8595106550907656_model_LargeFormer_epoch_6_fold_1.pt", help="Path to the trained model"
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="LargeFormer",
        choices=["BaseCNNPredictor", "SmallFormer", "LargeFormer", "MegaFormer", "IdiotPredictor", "AlwaysBenign"],
        help="Model class to use",
    )
    parser.add_argument("--fold", type=int, required=True, help="Fold number used for validation")
    parser.add_argument("--data_path", type=str, default="./kaggle/input", help="Path to the data directory")
    parser.add_argument("--dataset", type=str, default="mix5050", help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(42)  # Set a fixed seed for reproducibility

    # Load the training data
    data_tuple = get_data(input_path=args.data_path, dataset=args.dataset)

    # Map model class string to actual class
    model_class_map = {
        "BaseCNNPredictor": BaseCNNPredictor,
        "SmallFormer": SmallFormer,
        "LargeFormer": LargeFormer,
        "MegaFormer": MegaFormer,
        "IdiotPredictor": IdiotPredictor,
        "AlwaysBenign": AlwaysBenign,
    }
    model_class = model_class_map[args.model_class]

    evaluate(model_class, data_tuple, args)
