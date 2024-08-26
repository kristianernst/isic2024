import os
from typing import Optional, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class ISICDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: Optional[Compose] = None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.dataframe.iloc[idx]
        image_path = row["image_path"]

        image: np.ndarray = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        if self.transform:
            res = self.transform(image=image)
            if isinstance(res["image"], np.ndarray):
                # If the result is still a numpy array, convert it to a tensor
                image = torch.from_numpy(res["image"].transpose(2, 0, 1)).float()
            else:
                # If it's already a tensor, just ensure it's float
                image = res["image"].float()
        else:
            # If no transform, convert numpy array to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Ensure the image is in the correct format (C, H, W)
        if image.shape[0] != 3:
            image = image.permute(2, 0, 1)

        data = image.clone().detach().to(torch.float32)
        return data, torch.tensor(row.target, dtype=torch.long), image_path

class ContrastiveISICDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Dataset class for contrastive learning. Generates two augmented versions of each image.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the image paths and labels.
            transform (albumentations.Compose): Albumentations transformations to apply to the images.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Returns two augmented versions of the same image and the corresponding label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            image1 (torch.Tensor): First augmented version of the image.
            image2 (torch.Tensor): Second augmented version of the image.
            label (int): The label of the image.
        """
        row = self.dataframe.iloc[idx]
        image_path = row["image_path"]
        
        # Read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply the transformation twice to create two augmented versions
        if self.transform:
            image1 = self.transform(image=image)["image"]
            image2 = self.transform(image=image)["image"]
        else:
            image1 = torch.from_numpy(image.transpose(2, 0, 1)).float()
            image2 = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image1, image2, row["target"]
    

def prettyprint_dataset(dataset: ISICDataset):
    print(f"Number of samples: {len(dataset)}")  # this is printed
    # print(f"Tensor shape: {dataset[0][0].shape}")

    # check tensor shape is the same for all samples
    same = True
    for i in range(len(dataset) - 1):  # no prints in this loop
        print(f"i: {i}")
        if dataset[i][0].shape == dataset[i + 1][0].shape:
            print(f"Same shape")
            break
        else:
            same = False
            print(f"tensor1: {dataset[i][0].shape}")
            print(f"tensor2: {dataset[i + 1][0].shape}")
            break

    if same:
        print("All tensors have the same shape")


class Vignette(ImageOnlyTransform):
    def __init__(self, always_apply=False, p: float = 0.5):
        super(Vignette, self).__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        rows, cols = img.shape[:2]
        center = (cols // 2, rows // 2)
        radius = min(center[0], center[1], cols - center[0], rows - center[1])

        # Create a mask with a white circle in the center
        mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)

        # Apply Gaussian blur to the mask
        mask = cv2.GaussianBlur(src=mask, ksize=(7, 7), sigmaX=50)
        # mask = cv2.GaussianBlur(src=mask, ksize=(3, 3), sigmaX=50, sigmaY=50)

        # Create a 3-channel version of the mask
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Normalize the mask to the range [0, 1]
        mask = mask / 255.0

        # Apply the mask to the image
        vignette = img * mask + (1 - mask) * 0

        return vignette.astype(np.uint8)


def display_vignette_effect(image_path: str):
    # Read the original image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply vignette transform
    vignette_transform = Vignette(p=1.0)
    transformed = vignette_transform(image=image_rgb)

    transformed = transformed["image"]
    # Ensure the transformed image is a valid type for imshow
    if transformed.dtype != np.uint8:
        transformed = transformed.astype(np.uint8)

    # Display the original and vignette images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(transformed)
    axes[1].set_title("Vignette Effect")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def transform_images(image_size: int = 128, p: float = 0.0) -> Tuple[A.Compose, A.Compose]:
    """Function to transform images using albumentations library while training and validating the model, only transformation done for validation is image resizing"""
    transform_train = A.Compose([
        A.Transpose(p=p),  # 0.5
        A.VerticalFlip(p=p),  # 0.5
        A.HorizontalFlip(p=p),  # 0.5
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p),  # 0.7
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 5)),
                A.MedianBlur(blur_limit=(3, 5)),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ],
            p=p,  # 0.7
        ),
        A.OneOf(
            [
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.0),
                A.ElasticTransform(alpha=3),
            ],
            p=p,  # 0.7
        ),
        A.CLAHE(clip_limit=4.0, p=p),  # 0.7
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=p),  # 0.5
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=p),  # 0.85
        A.Resize(
            image_size, image_size, p=1.0, interpolation=cv2.INTER_CUBIC
        ),  # resize image, inter cv2.INTER_CUBIC: Bicubic interpolation over 4x4 pixel neighborhood, providing higher quality but at a computational cost.cv2.INTER_LANCZOS4: Lanczos interpolation over 8x8 pixel neighborhood, often used for high-quality image resizing.
        Vignette(p=p),  # 0.3
        A.Normalize(),
        ToTensorV2(),
    ])

    transform_valid = A.Compose([
        A.Resize(image_size, image_size, p=1.0, interpolation=cv2.INTER_CUBIC),
        ToTensorV2(),
    ])

    return transform_train, transform_valid


def transform_images_encoder(image_size: int = 128, p: float = 0.5) -> Tuple[A.Compose, A.Compose]:
    """Function to transform images using the albumentations library while training and validating the model."""
    
    # Transformation pipeline for training (with heavier augmentations)
    transform_train = A.Compose([
        A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=p),  # Flipping images horizontally
        A.VerticalFlip(p=p),    # Flipping images vertically
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=p),  # Shifting, scaling, rotating
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p),  # Random brightness and contrast adjustment
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=p),  # Slight color jitter
        A.GaussianBlur(blur_limit=(3, 5), p=p),  # Applying Gaussian blur
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=p),  # Hue, saturation, and value shifting
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=p),  # Applying CLAHE (Contrast Limited Adaptive Histogram Equalization)
        A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC, p=1.0),  # Resize to the desired image size
        Vignette(p=0.1),  # Create vignette effect
        A.Normalize(),  # Normalize image to standard values
        ToTensorV2(),  # Convert image to PyTorch tensor
    ])
    
    # Transformation pipeline for validation (lighter augmentations)
    transform_valid = A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC, p=1.0),  # Resize to the desired image size
        A.Normalize(),  # Normalize image to standard values
        ToTensorV2(),  # Convert image to PyTorch tensor
    ])

    return transform_train, transform_valid


if __name__ == "__main__":
    IMG_DIMENSION = 128  # (64, 64, 3)
    df = pd.read_csv("../../kaggle/input/train_metadata_combined.csv")
    print(df.head())
    transform_train, _ = transform_images(IMG_DIMENSION)
    dataset = ISICDataset(df, transform=transform_train)
    # prettyprint_dataset(dataset) # takes forever to run

    # display vignette effect:
    sample_image_path = "../" + df.iloc[0, 2]
    display_vignette_effect(sample_image_path)
