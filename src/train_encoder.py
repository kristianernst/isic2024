import argparse
import multiprocessing
import gc
import logging
from typing import Tuple, Union

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import wandb
import pandas as pd
from tqdm import tqdm

from src.utils.helpers import get_device, get_data
from src.model.contrastive_learners import ContrastiveLearner, NTXentLoss
from src.data.dataset import ContrastiveISICDataset, transform_images_encoder



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name=__name__)

def train_epoch(model: ContrastiveLearner, dataloader: DataLoader, criterion: NTXentLoss, optimizer: optim.Adam, device: str, epoch: int, warmup_epochs: int, warmup_lr: float, base_lr: float) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The ContrastiveLearner model.
        dataloader: DataLoader for training data.
        criterion: Loss function (NTXentLoss).
        optimizer: Optimizer for model parameters.
        device: Device to run computations on.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    
    tot_batches = len(dataloader) * warmup_epochs
    
    
    for i, (x1, x2, _) in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch + 1} ...")):
        x1, x2 = x1.to(device), x2.to(device)

        logger.debug(f"Batch {i+1}: x1 shape: {x1.shape}, x2 shape: {x2.shape}")
 
        
        z_i: torch.Tensor = model(x1)
        z_j: torch.Tensor = model(x2)
        
        # correct lr for warmup state, if warmup_epochs > 0, smoothly increase lr from warmup_lr to base_lr
        if epoch < warmup_epochs:
            warmup_progress = (epoch * len(dataloader) + i + 1) / tot_batches
            lr = warmup_lr + (base_lr - warmup_lr) * warmup_progress
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        optimizer.zero_grad()
        
        loss: torch.Tensor = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate_epoch(model: ContrastiveLearner, dataloader: DataLoader, criterion: NTXentLoss, device: str) -> float:
    """
    Validate the model for one epoch.

    Args:
        model: The ContrastiveLearner model.
        dataloader: DataLoader for validation data.
        criterion: Loss function (NTXentLoss).
        device: Device to run computations on.

    Returns:
        Average loss for the validation epoch.
    """
    model.eval()
    running_loss = 0.0
    
    with torch.inference_mode():
        for x1, x2, _ in tqdm(dataloader, desc="Validating epoch ..."):
            x1, x2 = x1.to(device), x2.to(device)
            
            z_i: torch.Tensor = model(x1)
            z_j: torch.Tensor = model(x2)
            
            loss: torch.Tensor = criterion(z_i, z_j)
            running_loss += loss.item()
    return running_loss / len(dataloader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./kaggle/input")
    parser.add_argument("--dataset", type=str, default="train_metadata_combined")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup_lr", type=float, default=0.000001)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--encoder_name", type=str, default="mobilenet_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--transform_p", type=float, default=0.5)
    # parser.add_argument("--weights", type=str, default=None)
    
    return parser.parse_args()
    
    

def train():
    """
    Run the training loop for the contrastive learner.
    """
    # Configuration
    args = parse_args()
    
    # Initialize Weights & Biases
    num_workers = multiprocessing.cpu_count() if args.num_workers == 0 else args.num_workers
    wandb.init(project='contrastive-learning', config=args)
    device = get_device()
    
    pd_train, _ = get_data(input_path=args.data_path, dataset=args.dataset)    
    transform_train, transform_valid = transform_images_encoder(args.img_size, p=args.transform_p)
    
    train_dataset = ContrastiveISICDataset(pd_train, transform=transform_train)
    val_dataset = ContrastiveISICDataset(pd_train, transform=transform_valid)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    
    # Initialize Model, Loss, and Optimizer
    model = ContrastiveLearner(encoder_name=args.encoder_name).to(device)
    criterion = NTXentLoss(batch_size=args.batch_size, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    
    best_val_loss = float("inf")
    # Training Loop
    for epoch in tqdm(range(args.num_epochs), desc="Training epochs"):     
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, warmup_epochs=args.warmup_epochs, warmup_lr=args.warmup_lr, base_lr=args.lr)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'learning_rate': optimizer.param_groups[0]['lr']})
        logger.info(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"encoder_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pt")
    
    # Finalize Weights & Biases
    wandb.finish()
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()