import argparse
import gc
import logging
import multiprocessing
from typing import Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import wandb
from src.data.dataset import ISICDataset, transform_images
from src.model.baseline import AlwaysBenign, IdiotPredictor
from src.model.cnn_predictor import BaseCNNPredictor
from src.model.vit import LargeFormer, MegaFormer, SmallFormer
from src.utils.helpers import get_data, get_device
from src.utils.save import save_model
from src.utils.seed import set_seed

# torch.multiprocessing.set_sharing_strategy("file_system")
# torch.multiprocessing.set_sharing_strategy('file_descriptor') # reccomendation by chatgpt, according to docs, might be the way to go: https://pytorch.org/docs/stable/multiprocessing.html#file-descriptor-file-descriptor
# ulimit -n 4096 in terminal might also help.
# lower num workers will also help

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name=__name__)


def train_one_epoch(
    epoch: int,
    model: Union[BaseCNNPredictor, IdiotPredictor, SmallFormer],
    train_loader: DataLoader,
    criterion: nn.BCELoss,
    optimizer: optim.Adam,
    device: str,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau = None,
    warmup_epochs: int = 1,
    warmup_lr: float = 0.00001,
    base_lr: float = 0.001,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    tot_batches = len(train_loader) * warmup_epochs

    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
        # move data to device
        images: torch.Tensor = images.to(device)
        labels: torch.Tensor = labels.to(device)

        # correct lr for warmup state
        if epoch < warmup_epochs:
            warmup_progress = (epoch * len(train_loader) + i + 1) / tot_batches
            lr = warmup_lr + (base_lr - warmup_lr) * warmup_progress

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        optimizer.zero_grad()
        outputs = model.forward(images)

        # visualize the two tensors
        logger.debug(f"outputs: {outputs.shape}")
        logger.debug(f"labels: {labels.shape}")

        logger.debug(f"first 10 outputs: {outputs[:10]}")
        logger.debug(f"first 10 example: {labels[:10]}")

        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels.unsqueeze(1).float()).sum().item()
        total += labels.size(0)
        wandb.log({"granular train loss": loss.item()})

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    return train_loss, train_acc


def validate_one_epoch(
    epoch: int, model: BaseCNNPredictor, val_loader: DataLoader, criterion: nn.BCELoss, device: str, check_representation: bool = True
) -> Tuple[float, float]:
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
            images: torch.Tensor = images.to(device)
            labels: torch.Tensor = labels.to(device)
            outputs = model(images) if not isinstance(model, AlwaysBenign) else torch.zeros(labels.size(0), 1).to(device)

            if check_representation:
                logger.debug(f"\n\nlabels: {labels}")
                logger.debug(f"\n\noutputs: {outputs}")

            if not isinstance(model, AlwaysBenign):
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()

            predicted = (outputs > 0.5).float()
            correct += (predicted == labels.unsqueeze(1).float()).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    if not isinstance(model, AlwaysBenign):
        val_loss /= len(val_loader)
        return val_loss, val_acc
    else:
        return float("inf"), val_acc


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./kaggle/input")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--k_folds", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mix5050")
    parser.add_argument("--transform_p", type=float, default=0.0)
    parser.add_argument("--scheduler", type=int, default=0)
    parser.add_argument("--scheduler_intensity", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--warmup_lr", type=float, default=0.00001)
    parser.add_argument("--weights", type=str, default=None)
    return parser.parse_args()


def pretty_print_args(args):
    logger.info(f"{'=' * 20}")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info(f"{'=' * 20}")


def train():
    DEVICE = get_device()
    args = parse_args()
    num_workers = multiprocessing.cpu_count() if args.num_workers == 0 else args.num_workers
    pd_train, _ = get_data(args.data_path, dataset=args.dataset)

    logger.info(f"device: {DEVICE}")
    logger.info(f"num_workers: {num_workers}")

    set_seed(args.seed)

    # init wandb
    wandb.init(
        project="my-cancer-prediction-project",
        config=args,
    )
    # include to make sweeps easier, as wandb sweeps will override the args, therefore we need to update the args with the wandb config
    config = wandb.config
    args.lr = config.lr
    args.batch_size = config.batch_size
    args.num_epochs = config.num_epochs
    args.img_size = config.img_size
    args.model = config.model
    args.transform_p = config.transform_p
    args.scheduler = config.scheduler
    args.scheduler_intensity = config.scheduler_intensity
    args.warmup_epochs = config.warmup_epochs
    args.warmup_lr = config.warmup_lr
    args.weights = config.weights

    pretty_print_args(args)

    for fold in range(args.k_folds):
        logger.info(f"{'=' * 20} Fold {fold + 1} / {args.k_folds} {'=' * 20}")
        # init model
        model = (
            IdiotPredictor(img_size=args.img_size, num_classes=1)
            if args.model == "baseline"
            else AlwaysBenign()
            if args.model == "naive"
            else BaseCNNPredictor(num_classes=1, img_size=args.img_size)
            if args.model == "cnn"
            else LargeFormer(num_classes=1, pretrained_weights=args.weights)
            if args.model == "ViT"
            else MegaFormer(num_classes=1, dropout=0.3, pretrained_weights=args.weights)
            if args.model == "MegaFormer"
            else SmallFormer(num_classes=1)
        )
        logger.info(f"using model: {model.__class__.__name__}")
        model.to(DEVICE)

        # load data dependent on the fold
        train_data = pd_train[pd_train["fold"] != fold]
        val_data = pd_train[pd_train["fold"] == fold]

        transform_train, transform_valid = transform_images(args.img_size, p=args.transform_p)
        train_dataset = ISICDataset(train_data, transform=transform_train)
        val_dataset = ISICDataset(val_data, transform=transform_valid)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

        # init criterion
        criterion = nn.BCELoss()
        # init optimizer
        if not isinstance(model, AlwaysBenign):
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            if args.scheduler:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.scheduler_intensity, patience=1, min_lr=1e-10)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_runs_docs = {}

        for epoch in range(args.num_epochs):
            gc.collect()  # garbage collection
            logger.debug(f"{'=' * 10} Epoch {epoch + 1} / {args.num_epochs} {'=' * 10}")

            if isinstance(model, AlwaysBenign):
                logger.info("Skipping training for AlwaysBenign model, as it will always predict benign.")
            else:
                train_loss, train_acc = train_one_epoch(
                    epoch,
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    DEVICE,
                    scheduler,
                    warmup_epochs=args.warmup_epochs,
                    warmup_lr=args.warmup_lr,
                    base_lr=args.lr,
                )
                wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc})

            val_loss, val_acc = validate_one_epoch(epoch, model, val_loader, criterion, DEVICE)
            wandb.log({"val_loss": val_loss, "val_acc": val_acc}) if not isinstance(model, AlwaysBenign) else wandb.log({"val_acc": val_acc})

            if not isinstance(model, AlwaysBenign) and epoch < args.num_epochs - 1:
                if epoch >= args.warmup_epochs and args.scheduler:
                    scheduler.step(val_loss)
                    # logger.info(f"Updating learning rate to: {optimizer.param_groups[0]['lr']}")

                # save best model
                if val_acc > best_val_acc:
                    best_runs_docs = {f"epoch {epoch + 1} fold {fold}": {"val_loss": val_loss, "val_acc": val_acc}}
                    best_val_acc = val_acc
                    save_model(model.state_dict(), "output", f"best_acc_{val_acc}_model_{model.__class__.__name__}_epoch_{epoch + 1}_fold_{fold}.pt")

    logger.info("Training complete.")
    logger.info(f"best: {best_runs_docs}")
    wandb.log({"best_val_acc": best_val_acc})
    wandb.finish()
    logger.info(f"best accuracy obtained was {best_val_acc} with a loss of {best_val_loss}")


if __name__ == "__main__":
    sweep_config = {
        "method": "bayes",  # grid, random, bayes
        "metric": {
            "name": "val_acc",
            "goal": "maximize",  # We want to maximize validation accuracy
        },
        "parameters": {
            "lr": {"values": [0.001, 0.0001]},
            "batch_size": {"values": [16, 32, 64]},
            "num_epochs": {"min": 5, "max": 15},
            "model": {"values": ["MegaFormer", "ViT"]},
            "img_size": {"values": [224]},
            "transform_p": {"values": [0.0, 0.1, 0.3]},
            "scheduler": {"values": [0, 1]},
            "scheduler_intensity": {"values": [0.1, 0.2, 0.3]},
            "warmup_epochs": {"values": [1, 2, 3]},
            "warmup_lr": {"values": [0.00001, 0.00005, 0.000001]},
        },
    }

    # sweep_id = wandb.sweep(sweep_config, project="my-cancer-prediction-project")
    # wandb.agent(sweep_id, function=train)
    train()
