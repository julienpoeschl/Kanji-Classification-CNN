import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from cnn_model import create_model, get_optimizer, get_scheduler, get_criterion

from dataset.src.data_augmentation import DataAugmenter, AugmentationSettings
from utils.src.cuda import cuda_device
from model.src.paths import BEST_MODEL_PATH, FINAL_MODEL_PATH, CHECKPOINT_DIR
from dataset.src.paths import PROCESSED_INPUTS_PATH, PROCESSED_OUTPUTS_PATH, LABEL_MAP_PATH


class AugmentedKanjiDataset(Dataset):
    """
    Custom Dataset for Kanji images with on-the-fly augmentation and virtual expansion.
    Each original image appears 'virtual_factor' times per epoch, each time with a new augmentation.
    """
    def __init__(
        self, 
        images: np.ndarray, 
        labels: np.ndarray, 
        augmenter: Optional[DataAugmenter] = None, 
        virtual_factor: int = 1
    ) -> None:
        self.images = images
        self.labels = labels
        self.augmenter = augmenter
        self.virtual_factor = virtual_factor
        # Handle channel dimension
        if self.images.ndim == 4 and self.images.shape[-1] == 1:
            self.images = self.images.squeeze(axis=-1)
        elif self.images.ndim == 3:
            pass
        else:
            raise ValueError("Images must have shape (N, H, W) or (N, H, W, 1)")

    def __len__(self) -> int:
        """Returns virtual length."""
        return len(self.images) * self.virtual_factor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets (augmented) item at given index."""
        real_idx = idx % len(self.images)
        img = self.images[real_idx]
        label = self.labels[real_idx]
        if self.augmenter:
            img = self.augmenter.augment_single(img)
        # Add channel dimension for PyTorch
        img = img[np.newaxis, :, :]
        img = torch.from_numpy(img).float()
        label = torch.tensor(label).long()
        return img, label


class KanjiTrainer:
    """Trainer class for Kanji CNN model."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        verbose: bool = True
    ) -> None:
        self.model = model
        self.device = device
        self.criterion = get_criterion()
        self.optimizer = get_optimizer(model, learning_rate, weight_decay)
        self.verbose = verbose
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_accuracies: list[float] = []
        self.val_accuracies: list[float] = []
    
    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        previous_percent = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if self.verbose:
                percent = 100 * (batch_idx + 1) / len(train_loader)
                if percent != previous_percent:
                    print(f"Epoch {self.current_epoch+1}, Progress: {percent:.1f}%", end="\r")
                previous_percent = percent
        
        if self.verbose:
            print("")
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = 0
        accuracy = 0
        if total != 0:
            avg_loss = total_loss / total
            accuracy = correct / total
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 10
    ) -> None:
        """
        Full training loop with early stopping.
        
        :param train_loader: Training data loader
        :param val_loader: Validation data loader
        :param epochs: Number of epochs to train
        :param save_dir: Directory to save checkpoints
        :param patience: Early stopping patience
        """
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0
        self.current_epoch = 0
        
        self.scheduler = get_scheduler(self.optimizer, epochs)
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, BEST_MODEL_PATH)
                print(f"  -> Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            self.current_epoch += 1
        
        # Save final model
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }, FINAL_MODEL_PATH)
        print(f"Saved final model to {CHECKPOINT_DIR}")
        print(f"Train losses per epoch: {self.train_losses}.")
        print(f"Train accuracy per epoch: {self.train_accuracies}.")
        print(f"Validation losses per epoch: {self.val_losses}.")
        print(f"Validation accuracy per epoch: {self.val_accuracies}.")


def prepare_data_loaders(
    inputs: np.ndarray,
    outputs: np.ndarray,
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 0,
    augmentation_settings: Optional[AugmentationSettings] = None,
    virtual_factor: int = 1
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders with on-the-fly augmentation.
    """

    if augmentation_settings is None:
        augmentation_settings = AugmentationSettings(
            rotation_enabled=True,
            rotation_range=5.0,
            scale_enabled=True,
            scale_range=(0.85, 1.15),
            noise_enabled=True,
            noise_stddev=0.05,
            brightness_enabled=True,
            brightness_range=(-0.2, 0.2),
            contrast_enabled=True,
            contrast_range=(0.8, 1.2),
            shift_enabled=True,
            shift_range=0.1,
            elastic_enabled=False,
            augmentation_factor=1,
            keep_originals=True
        )
    augmenter = DataAugmenter(augmentation_settings)
    augmenter.set_seed(42) # reproducability

    # Split into train and validation
    n_samples = len(inputs)
    val_size = int(n_samples * val_split)
    train_size = n_samples - val_size
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_images = inputs[train_indices]
    train_labels = outputs[train_indices]
    val_images = inputs[val_indices]
    val_labels = outputs[val_indices]

    train_dataset = AugmentedKanjiDataset(train_images, train_labels, augmenter, virtual_factor=virtual_factor)
    val_dataset = AugmentedKanjiDataset(val_images, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader


if __name__ == "__main__":

    # --- Settings ---
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    VAL_SPLIT = 0.1
    PATIENCE = 10
    VIRTUAL_FACTOR = 100

    # --- Set device ---
    device = cuda_device.get_device()
    print(f"Using device: {device}")

    # --- Building training/validation set ---
    print("Loading processed data...")
    inputs = np.load(PROCESSED_INPUTS_PATH)["arr"]
    outputs = np.load(PROCESSED_OUTPUTS_PATH)["arr"]
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        output_map = json.load(f)
    num_classes = len(output_map)
    print(f"Loaded {len(inputs)} samples with {num_classes} classes")
    print(f"Input shape: {inputs.shape}")

    print("Preparing data loaders...")
    train_loader, val_loader = prepare_data_loaders(
        inputs, outputs,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        virtual_factor=VIRTUAL_FACTOR
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- Model creation ---
    print("Creating model...")
    model = create_model(num_classes=num_classes, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --- Trainer creation ---
    trainer = KanjiTrainer(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --- Training epochs ---
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("-" * 80)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        patience=PATIENCE
    )

    print("-" * 80)
    print("Training complete!")
