"""
Trainer module for cat/dog image classification.
Handles training, validation, and model saving/loading.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import time
from pathlib import Path
import json
from tqdm import tqdm
from utils import auto_select_device


class Trainer:
    """Trainer class for model training and validation."""

    def __init__(self,
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 checkpoint_dir: str = "checkpoints"):
        """
        Args:
            model: PyTorch model to train
            device: Device to use for training (cuda/cpu)
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.device = device or auto_select_device()
        self.model.to(self.device)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = None

        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def train_epoch(self,
                    train_loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    scheduler: Optional[Any] = None) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        # Update learning rate if scheduler is provided
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # This scheduler needs validation metrics
                pass
            else:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self,
                 val_loader: DataLoader,
                 criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation", leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # Update statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 10,
              learning_rate: float = 0.001,
              optimizer_name: str = 'adam',
              criterion_name: str = 'cross_entropy',
              scheduler_name: Optional[str] = None,
              patience: int = 5,
              save_best: bool = True) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            optimizer_name: Name of optimizer ('adam', 'sgd', 'adamw')
            criterion_name: Name of loss function ('cross_entropy', 'bce')
            scheduler_name: Name of scheduler ('step', 'plateau', 'cosine')
            patience: Early stopping patience
            save_best: Whether to save best model

        Returns:
            Training history
        """
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Optimizer: {optimizer_name}, LR: {learning_rate}")
        print(f"Criterion: {criterion_name}")
        if scheduler_name:
            print(f"Scheduler: {scheduler_name}")

        # Setup criterion
        if criterion_name == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        elif criterion_name == 'bce':
            criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown criterion: {criterion_name}")

        # Setup optimizer
        if optimizer_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Setup scheduler
        scheduler = None
        if scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif scheduler_name == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3
            )
        elif scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Early stopping
        best_val_acc = 0.0
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, criterion, optimizer, scheduler
            )

            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)

            # Update scheduler if ReduceLROnPlateau
            if scheduler_name == 'plateau' and scheduler is not None:
                scheduler.step(val_acc)
                current_lr = optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"Time: {epoch_time:.2f}s | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_val_acc = best_val_acc
                self.best_model_path = self.checkpoint_dir / f"best_model_epoch{epoch+1}_acc{val_acc:.2f}.pth"
                self.save_checkpoint(self.best_model_path, epoch, val_acc)
                print(f"Saved best model: {self.best_model_path}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Save final model
        final_path = self.checkpoint_dir / "final_model.pth"
        self.save_checkpoint(final_path, epochs, val_acc)
        print(f"\nSaved final model: {final_path}")

        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history: {history_path}")

        return self.history

    def save_checkpoint(self,
                       path: Path,
                       epoch: int,
                       val_acc: float) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            val_acc: Validation accuracy
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        if 'val_acc' in checkpoint:
            self.best_val_acc = checkpoint['val_acc']

        print(f"Loaded checkpoint from {path}")
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")

        return checkpoint

    def get_model_summary(self) -> str:
        """Get model summary as string."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        summary = "Model Summary:\n"
        summary += f"  Total parameters: {total_params:,}\n"
        summary += f"  Trainable parameters: {trainable_params:,}\n"
        summary += f"  Device: {self.device}\n"

        return summary


class HyperparameterConfig:
    """Configuration class for hyperparameters."""

    def __init__(self,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 10,
                 optimizer: str = 'adam',
                 criterion: str = 'cross_entropy',
                 scheduler: Optional[str] = None,
                 patience: int = 5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.patience = patience

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'optimizer': self.optimizer,
            'criterion': self.criterion,
            'scheduler': self.scheduler,
            'patience': self.patience
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HyperparameterConfig':
        """Create from dictionary."""
        return cls(**config_dict)

    def __str__(self) -> str:
        config_str = "Hyperparameter Configuration:\n"
        for key, value in self.to_dict().items():
            config_str += f"  {key}: {value}\n"
        return config_str
