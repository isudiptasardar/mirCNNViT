import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Literal
from utils.early_stop import EarlyStop
from tqdm import tqdm
from sklearn.metrics import accuracy_score
class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, device: torch.device, epochs: int, early_stopping_patience: int, early_stopping_delta: float, early_stopping_mode: Literal["min", "mix"], save_path: str) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_mode = early_stopping_mode
        self.save_path = save_path

        self.early_stop = EarlyStop(
            patience=self.early_stopping_patience,
            delta=self.early_stopping_delta,
            mode=self.early_stopping_mode
        )

        self.model.to(self.device)
    
    def train_epoch(self, epoch: int):
        self.model.train()

        total_loss: float = 0.0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(self.train_dataloader, desc=f"Training - Epoch {epoch + 1}", leave=False)
        
        for idx, (x_mrna, x_mirna, label) in enumerate(progress_bar):
            x_mrna = x_mrna.to(self.device)
            x_mirna = x_mirna.to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(x_mrna, x_mirna)
            loss = self.criterion(outputs, label)

            loss.backward()

            # Prevent Gradient Exploding
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss = total_loss + loss.item()

            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

            # Calculate running accuracy
            running_accuracy = accuracy_score(all_labels, all_predictions)

            # Update progress bar
            progress_bar.set_postfix({"Loss": f'{loss.item():.4f}', "Accuracy": f'{running_accuracy:.4f}'})
        
        # Calculate Average Loss and Total Accuracy for the epoch
        avg_loss = total_loss / len(self.train_dataloader)
        total_accuracy = accuracy_score(all_labels, all_predictions)

        logging.info(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {total_accuracy:.4f}")

        return avg_loss, total_accuracy
    
    def validate_epoch(self, epoch: int):
        self.model.eval()

        total_loss: float = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []

        progress_bar = tqdm(self.val_dataloader, desc=f"Validating - Epoch {epoch + 1}", leave=False)

        with torch.no_grad():
            for idx, (x_mrna, x_mirna, label) in enumerate(progress_bar):
                x_mrna = x_mrna.to(self.device)
                x_mirna = x_mirna.to(self.device)
                label = label.to(self.device)

                outputs = self.model(x_mrna, x_mirna)
                loss = self.criterion(outputs, label)

                total_loss = total_loss + loss.item()

                probabilities = torch.softmax(outputs, dim = 1)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                # Calculate running accuracy
                running_accuracy = accuracy_score(all_labels, all_predictions)

                # Update progress bar
                progress_bar.set_postfix({"Loss": f'{loss.item():.4f}', "Accuracy": f'{running_accuracy:.4f}'})

        # Calculate Average Loss and Total Accuracy for the epoch
        avg_loss = total_loss / len(self.val_dataloader)
        total_accuracy = accuracy_score(all_labels, all_predictions)

        logging.info(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {total_accuracy:.4f}")
        return avg_loss, total_accuracy
    
    def train(self):
        train_losses = []
        val_losses = []

        train_accuracies = []
        val_accuracies = []

        # Best Metrics -> Store
        best_val_accuracy: float = -float('inf')
        best_val_loss: float = float('inf')

        for epoch in range(self.epochs):
            train_loss, train_accuracy = self.train_epoch(epoch)
            val_loss, val_accuracy = self.validate_epoch(epoch)

            # Append to the history
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            # Early Stopping
            if self.early_stopping_mode == "min":
                stop_motinor_val = val_loss
            elif self.early_stopping_mode == "max":
                stop_motinor_val = val_accuracy
            else:
                raise ValueError("Invalid early stopping mode... Ref. utils/early_stop.py")
            
            should_stop: bool = self.early_stop(stop_motinor_val)

            # Save best model if improved
            if self.early_stop.is_improved:
                logging.info(f"Model improved on epoch {epoch + 1}...")
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
            else:
                logging.info(f"Model did not improve on epoch {epoch + 1}...")

            if should_stop:
                logging.info(f"Early stopping triggered on epoch {epoch + 1}...")
                break
        return best_val_accuracy, best_val_loss, train_accuracies, train_losses, val_accuracies, val_losses