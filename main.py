import logging
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from utils.dataset import FCGRDataset
from core.model import HybridCNNViT
from config import CONFIG
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
from core.train import Trainer
import optuna

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('log.log', mode='w', encoding='utf-8')
        ]
    )

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def seed_all(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def objective(trial: optuna.Trial):
    
    seed_all(seed=CONFIG['seed'])

    # Hyperparameter Tuning
    batch_size = trial.suggest_categorical('batch_size', [32, 64])

    df = pd.read_csv(CONFIG['raw_data_path'])
    required_cols = [CONFIG['m_rna_col'], CONFIG['mi_rna_col'], CONFIG['label_col']]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    df = df[required_cols]
    logging.info(f"Dataset Shape: {df.shape},\nDistribution: {df[CONFIG['label_col']].value_counts()}")

    # Remove duplicated
    df = df.drop_duplicates(subset=[CONFIG['m_rna_col'], CONFIG['mi_rna_col']], keep='first').reset_index(drop=True)
    logging.info(f"Distribution after removing duplicates: {df[CONFIG['label_col']].value_counts()}")

    if CONFIG['use_test'] == False:
        # Split
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=CONFIG['seed'], shuffle=True, stratify=df[CONFIG['label_col']])

        # Load Dataset
        train_ds = FCGRDataset(data=train_df, k = 6, m_rna_col=CONFIG['m_rna_col'], mi_rna_col=CONFIG['mi_rna_col'], label_col=CONFIG['label_col'], dataset_type="Training")
        val_ds = FCGRDataset(data=val_df, k = 6, m_rna_col=CONFIG['m_rna_col'], mi_rna_col=CONFIG['mi_rna_col'], label_col=CONFIG['label_col'], dataset_type="Validation")

        # Dataloader
        train_dataloader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=CONFIG['num_workers'], worker_init_fn=seed_worker)
        val_dataloader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, num_workers=CONFIG['num_workers'], worker_init_fn=seed_worker)

        logging.info(f"Train Dataloader Size: {len(train_dataloader)}")
        logging.info(f"Validation Dataloader Size: {len(val_dataloader)}")

        model = HybridCNNViT(embed_dim=256, num_heads=8, num_layers=4, mlp_dim=512, num_classes=2, dropout_rate=0.3).to(CONFIG['device'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

        best_val_accuracy, best_val_loss, train_accuracies, train_losses, val_accuracies, val_losses = Trainer(model=model, optimizer=optimizer, criterion=criterion, train_dataloader=train_dataloader, val_dataloader=val_dataloader, device=CONFIG['device'], epochs=100, early_stopping_patience=10, early_stopping_delta=0.0001, early_stopping_mode="max", save_path="out").train()

        logging.info(f"Best Validation Accuracy: {best_val_accuracy}")
        logging.info(f"Best Validation Loss: {best_val_loss}")
        return best_val_accuracy
        

    else:
        logging.info("Using Train, Validation and Test datasets...")

        # Split
        train_df, val_test_df = train_test_split(df, test_size=0.3, random_state=CONFIG['seed'], shuffle=True, stratify=df[CONFIG['label_col']])
        val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=CONFIG['seed'], shuffle=True, stratify=val_test_df[CONFIG['label_col']])

        # Load Dataset
        train_ds = FCGRDataset(data=train_df, k = 6, m_rna_col=CONFIG['m_rna_col'], mi_rna_col=CONFIG['mi_rna_col'], label_col=CONFIG['label_col'], dataset_type="Training")
        val_ds = FCGRDataset(data=val_df, k = 6, m_rna_col=CONFIG['m_rna_col'], mi_rna_col=CONFIG['mi_rna_col'], label_col=CONFIG['label_col'], dataset_type="Validation")
        test_ds = FCGRDataset(data=test_df, k = 6, m_rna_col=CONFIG['m_rna_col'], mi_rna_col=CONFIG['mi_rna_col'], label_col=CONFIG['label_col'], dataset_type="Testing")

        # Dataloader
        train_dataloader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True, num_workers=CONFIG['num_workers'], worker_init_fn=seed_worker)
        val_dataloader = DataLoader(dataset=val_ds, batch_size=32, shuffle=False, num_workers=CONFIG['num_workers'], worker_init_fn=seed_worker)
        test_dataloader = DataLoader(dataset=test_ds, batch_size=32, shuffle=False, num_workers=CONFIG['num_workers'], worker_init_fn=seed_worker)

        logging.info(f"Train Dataloader Size: {len(train_dataloader)}")
        logging.info(f"Validation Dataloader Size: {len(val_dataloader)}")
        logging.info(f"Test Dataloader Size: {len(test_dataloader)}")

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
if __name__ == "__main__":
    setup_logger()
    main()