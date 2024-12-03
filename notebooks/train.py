import os
import random
import argparse
from functools import partial

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm

from healnet.models import HealNet  # Adjust import path as necessary

# ------------------ Set Seed for Reproducibility ------------------

def set_seed(seed: int = 42):
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For CUDA
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------ Custom Dataset Class ------------------

class CustomDataset(Dataset):
    def __init__(self, df, img_features_path, scaler=None, limit=None):
        self.df = df.reset_index(drop=True)
        if limit is not None:
            self.df = self.df.iloc[:limit]
        self.img_features_path = img_features_path
        self.tabular_columns = [col for col in df.columns if col.startswith('cnv')]

        self.scaler = scaler

        if self.scaler is not None:
            # Ensure the scaler has been fitted
            if not hasattr(self.scaler, 'mean_') or not hasattr(self.scaler, 'scale_'):
                raise ValueError("Scaler must be fitted before being passed to the dataset.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Extract row data
        row = self.df.iloc[idx]
        wsi_file_name = row["wsi_file_name"][:-5]  # Adjust as needed
        target_label = row["progressor_status"]

        # Convert target label from 'NP'/'P' to 0/1
        target = 1 if target_label == 'P' else 0

        # Load tabular features
        tabular_features = row[self.tabular_columns].values.astype('float32')
        if self.scaler:
            # Apply scaling
            tabular_features = self.scaler.transform(tabular_features.reshape(1, -1)).flatten()

        tabular_tensor = torch.tensor(tabular_features, dtype=torch.float32)

        # Load image features from .h5 file
        img_features_file = os.path.join(self.img_features_path, wsi_file_name + ".h5")
        with h5py.File(img_features_file, 'r') as h5_file:
            # Extract 'cluster_features' dataset
            if "cluster_features" in h5_file:
                cluster_features = h5_file["cluster_features"][:]
            else:
                raise KeyError(f"'cluster_features' not found in {img_features_file}")

        # Convert cluster features to a tensor
        img_features_tensor = torch.tensor(cluster_features, dtype=torch.float32)

        # Prepare the target tensor
        target_tensor = torch.tensor(target, dtype=torch.long)  # Changed to long for CrossEntropyLoss

        return {
            'tabular': tabular_tensor,
            'image': img_features_tensor,
            'target': target_tensor
        }

# ------------------ Training Function ------------------

def train_and_evaluate(config, args, no_wandb=False):
    # Initialize wandb if not disabled
    if not no_wandb:
        wandb.init(project=args.project_name, config=config)
        config = wandb.config
    else:
        config = argparse.Namespace(**config)

    # Set seed
    set_seed(config.seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    path_to_img_features = args.img_features_path
    path_to_csv = args.csv_path

    # Load the CSV file
    df = pd.read_csv(path_to_csv)

    # Filter for the current split
    split = config.split
    split_name = f"healnet_clustfeats_{split}"
    split_column = f"split_{split}"
    if split_column not in df.columns:
        raise ValueError(f"Column {split_column} not found in the dataframe.")

    train_df = df[df[split_column] == "train"]
    val_df = df[df[split_column] == "val"]
    test_df = df[df[split_column] == "test"]

    print(f"Split {split} - Train samples: {len(train_df)}")
    print(f"Split {split} - Validation samples: {len(val_df)}")
    print(f"Split {split} - Test samples: {len(test_df)}")

    # Identify tabular columns
    tabular_columns = [col for col in df.columns if col.startswith('cnv')]

    # Initialize the scaler
    scaler = StandardScaler()
    scaler.fit(train_df[tabular_columns].values)

    # Determine if we need to limit dataset sizes
    #limit = 100 if no_wandb else None
    limit = None

    # Create datasets for each split
    train_dataset = CustomDataset(train_df, path_to_img_features, scaler=scaler, limit=limit)
    val_dataset = CustomDataset(val_df, path_to_img_features, scaler=scaler, limit=limit)
    test_dataset = CustomDataset(test_df, path_to_img_features, scaler=scaler, limit=limit)

    # Define loader arguments
    loader_args = {
        "batch_size": config.batch_size,  # Adjustable via wandb
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": True,
    }

    # Create DataLoaders for each dataset
    train_loader = DataLoader(train_dataset, **loader_args)
    # For validation and test, shuffle=False
    val_loader = DataLoader(val_dataset, **loader_args)
    test_loader = DataLoader(test_dataset, **loader_args)

    # Inspect data shapes
    if len(train_dataset) > 0:
        sample_image = train_dataset[0]['image']
        print(f"Sample image shape: {sample_image.shape}")  # Example: (channels, features)

        tabular_features = train_dataset[0]['tabular'].shape[0]
        image_features = train_dataset[0]['image'].shape[0]

        print('Tabular features:', tabular_features)
        print('Image features:', image_features)
    else:
        print("No data available in the dataset.")
        return

    # Instantiate the HealNet model with dynamic input_channels
    model = HealNet(
        modalities=2,
        input_channels=[tabular_features, image_features],  # Dynamically set based on data
        input_axes=[1, 2],
        num_classes=2,  # Adjust based on your dataset
        # Add other parameters as required by your model configuration
    ).to(device)

    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['progressor_status']),
        y=train_df['progressor_status']
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

    # Define the scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=config.num_epochs,
        anneal_strategy='linear',  # or 'cos', depending on preference
        pct_start=config.pct_start,             # percentage of cycle spent increasing LR
        div_factor=config.div_factor,           # initial lr = max_lr / div_factor
        final_div_factor=config.final_div_factor,  # minimum lr = max_lr / (div_factor * final_div_factor)
        verbose=False  # Set to True if you want to see LR updates
    )

    # Training parameters
    num_epochs = config.num_epochs
    best_val_loss = float('inf')
    patience = config.patience
    counter = 0

    # Lists to store metrics for visualization
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training and Validation Loop
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        # ------------------ Training Phase ------------------
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            tabular_data = batch['tabular'].to(device)
            image_data = batch['image'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model([tabular_data, image_data])

            # Compute loss
            loss = criterion(outputs, targets)
            train_loss += loss.item() * tabular_data.size(0)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            # Scheduler step
            scheduler.step()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # ------------------ Validation Phase ------------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                tabular_data = batch['tabular'].to(device)
                image_data = batch['image'].to(device)
                targets = batch['target'].to(device)

                outputs = model([tabular_data, image_data])
                loss = criterion(outputs, targets)
                val_loss += loss.item() * tabular_data.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        # Log metrics to wandb if enabled
        if not no_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": scheduler.get_last_lr()[0],
                "split": split_name
            })

        # ------------------ Early Stopping ------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f'healnet_clustfeats_{split}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Model saved at {checkpoint_path}!")
            counter = 0
        else:
            counter += 1
            print(f"No improvement. Counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # ------------------ Testing Phase ------------------

    # Load the best model
    best_model_path = os.path.join(args.checkpoint_dir, f'healnet_clustfeats_{split}.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("✅ Best model loaded for testing.")
    else:
        print("❌ Best model not found. Skipping testing.")
        return

    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            tabular_data = batch['tabular'].to(device)
            image_data = batch['image'].to(device)
            targets = batch['target'].to(device)

            outputs = model([tabular_data, image_data])
            loss = criterion(outputs, targets)
            test_loss += loss.item() * tabular_data.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = 100 * correct / total

    print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")

    # Log test metrics to wandb if enabled
    if not no_wandb:
        wandb.log({
            "test_loss": avg_test_loss,
            "test_accuracy": test_accuracy,
            "split": split_name
        })

    # Finish wandb run if enabled
    if not no_wandb:
        wandb.finish()

# ------------------ Argument Parser ------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train HealNet with wandb integration.")

    parser.add_argument(
        "--project_name",
        type=str,
        default="HealNet_Project",
        help="Name of the wandb project."
    )
    parser.add_argument(
        "--img_features_path",
        type=str,
        required=True,
        help="Path to the directory containing image feature .h5 files."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV file containing dataset information."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker threads for DataLoader."
    )
    parser.add_argument(
        "--no_wandb",
        action='store_true',
        help="Flag to disable wandb logging for testing purposes."
    )

    return parser.parse_args()

# ------------------ Main Function ------------------

def main():
    args = parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Define hyperparameter search space for wandb sweeps
    sweep_configuration = {
        'method': 'grid',  # grid, random, bayes
        'name': 'HealNet_Sweep',
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'   
        },
        'parameters': {
            'split': {
                'values': list(range(10))  # Splits 0 to 9
            },
            'learning_rate': {
                'values': [0.001, 0.005, 0.01]
            },
            'batch_size': {
                'values': [4, 8, 16]
            },
            'momentum': {
                'values': [0.9, 0.95]
            },
            'max_lr': {
                'values': [0.05, 0.1]
            },
            'pct_start': {
                'values': [0.3, 0.4]
            },
            'div_factor': {
                'values': [10.0, 25.0]
            },
            'final_div_factor': {
                'values': [1e4, 1e3]
            },
            'num_epochs': {
                'values': [20]
            },
            'patience': {
                'values': [10]
            },
            'seed': {
                'value': 42
            }
        }
    }

    # Initialize the sweep if wandb is not disabled
    if not args.no_wandb:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project_name)
        print(f"Create sweep with ID: {sweep_id}")
        print(f"Sweep URL: https://wandb.ai/{wandb.api.api_key}/healnet/sweeps/{sweep_id}")

        # Launch the sweep agent
        wandb.agent(sweep_id, function=partial(train_and_evaluate, config=sweep_configuration['parameters'], args=args, no_wandb=args.no_wandb), count=10)  # count=10 to cover splits 0-9
    else:
        # If wandb is disabled, run a single training run on split 0 with limited data
        print("Running in test mode without wandb logging.")

        # Define a default configuration
        test_config = {
            'split': 0,
            'learning_rate': 0.001,
            'batch_size': 4,
            'momentum': 0.9,
            'max_lr': 0.05,
            'pct_start': 0.3,
            'div_factor': 10.0,
            'final_div_factor': 10000.0,
            'num_epochs': 20,  # Reduced epochs for testing
            'patience': 2,     # Reduced patience for testing
            'seed': 42
        }

        train_and_evaluate(test_config, args, no_wandb=True)

if __name__ == "__main__":
    main()
