import torch
from VAE import VAE, VAELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tensor_dataset import CarRacingDataset
import argparse
import os

parser = argparse.ArgumentParser(description="VAE Trainer for CarRacing-v3")
parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the training data")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
parser.add_argument("--model_save_path", type=str, default="vae_model.pth", help="Path to save the trained model")
parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of data to use for training")
args = parser.parse_args()

dataset = CarRacingDataset(data_dir=args.data_dir)

indices = torch.randperm(len(dataset))

train_size = int((args.train_split-0.1) * len(dataset))
validation_size = int(0.1 * len(dataset))

train_indices = indices[:train_size]
validation_indices = indices[train_size:train_size+validation_size]
test_indices = indices[train_size+validation_size:]

train_dataset = Subset(dataset, train_indices)
validation_dataset = Subset(dataset, validation_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

