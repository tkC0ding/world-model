import torch
from VAE import VAE, VAELoss
from utils import train_test_split
from torch.optim import Adam
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

train_loader, validation_loader, test_loader = train_test_split(dataset, args.train_split, 0.1, args.batch_size)