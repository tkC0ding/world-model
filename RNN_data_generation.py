import torch
import os
from VAE import VAE
import argparse

VAE_model = VAE()

parser = argparse.ArgumentParser(description="Data Generation for RNN Training")
parser.add_argument("--checkpoint_path", type=str, default="VAE_model/checkpoint.pth", help="Path to the VAE checkpoint")
parser.add_argument("--data_dir", type=str, default="data", help="Directory to images generated data")
parser.add_argument("--seq_length", type=int, default=10, help="Sequence length for RNN training")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for RNN training")
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint_path)
VAE_model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']