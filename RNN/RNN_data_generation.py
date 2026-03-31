import torch
import os
from VariationalAutoencoder.VAE import VAE
import argparse
import json
import cv2
import pickle
from utils import progress_bar

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

VAE_model.eval()
for param in VAE_model.parameters():
    param.requires_grad = False

episodes = os.listdir(args.data_dir)
training_data = []

for episode in episodes:
    episode_path = os.path.join(args.data_dir, episode)
    temp = []
    with open(os.path.join(episode_path, 'data.json'), 'r') as f:
        data = json.load(f)
        for i,item in enumerate(data):
            s = "progress: " + progress_bar(i, len(data))
            print(f"\r{s}", end='')
            img = cv2.imread(item['image'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                mu, logvar = VAE_model.encode(img)
                z = VAE_model.reparameterize(mu, logvar)
                if i >= 993:
                    continue
                temp.append((z.squeeze(0), torch.tensor(item['action'], dtype=torch.float32)))
    training_data.append(temp)

with open(f'{args.data_dir}/RNN_training_data.pkl', 'wb') as f:
    pickle.dump(training_data, f)