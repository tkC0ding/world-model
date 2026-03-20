from torch.utils.data import Dataset
import os
import json
import cv2
from torchvision import transforms
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="CarRacing Dataset")
parser.add_argument("--data_dir", type=str, default="data", help="Directory where the generated data is stored")

args = parser.parse_args()

DATA_DIR = args.data_dir

class CarRacingDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.json_data_files = [os.path.join(data_dir, i, j) for i in os.listdir(data_dir) for j in os.listdir(os.path.join(data_dir, i)) if j.endswith(".json")]

    def __len__(self):
        a = 0
        for i in self.json_data_files:
            with open(i, "r") as f:
                data = json.load(f)
                a += len(data)
        return a
    
    def __getitem__(self, idx):
        goal = idx // 1000
        for j,i in enumerate(self.json_data_files):
            with open(i, "r") as f:
                data = json.load(f)
                if j == goal:
                    img = cv2.imread(data[idx % 1000]["image"])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                    img = np.transpose(img, (2, 0, 1))
                    action = data[idx % 1000]["action"]
                    if self.transforms:
                        img = self.transforms(img)
                    return img, action