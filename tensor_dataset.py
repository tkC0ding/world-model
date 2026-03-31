from torch.utils.data import Dataset
import os
import json
import cv2
import numpy as np
import pickle

class CarRacingDataset(Dataset):
    def __init__(self, data_dir="", transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.json_data_files = [os.path.join(data_dir, i, j) for i in os.listdir(data_dir) for j in os.listdir(os.path.join(data_dir, i)) if j.endswith(".json")]
        self.total_samples = 0
        for i in self.json_data_files:
            with open(i, "r") as f:
                data = json.load(f)
                self.total_samples += len(data)

    def __len__(self):
        return self.total_samples

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

class RNN_Dataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.x = []
        self.y = []
        for episode in self.data:
            for i in range(0, len(episode)-16, 16):
                self.x.append(episode[i:i+32])
                self.y.append([i[0] for i in episode[i+1:i+33]])
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]