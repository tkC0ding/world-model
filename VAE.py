from torch.nn import Module
import torch

class VAE(Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU()
        )

        self.flatten = torch.nn.Flatten()
        self.fc_mu = torch.nn.Linear(256*4*4, 32)
        self.fc_logvar = torch.nn.Linear(256*4*4, 32)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 1024),
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (256, 4, 4)),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid()
        )