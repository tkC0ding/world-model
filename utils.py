from torch.utils.data import Subset, DataLoader
import torch

def train_test_split(dataset, train_split=0.8, validation_split=0.1, batch_size=64):
    indices = torch.randperm(len(dataset))

    train_size = int((train_split-0.1) * len(dataset))
    validation_size = int(validation_split * len(dataset))

    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size+validation_size]
    test_indices = indices[train_size+validation_size:]

    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader