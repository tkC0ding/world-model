from torch.utils.data import Subset, DataLoader
import torch


def progress_bar(current, total, bar_length=30):
    fraction = current / total
    filled = int(bar_length * fraction)
    
    bar = '=' * filled + '-' * (bar_length - filled)
    percent = fraction * 100

    return f'[{bar}] {percent:.2f}%'

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

def train_VAE(model, train_loader, val_loader, optimizer, device, loss_fn, epochs, save_dir):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()

            recon_x, mu, logvar = model(x)

            loss = loss_fn(recon_x, x, mu, logvar, x.size(0))
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
            s = f"Epoch {epoch+1}/{epochs}" + progress_bar(i, len(train_loader))
            print(f"\r{s}", end='')
        print(f"\nEpoch {epoch+1} Train_Loss: {train_loss/len(train_loader)}")

        val_loss = 0
        model.eval()
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)

            recon_x, mu, logvar = model(x)

            loss = loss_fn(recon_x, x, mu, logvar, x.size(0))

            val_loss += loss.item()
            s = f"Epoch {epoch+1}/{epochs}" + progress_bar(i, len(val_loader))
            print(f"\r{s}", end='')
        print(f"\nEpoch {epoch+1} Val_Loss: {val_loss/len(val_loader)}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, save_dir)
        print(f"Model saved to {save_dir}\n\n")