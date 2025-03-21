import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
import os

from AttenUNet import ResU_VAE
import random


def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, train_loader, optimizer, epoch, scheduler):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    scheduler.step()
    print("Epoch: {} Average loss: {:.4f}".format(epoch, train_loss / len(train_loader.dataset)))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device, non_blocking=True)
            recon, mu, logvar = model(data)
            loss = loss_function(recon, data, mu, logvar)
            test_loss += loss.item()
    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))

def plot_sampled_digits(model):
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, model.channels).to(device)
        samples = model.decode(z).cpu()
        vutils.save_image(samples, "MNIST_Experiments/Output/VAE/vae_sampled_digits.png", nrow=8)


def plot_digit_interpolation(model, digits, steps=10):
    # get digits are image samples
    model.eval()
    with torch.no_grad():
        # Extract latent representation and remove extra dimensions if needed
        z = model.encode(digits.to(device))[0].view(digits.size(0), -1)
        z_interpolated = torch.zeros(steps, z.size(1)).to(device)
        for i in range(steps):
            z_interpolated[i] = (1 - i/(steps-1)) * z[0] + (i/(steps-1)) * z[1]
        samples = model.decode(z_interpolated).cpu()
        vutils.save_image(samples, "MNIST_Experiments/Output/VAE/vae_digit_interpolation.png", nrow=steps)

        
def main():
    # Data loading and transformation for MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer and scheduler
    model = ResU_VAE(channels=channels).to(device)
    print(f"Initialized Convolutional VAE with parameters: {sum(p.numel() for p in model.parameters())}")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch, scheduler)
        test(model, test_loader)
        torch.save(model.state_dict(), "MNIST_Experiments/Output/VAE/vae.pth")

        # Generate samples from the learned distribution
        plot_sampled_digits(model)

        # sample two images from different classes
        idx1 = random.randrange(len(train_dataset))
        idx2 = random.randrange(len(train_dataset))
        while train_dataset[idx1][1] == train_dataset[idx2][1]:
            idx2 = random.randrange(len(train_dataset))
        digits = torch.stack([train_dataset[idx1][0], train_dataset[idx2][0]])
        plot_digit_interpolation(model, digits, steps=20)


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    epochs = 100
    channels = 16
    learning_rate = 1e-3
    weight_decay = 1e-2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("MNIST_Experiments/Output/VAE", exist_ok=True)
    main()