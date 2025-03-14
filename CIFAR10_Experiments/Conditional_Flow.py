import torch
from torchvision import transforms, datasets
from tqdm import tqdm
import os

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from AttenUNet import AttenUNet

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load CIFAR10 dataset
def get_dataloader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                             num_workers=2, pin_memory=True)
    return dataloader

# Training procedure using flow matching.
def train_flow_matching(model, dataloader, num_epochs=5, lr=1e-3, wd=1e-2):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-8)
    mse_loss = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, classes in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            images = images.to(device, non_blocking=True)  # CIFAR10 images: [batch, 1, 28,28]
            classes = classes.to(device, non_blocking=True)

            batch = images.shape[0]

            # Sample a random scalar time t uniformly in [0, 1] for each sample.
            t = torch.rand(batch, device=device)
            # t = torch.zeros(batch, device=device).float()

            # Sample noise images from Normal distribution.
            noise = torch.randn_like(images, device=device)

            # Create intermediate samples by linear interpolation:
            # x(t) = (1-t)*noise + t*data. We need to reshape t.
            t_reshaped = t.view(batch, 1, 1, 1)
            x_t = (1 - t_reshaped) * noise + t_reshaped * images

            # The ideal velocity field is simply the difference (data - noise)
            # since d/dt x(t) = data - noise.
            target = images - noise

            pred = model(x_t, t, classes)
            loss = mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "CIFAR10_Experiments/Output/CIFAR10_flow_model.pth")

# Generate new samples via Euler integration using the learned flow field.
def generate_samples(model, classes, num_steps=100, fixed_sample=None):
    model.eval()
    B = classes.size(0)

    # Start from pure noise (i.e. at time t=0).
    if fixed_sample is None:
        x = torch.randn(B, 1, 28, 28, device=device)
    else:
        x = fixed_sample.to(device)

    # Integrate from t=0 to t=1 using Euler method.
    t = 0.0
    dt = 1.0 / num_steps
    with torch.no_grad():
        for step in range(num_steps):
            t_tensor = torch.full((B,), t, device=device)
            # Euler integration: x <- x + dt * v(x, t)
            v = model(x, t_tensor, classes)
            x = x + dt * v
            t += dt

    return x

def plot_digit_samples(model, steps=20, fixed_sample=None):
    # Plot character generation 0-9
    classes = torch.arange(10).to(device)
    samples = generate_samples(model, classes, num_steps=steps, fixed_sample=fixed_sample)
    samples = samples.cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0], cmap='gray')
        ax.axis('off')
        ax.set_title(f"Digit {i}")
    plt.suptitle("Generated sample digits")
    plt.tight_layout()
    plt.savefig(f"CIFAR10_Experiments/Output/sample_digits-{steps}-steps.png")

def train_model():
    batch_size = 64
    num_epochs = 100
    learning_rate = 3e-4
    weight_decay = 1e-3
    layers = 3
    channels = 24
    dataloader = get_dataloader(batch_size)
    
    model = AttenUNet(layers, channels).to(device)
    # model = torch.compile(model)
    print("Starting training flow matching model...")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    train_flow_matching(model, dataloader, num_epochs=num_epochs, lr=learning_rate, wd=weight_decay)

    print("Generating samples...")
    steps = [1, 2, 3, 4, 5, 10, 20, 50]
    fixed_sample = torch.randn(1, 1, 28, 28, device=device)
    for step in steps:
        plot_digit_samples(model, step, fixed_sample)

if __name__ == '__main__':
    os.makedirs("CIFAR10_Experiments/Output", exist_ok=True)
    train_model()