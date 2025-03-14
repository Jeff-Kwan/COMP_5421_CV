import torch
from torchvision import transforms, datasets
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple flow matching network
class ResBlock(nn.Module):
    def __init__(self, channels, classes):
        super(ResBlock, self).__init__()
        self.num_blocks = 2
        self.channels = channels
        self.blocks = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(channels, channels*4, 3, 1, 1, bias=False),
            nn.GroupNorm(1, channels*4, affine=False),
            nn.SiLU(),
            nn.Conv2d(channels*4, channels, 3, 1, 1, bias=False))
        for _ in range(self.num_blocks)])
        self.norm = nn.ModuleList([nn.GroupNorm(1, channels, affine=False) for _ in range(self.num_blocks)])
        self.c_embed = nn.Embedding(classes, 2*channels)

    def forward(self, x, c):
        c_mu, c_logvar = self.c_embed(c).view(-1, self.channels*2, 1, 1).chunk(2, dim=1)
        c_std = torch.exp(0.5 * c_logvar)
        x = c_std * x + c_mu
        for i in range(self.num_blocks):
            x = self.norm[i](x + self.blocks[i](x))
        return x

class FlowMatchingNet(nn.Module):
    def __init__(self, layers=3, channels=16):
        super(FlowMatchingNet, self).__init__()
        self.layers = layers - 1    # Original resolution is 1 layer
        self.channels = channels
        self.classes = 10

        self.in_conv = nn.Conv2d(1, self.channels, 3, 1, 1, bias=False)
        self.t_embeds = nn.Linear(2, self.channels, bias=False)
        self.in_norm = nn.GroupNorm(1, self.channels, affine=False)

        # Encoder 
        self.encoder_blocks = nn.ModuleList([ResBlock(self.channels, self.classes) for _ in range(self.layers)])
        self.downs = nn.ModuleList([nn.Conv2d(self.channels, self.channels, 2, 2, 0, bias=None) 
                                    for _ in range(self.layers)])

        # Bottleneck
        self.attention = nn.ModuleList([nn.MultiheadAttention(self.channels, 2) for _ in range(2)])
        self.ffn = nn.ModuleList([nn.Sequential(
            nn.Linear(self.channels, self.channels*4),
            nn.SiLU(),
            nn.Linear(self.channels*4, self.channels)) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.channels) for _ in range(2*2)])
        self.c_embed = nn.Embedding(self.classes, 2*channels)
        

        # Decoder
        self.ups = nn.ModuleList([nn.ConvTranspose2d(self.channels, self.channels, 2, 2, 0, bias=False)
                        for _ in range(self.layers)])
        self.merges = nn.ModuleList([nn.Conv2d(self.channels*2, self.channels, 1, 1, 0, bias=False)
                        for _ in range(self.layers)])
        self.decoder_blocks = nn.ModuleList([ResBlock(self.channels, self.classes) for _ in range(self.layers)])

        self.out_norm = nn.GroupNorm(1, self.channels, affine=False)
        self.out = nn.Conv2d(self.channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, x, t, c):
        x = self.in_conv(x)
        x = x + self.t_embeds(torch.stack([t, 1-t], dim=-1)).view(-1, self.channels, 1, 1)
        x = self.in_norm(x)

        # Encoder
        skips = []
        for i in range(self.layers):
            x = self.encoder_blocks[i](x, c)
            skips.append(x)
            x = self.downs[i](x)

        # Bottleneck
        B, C, H, W = x.size()
        c_mu, c_logvar = self.c_embed(c).unsqueeze(1).chunk(2, dim=-1)
        c_std = torch.exp(0.5 * c_logvar)
        x = x.view(B, C, H*W).transpose(1,2).contiguous()
        x = c_mu * x + c_std
        for i in range(2):
            x = self.norms[2*i](x + self.attention[i](x, x, x)[0])
            x = self.norms[2*i+1](x + self.ffn[i](x))
        x = x.transpose(1,2).view(B, C, H, W)

        # Decoder
        for i in range(self.layers):
            x = self.ups[i](x)
            x = self.merges[i](torch.cat([x, skips[-i-1]], dim=1))
            x = self.decoder_blocks[i](x, c)

        x = self.out(self.out_norm(x))
        return x

# Load MNIST dataset
def get_dataloader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
            images = images.to(device)  # MNIST images: [batch, 1, 28,28]
            classes = classes.to(device)

            batch = images.shape[0]

            # Sample a random scalar time t uniformly in [0, 1] for each sample.
            t = torch.rand(batch, device=device)

            # Sample noise images from Normal distribution.
            noise = torch.randn_like(images)

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

# Generate new samples via Euler integration using the learned flow field.
def generate_samples(model, classes, num_steps=100):
    model.eval()
    B = classes.size(0)
    dt = 1.0 / num_steps

    # Start from pure noise (i.e. at time t=0).
    x = torch.randn(B, 1, 28, 28, device=device)

    # Integrate from t=0 to t=1 using Euler method.
    t = 0.0
    with torch.no_grad():
        for step in range(num_steps):
            t_tensor = torch.full((B,), t, device=device)
            # Euler integration: x <- x + dt * v(x, t)
            v = model(x, t_tensor, classes)
            x = x + dt * v
            t += dt

    return x

def plot_samples(model):
    # Plot character generation 0-9
    classes = torch.arange(10).to(device)
    samples = generate_samples(model, classes, num_steps=50)
    samples = samples.cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0], cmap='gray')
        ax.axis('off')
        ax.set_title(f"Digit {i}")
    plt.suptitle("Generated sample digits")
    plt.tight_layout()
    plt.show()

def main():
    batch_size = 128
    num_epochs = 50
    dataloader = get_dataloader(batch_size)
    
    model = FlowMatchingNet().to(device)
    print("Starting training flow matching model...")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    train_flow_matching(model, dataloader, num_epochs=num_epochs, lr=1e-3)

    print("Generating samples...")
    plot_samples(model)

if __name__ == '__main__':
    main()