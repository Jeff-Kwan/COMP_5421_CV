# Reflow_single_gpu.py
# Single–GPU Rectified-Flow Training on MNIST
# ----------------------------------------------------------------------------
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms
from tqdm import tqdm

from AttnUNet2 import AttenUNet
import signal
import sys
# ----------------------------------------------------------------------------
def get_mnist_dataloader(batch_size: int, num_workers: int = 2) -> DataLoader:
    """Load MNIST using the standard DataLoader."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

# Generate new samples via Euler integration using the learned flow field.
def generate_samples(model, classes, num_steps, device):
    model.eval()
    B = classes.size(0)
    x = torch.randn(B, 1, 28, 28, device=device)

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


# ----------------------------------------------------------------------------
def train_rectified_flow(
    net:         nn.Module,
    dataloader:  DataLoader,
    device:      torch.device,
    num_epochs:  int,
    lr:          float    = 3e-4,
    wd:          float    = 1e-4,
    save_path:   str      = "MNIST_Experiments/Output/Reflow",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    net = net.to(device).train()
    opt       = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs * len(dataloader))
    mse       = nn.MSELoss()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} ({device})", unit="batch")
        
        for x1, cls in pbar:
            # move batch → device
            x1  = x1.to(device, non_blocking=True)
            cls = cls.to(device, non_blocking=True)
            b   = x1.size(0)

            # sample initial noise and time
            x0 = torch.randn_like(x1, device=device)
            t  = torch.rand(b, device=device)

            # build interpolation x_t and true velocity v = x1 - x0
            x_t    = (1 - t).view(-1,1,1,1) * x0 + t.view(-1,1,1,1) * x1
            target = x1 - x0

            # forward + loss
            pred = net(x_t, t, cls)                    # predict velocity
            loss = mse(pred, target)

            # backward + step
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", norm=f"{norm:.3f}")

        save_image(generate_samples(net, torch.arange(10).to(device), 20, device), 
               f"{save_path}/samples.png", 
               nrow=5, normalize=True, value_range=(-1, 1))
        torch.save(net.state_dict(), f"{save_path}/MNIST_1-rectified.pth")
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {epoch_loss / len(dataloader):.4f}")
    
    print(f"✓ 1-Rectified-flow model training complete")


# ----------------------------------------------------------------------------
def main():
    # choose cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # hyper-parameters
    args = {
        "layers":    3,
        "channels":  16,
        "heads":     2,
        "batch":     128,
        "epochs":    100,
        "lr":        3e-4,
        "wd":        1e-2,
        "save_path":"MNIST_Experiments/Output/Reflow",
    }

    # data + model
    loader = get_mnist_dataloader(batch_size=args["batch"])
    net    = AttenUNet(layers=args["layers"], channels=args["channels"], heads=args["heads"])
    print(f"✓ Model: {net.__class__.__name__} ({args['layers']} layers, {args['channels']} channels)")
    print(f"Model Parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6:.2f}M")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # train
    train_rectified_flow(
        net=net,
        dataloader=loader,
        device=device,
        num_epochs=args["epochs"],
        lr=args["lr"],
        wd=args["wd"],
        save_path=args["save_path"],
    )

if __name__ == "__main__":
    def handle_sigint(signal, frame):
        print("\nExiting gracefully...")
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)
    main()
