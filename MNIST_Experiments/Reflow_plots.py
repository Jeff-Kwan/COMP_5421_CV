import torch
from torch import nn
from torchvision.utils import save_image
import os

from AttnUNet2 import AttenUNet

@torch.no_grad()
def generate_samples(model, classes, num_steps, device, save_dir, file_name, fixed_samples=None):
    model.eval()
    B = classes.size(0)
    if fixed_samples is not None:
        x = fixed_samples
    else:
        x = torch.randn(B, 1, 28, 28, device=device)
    t = 0.0
    dt = 1.0 / num_steps
    for _ in range(num_steps):
        t_tensor = torch.full((B,), t, device=device)
        v = model(x, t_tensor, classes)
        x = x + dt * v
        t += dt
    x = x.clamp(-1.0, 1.0)
    save_image(x, os.path.join(save_dir, file_name), nrow=5, normalize=True, value_range=(-1,1))
    return 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reflow_dir = "MNIST_Experiments/Output/Reflow"
subdir = "bootstrap2-rectified"
output_dir = os.path.join(reflow_dir, subdir)
os.makedirs(output_dir, exist_ok=True)

model = AttenUNet(3, 16, 2)
model.load_state_dict(torch.load(f"{reflow_dir}/MNIST_{subdir}.pth"))
model.to(device)
model.eval()

steps = [1, 2, 3, 4, 5, 10]
sample = torch.randn(10, 1, 28, 28, device=device)
classes = torch.arange(10).long().to(device)
for step in steps:
    generate_samples(model, classes, step, device, output_dir, 
                    f"MNIST{step}step_{subdir}.png", sample)
