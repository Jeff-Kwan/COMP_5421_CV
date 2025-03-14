import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from AttenUNet import AttenUNet


def generate_trajectory(model, digit, x, t, device):
    digit = torch.tensor([digit], device=device, dtype=torch.long)
    model.eval()

    # Integrate from t=0 to t=1 using Euler method.
    outputs = [x]
    with torch.no_grad():
        for i in range(1, len(t)):
            t_tensor = torch.full((1,), t[i-1], device=device, dtype=torch.float)
            dt = t[i] - t[i-1]
            v = model(x, t_tensor, digit)
            x = x + dt * v  # Euler integration
            outputs.append(x)
    return outputs



def plot_trajectory(model, digit, device):
    '''Euler up to defined timestep and the 1-step prediction'''
    # Define multiple trajectories with different intermediate timesteps.
    t_list = [list(np.linspace(0, inter, 20))+[1] for inter in np.linspace(0.0, 1.0, 11)]
    trajectories = []
    x_init = torch.randn(1, 1, 28, 28, device=device)   # Fixed initial sample.
    for t in t_list:
        outputs = generate_trajectory(model, digit, x_init, t, device)
        # Convert each output to numpy.
        trajectories.append([output.cpu().numpy() for output in outputs])
    
    # Each row is a timestep and each column is a trajectory.
    n_rows = 3      # Each timestep a row.
    n_cols = len(trajectories)    # Each trajectory a column.
    trajectories = [[tr[0],tr[-2],tr[-1]] for tr in trajectories]
    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    
    # Plot each image in the grid.
    for i in range(n_rows):
        for j in range(n_cols):
            ax = plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
            image = trajectories[j][i][0, 0]
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            # Add a title to the top row for each column.
            if i == 0:
                ax.set_title(f"t = {t_list[j][-2]:.2f}")
    plt.suptitle(f"20-step Euler Integration then 1-step Prediction at t for digit {digit}")
    plt.tight_layout()
    plt.savefig(f'MNIST_Experiments/Output/trajectory_digit_{digit}.png')


def plot_sample_average():
    # Load the MNIST training dataset.
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Initialize sums and counts for each digit.
    sums = {i: torch.zeros(28, 28) for i in range(10)}
    counts = {i: 0 for i in range(10)}
    
    # Accumulate pixel values and counts for each digit.
    for image, label in train_dataset:
        sums[label] += image.squeeze(0)
        counts[label] += 1
    
    # Compute the average image for each digit.
    averages = {i: sums[i] / counts[i] for i in range(10)}
    
    # Plot the average images in a 2x5 grid.
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i in range(10):
        ax = axes[i // 5][i % 5]
        ax.imshow(averages[i], cmap='gray')
        ax.set_title(f"Digit {i}")
        ax.axis('off')
    plt.suptitle("Sample Averages for Each Digit")
    plt.tight_layout()
    plt.savefig('MNIST_Experiments/Output/MNIST_sample_averages.png')




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttenUNet(layers=3, channels=16).to(device)
    model.load_state_dict(torch.load('MNIST_Experiments/Output/MNIST_flow_model.pth', weights_only=True))

    # plot_sample_average()
    for digit in range(10):
        plot_trajectory(model, digit, device)