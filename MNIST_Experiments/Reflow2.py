# Reflow2.py
# ----------------------------------------------------------------------------
import os
import threading
import queue
import time

import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm

from AttnUNet2 import AttenUNet

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
def generate_synthetic(model, device_gen, q, args, stop_event):
    """
    Continuously generate (x0, x1, cls) tuples on device_gen (cuda:1) and enqueue them (on CPU).
    """
    batch_size  = args["batch"]
    num_steps   = args["gen_steps"]
    num_classes = args.get("num_classes", 10)

    model.to(device_gen).eval()

    dt = 1.0 / num_steps
    while not stop_event.is_set():
        # 1) sample a random class vector and initial noise x0
        cls = torch.randint(0, num_classes, (batch_size,), device=device_gen)
        x0  = torch.randn(batch_size, 1, 28, 28, device=device_gen)
        x   = x0.clone()
        t   = 0.0

        # 2) Euler integrate from t=0→1 to get x1
        with torch.no_grad():
            for _ in range(num_steps):
                t_tensor = torch.full((batch_size,), t, device=device_gen)
                v        = model(x, t_tensor, cls)
                x       += dt * v
                t       += dt
        x1 = torch.clamp(x, -1.0, 1.0)

        # 3) move to CPU and enqueue
        q.put((x0.cpu(), x1.cpu(), cls.cpu()))

    # clean exit if stopped
    return

# ----------------------------------------------------------------------------
def train_2rectified_flow(model, device_train, q, args, stop_event):
    """
    Consume (x0, x1, cls) from the queue and train a new flow model on device_train (cuda:0)
    for exactly `train_steps` iterations.
    """
    max_steps = args["train_steps"]
    lr        = args["lr"]
    wd        = args["wd"]
    save_path = args["save_path"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.to(device_train).train()
    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)
    mse       = nn.MSELoss()

    pbar = tqdm(total=max_steps, desc=f"Training (on {device_train})", unit="step")
    step = 0
    while step < max_steps:
        x0_cpu, x1_cpu, cls_cpu = q.get()
        # move batch → device_train
        x0  = x0_cpu.to(device_train, non_blocking=True)
        x1  = x1_cpu.to(device_train, non_blocking=True)
        cls = cls_cpu.to(device_train, non_blocking=True)
        b   = x0.size(0)

        # sample random time t∈[0,1]
        t    = torch.rand(b, device=device_train)
        x_t  = (1 - t).view(-1,1,1,1) * x0 + t.view(-1,1,1,1) * x1
        tgt  = x1 - x0

        # forward + loss
        pred = model(x_t, t, cls)
        loss = mse(pred, tgt)

        # backward + step
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", norm=f"{norm:.4f}")
        pbar.update(1)

        if step % 200 == 0 or step == max_steps-1:
            save_image(generate_samples(model, torch.arange(10).to(model.device), 20, model.device), 
               f"{save_path}/samples-2-rect.png", 
               nrow=5, normalize=True, value_range=(-1, 1))
            torch.save(model.state_dict(), f"{save_path}/MNIST_2-rectified.pth")

    pbar.close()
    # tell the generator to stop
    stop_event.set()

    # save the distilled 2-rectified model
    print(f"✓ 2-Rectified-flow training complete. Model saved to {save_path}")

# ----------------------------------------------------------------------------
def main(args):
    # select devices
    device_gen   = torch.device("cuda:1")
    device_train = torch.device("cuda:0")

    # -- load pretrained 1-rectified model for generation
    gen_model = AttenUNet(layers=args["layers"],
                          channels=args["channels"],
                          heads=args["heads"])
    gen_model.load_state_dict(
        torch.load(args["pretrained_path"], map_location=device_gen)
    )
    print(f"Loaded pretrained model from {args['pretrained_path']} onto {device_gen}")

    # -- instantiate a fresh model for 2-rectified training
    train_model = AttenUNet(layers=args["layers"],
                            channels=args["channels"],
                            heads=args["heads"])
    print(f"Instantiated new model for 2-rectified flow on {device_train}")

    # prepare a bounded queue on CPU
    q = queue.Queue(maxsize=args.get("queue_size", 20))
    stop_event = threading.Event()

    # start generator thread (daemon so it dies if main thread exits)
    gen_thread = threading.Thread(
        target=generate_synthetic,
        args=(gen_model, device_gen, q, args, stop_event),
        daemon=True
    )
    # start trainer thread
    train_thread = threading.Thread(
        target=train_2rectified_flow,
        args=(train_model, device_train, q, args, stop_event),
    )

    gen_thread.start()
    # wait for 5s heuristically
    time.sleep(5)
    train_thread.start()
    train_thread.join()   # wait until training finishes

    print("All done.")

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Hyper‐parameters & paths
    args = {
        "layers":          3,
        "channels":       16,
        "heads":           2,
        "batch":        128,    # batch size for both gen & train
        "gen_steps":      20,    # Euler integration steps for generation
        "train_steps":  20000,   # total batch‐steps to train on cuda:0
        "lr":          3e-4,
        "wd":           1e-2,
        "queue_size":    32,    # max buffered batches in CPU queue
        "num_classes":   10,
        "pretrained_path": "MNIST_Experiments/Output/Reflow/MNIST_1-rectified.pth",
        "save_path":       "MNIST_Experiments/Output/Reflow/MNIST_2-rectified.pth",
    }
    # enable fast CuDNN
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    main(args)
