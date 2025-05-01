# Reflow2.py  (direct GPU-to-GPU queuing)
# -----------------------------------------------------------------------------
import os
import threading
import queue
import time

import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm

from AttnUNet2 import AttenUNet
# -----------------------------------------------------------------------------
def generate_samples(model, classes, num_steps, device):
    model.eval()
    B = classes.size(0)
    x = torch.randn(B, 1, 28, 28, device=device)

    t = 0.0
    dt = 1.0 / num_steps
    with torch.no_grad():
        for _ in range(num_steps):
            t_tensor = torch.full((B,), t, device=device)
            v = model(x, t_tensor, classes)
            x = x + dt * v
            t += dt
    return x.clamp(-1.0, 1.0)


# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_synthetic(model, device_gen, device_train, q, args, stop_event):
    """
    Continuously generate (x0, x1, cls) tuples on device_gen (e.g. cuda:0) and
    enqueue them directly on device_train (e.g. cuda:1), chunked into smaller
    train batches for VRAM-efficient training.
    """
    batch_gen   = args["batch_gen"]      # e.g. 1024
    batch_train = args["batch_train"]    # e.g. 128
    num_steps   = args["gen_steps"]
    num_classes = args.get("num_classes", 10)

    model.to(device_gen).eval()
    model = torch.compile(model)
    dt = 1.0 / num_steps

    while not stop_event.is_set():
        # 1) sample a big batch on the generator
        cls = torch.randint(0, num_classes, (batch_gen,), device=device_gen)
        x0  = torch.randn(batch_gen, 1, 28, 28, device=device_gen)
        x   = x0.clone()
        t   = 0.0

        # 2) Euler integrate from t=0 → 1
        for _ in range(num_steps):
            t_tensor = torch.full((batch_gen,), t, device=device_gen)
            v        = model(x, t_tensor, cls)
            x       += dt * v
            t       += dt
        x1 = x.clamp(-1.0, 1.0)

        # 3) direct GPU→GPU copy of chunks, no CPU hop
        for x0_c, x1_c, cls_c in zip(
                x0.split(batch_train),
                x1.split(batch_train),
                cls.split(batch_train)
        ):
            q.put((
                x0_c.to(device_train, non_blocking=True),
                x1_c.to(device_train, non_blocking=True),
                cls_c.to(device_train, non_blocking=True),
            ))

    # clean exit if stopped
    return


# -----------------------------------------------------------------------------
def train_2rectified_flow(model, device_train, q, args, stop_event):
    """
    Consume (x0, x1, cls) from the queue and train a new flow model on device_train
    (e.g. cuda:1) for exactly `train_steps` iterations.
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
        # tensors already live on device_train
        x0, x1, cls = q.get()
        b = x0.size(0)

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

        # periodically save samples & checkpoint
        if step % 100 == 0 or step == max_steps:
            samples = generate_samples(model,
                                       torch.arange(args.get("num_classes",10), device=device_train),
                                       args["gen_steps"], device_train)
            save_image(samples,
                       f"{save_path}/samples-2-rect.png",
                       nrow=5, normalize=True, value_range=(-1,1))
            torch.save(model.state_dict(),
                       f"{save_path}/MNIST_2-rectified.pth")

    pbar.close()
    stop_event.set()
    print("✓ 2-Rectified-flow training complete.")


# -----------------------------------------------------------------------------
def main(args):
    device_gen   = torch.device("cuda:0")
    device_train = torch.device("cuda:1")

    # -- load pretrained 1-rectified model for generation
    gen_model = AttenUNet(layers=args["layers"],
                          channels=args["channels"],
                          heads=args["heads"])
    gen_model.load_state_dict(
        torch.load(args["pretrained_path"], map_location=device_gen)
    )
    print(f"Loaded pretrained model from {args['pretrained_path']} onto {device_gen}")

    # -- instantiate fresh model for 2-rectified training
    train_model = AttenUNet(layers=args["layers"],
                            channels=args["channels"],
                            heads=args["heads"])
    print(f"Instantiated new model for 2-rectified flow on {device_train}")

    # prepare a bounded queue on CPU (stores Python refs to GPU tensors)
    q = queue.Queue(maxsize=args.get("queue_size", 20))
    stop_event = threading.Event()

    # start generator thread (daemon so it dies if main thread exits)
    gen_thread = threading.Thread(
        target=generate_synthetic,
        args=(gen_model, device_gen, device_train, q, args, stop_event),
        daemon=True
    )
    # start trainer thread
    train_thread = threading.Thread(
        target=train_2rectified_flow,
        args=(train_model, device_train, q, args, stop_event),
    )

    gen_thread.start()
    # wait for generator to fill queue before starting trainer
    while q.qsize() < args["queue_size"]:
        time.sleep(0.1)
    train_thread.start()
    train_thread.join()

    print("All done.")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Hyper-parameters & paths
    args = {
        "layers":           3,
        "channels":        16,
        "heads":            2,

        # generator ↔ trainer interface
        "batch_gen":      1024,   # samples per generator pass
        "batch_train":    128,    # samples per training step

        "gen_steps":       20,    # Euler steps for generation
        "train_steps":  10000,    # total training iterations
        "lr":           3e-4,
        "wd":            1e-2,

        "queue_size":    512,     # max buffered sub-batches
        "num_classes":    10,
        "pretrained_path": "MNIST_Experiments/Output/Reflow/MNIST_1-rectified.pth",
        "save_path":       "MNIST_Experiments/Output/Reflow",
    }

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.enable_math_sdp(False)

    main(args)
