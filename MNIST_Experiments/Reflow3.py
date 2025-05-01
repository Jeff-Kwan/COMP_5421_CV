# Reflow3_shuffled.py
# ---------------------------------------------------------------------------
import os
import threading
import queue
import time
import random

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from AttnUNet2 import AttenUNet


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


def generate_synthetic(model, device_gen, device_train, q, args, stop_event, reload_event):
    """
    Continuously generate (x0, x1, cls) tuples.
    Whenever reload_event is set, reloads the latest checkpoint.
    """
    batch_gen   = args["batch_gen"]
    batch_train = args["batch_train"]
    num_steps   = args["gen_steps"]
    num_classes = args.get("num_classes", 10)
    dt          = 1.0 / num_steps
    ckpt_path   = os.path.join(args["save_path"], "MNIST_bootstrapping-rectified.pth")

    model = model.to(device_gen)

    while not stop_event.is_set():
        if reload_event.is_set():
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state)
                model.to(device_gen)
                print(f"[{device_gen}] Reloaded weights from {ckpt_path}")
            except Exception as e:
                print(f"[{device_gen}] Failed to reload weights: {e}")
            reload_event.clear()

        cls = torch.randint(0, num_classes, (batch_gen,), device=device_gen)
        x0  = torch.randn(batch_gen, 1, 28, 28, device=device_gen)
        x   = x0.clone()
        t   = 0.0

        for _ in range(num_steps):
            t_tensor = torch.full((batch_gen,), t, device=device_gen)
            v        = model(x, t_tensor, cls)
            x       += dt * v
            t       += dt
        x1 = x.clamp(-1.0, 1.0)

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
    return


def train_2rectified_flow(model, device_train, q, args, stop_event, mnist_loader, reload_event):
    """
    Alternate real MNIST / synthetic steps; after each checkpoint save,
    write both numbered and 'latest' checkpoint and set reload_event.
    """
    max_steps   = args["train_steps"]
    lr          = args["lr"]
    wd          = args["wd"]
    save_path   = args["save_path"]
    gen_steps   = args["gen_steps"]
    num_classes = args.get("num_classes", 10)
    ckpt_latest = os.path.join(save_path, "MNIST_bootstrapping-rectified.pth")

    os.makedirs(save_path, exist_ok=True)

    model = model.to(device_train).train()
    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)
    mse       = nn.MSELoss()

    real_iter = iter(mnist_loader)
    pbar = tqdm(total=max_steps, desc=f"Training (on {device_train})", unit="step")

    for step in range(max_steps):
        # pick real or synthetic
        if torch.rand(1).item() < (max_steps - step) / max_steps:
            try:
                imgs, labels = next(real_iter)
            except StopIteration:
                # Start a new "epoch" of MNIST — this re-seeds the shuffle
                real_iter = iter(mnist_loader)
                imgs, labels = next(real_iter)

            x1  = imgs.to(device_train, non_blocking=True)
            cls = labels.to(device_train, non_blocking=True)
            x0  = torch.randn_like(x1, device=device_train)
        else:
            x0, x1, cls = q.get()

        b = x0.size(0)
        t = torch.rand(b, device=device_train)
        x_t = (1 - t).view(-1,1,1,1)*x0 + t.view(-1,1,1,1)*x1
        tgt = x1 - x0

        pred = model(x_t, t, cls)
        loss = mse(pred, tgt)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        pbar.set_postfix(
            step=step+1,
            loss=f"{loss.item():.4f}",
            grad_norm=f"{grad_norm:.4f}"
        )
        pbar.update(1)

        if (step + 1) % 200 == 0 or (step + 1) == max_steps:
            torch.save(model.state_dict(), ckpt_latest)
            reload_event.set()

            samples = generate_samples(
                model,
                torch.arange(num_classes, device=device_train),
                gen_steps,
                device_train
            )
            save_image(
                samples,
                os.path.join(save_path, "samples-bootstrapping-rect.png"),
                nrow=5, normalize=True, value_range=(-1,1)
            )

    pbar.close()
    stop_event.set()
    print("✓ 2-Rectified-flow training complete.")


def main(args):
    # GPUs
    ngpus = torch.cuda.device_count()
    args["queue_size"] = int(2 * args['batch_gen'] / args['batch_train'] * ngpus)
    if ngpus < 2:
        raise RuntimeError(f"Need at least 2 GPUs, found {ngpus}")
    device_train = torch.device("cuda:0")
    device_gens  = [torch.device(f"cuda:{i}") for i in range(1, ngpus)]

    # instantiate train model (from scratch)
    train_model = AttenUNet(
        layers=args["layers"],
        channels=args["channels"],
        heads=args["heads"]
    )
    print(f"Instantiated new train-model on {device_train}")

    # prepare an initial checkpoint so generators have something to load
    os.makedirs(args["save_path"], exist_ok=True)
    init_state = train_model.state_dict()
    init_ckpt  = os.path.join(args["save_path"], "MNIST_bootstrapping-rectified.pth")
    torch.save(init_state, init_ckpt)
    print(f"Saved initial bootstrap checkpoint to {init_ckpt}")

    # build generator models
    gen_models = []
    for dev in device_gens:
        m = AttenUNet(
            layers=args["layers"],
            channels=args["channels"],
            heads=args["heads"]
        )
        m.load_state_dict(init_state)
        gen_models.append(m)
        print(f"Instantiated generator-model on {dev}")

    # ─── here’s the change: shuffle=True so that each new iter(mnist_loader)
    # will re-shuffle the real MNIST images ────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    mnist_loader = DataLoader(
        datasets.MNIST(
            root=args["data_path"],
            train=True,
            transform=transform,
            download=True
        ),
        batch_size=args["batch_train"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    print(f"MNIST loader ready, batch_size={args['batch_train']}, shuffle=True")

    # shared queue & events
    q            = queue.Queue(maxsize=args["queue_size"])
    stop_event   = threading.Event()
    reload_event = threading.Event()

    # start generator threads
    for model, dev in zip(gen_models, device_gens):
        t = threading.Thread(
            target=generate_synthetic,
            args=(model, dev, device_train, q, args, stop_event, reload_event),
            daemon=True
        )
        t.start()
        print(f"Started generator thread on {dev}")

    # prime queue
    while q.qsize() < args["queue_size"]:
        time.sleep(0.1)

    # start trainer
    train_thread = threading.Thread(
        target=train_2rectified_flow,
        args=(train_model, device_train, q, args, stop_event, mnist_loader, reload_event),
    )
    train_thread.start()
    train_thread.join()

    print("All done.")


if __name__ == "__main__":
    args = {
        "layers":           3,
        "channels":        16,
        "heads":            2,
        "batch_gen":      1024,
        "batch_train":    128,
        "gen_steps":       20,
        "train_steps":  20000,
        "lr":           3e-4,
        "wd":            1e-2,
        "num_classes":    10,
        "save_path":       "MNIST_Experiments/Output/Reflow",
        "data_path":       "data/",
    }

    torch.backends.cudnn.enabled        = True
    torch.backends.cudnn.benchmark      = True
    torch.backends.cudnn.allow_tf32     = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.enable_math_sdp(False)

    main(args)
