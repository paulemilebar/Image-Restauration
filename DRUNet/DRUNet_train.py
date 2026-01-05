import os, time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from DRUNet import RandomPatchSigmaMapDataset, DRUNetSigmaMap

import torchvision.transforms.functional as TF

def train_drunet(
    clean_dir=r"./BSDS300/images/train",
    out_dir="weights_drunet_sigmap",
    patch=128,
    sigma_min=0.0,
    sigma_max=50.0,
    batch_size=16,
    steps_per_epoch=5000,
    max_epochs=10,
    lr0=1e-4,
    lr1=5e-5,
    plateau_epochs=5,
    log_every=50,
    num_workers=0,
    use_amp=False,         
    grad_clip=1.0
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = RandomPatchSigmaMapDataset(clean_dir, patch=patch, sigma_min=sigma_min, sigma_max=sigma_max)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True
    )

    model = DRUNetSigmaMap(in_nc=4, out_nc=3, nc=(64,128,256,512), nb=4).to(device).train()
    opt = Adam(model.parameters(), lr=lr0)
    loss_fn = nn.L1Loss()  # paper-like : L1 loss

    # AMP
    if device == "cuda":
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler(enabled=use_amp)
        autocast_ctx = lambda: autocast(enabled=use_amp)
    else:
        scaler = None
        autocast_ctx = lambda: torch.no_grad()  # dummy, pas utilisé

    best = float("inf")
    stagnant = 0
    using_lr1 = False

    for epoch in range(1, max_epochs + 1):
        running = 0.0
        start_t = time.time()

        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}/{max_epochs}", leave=True)
        it = iter(dl)

        for step in range(steps_per_epoch):
            try:
                inp, clean = next(it)
            except StopIteration:
                it = iter(dl)
                inp, clean = next(it)

            inp = inp.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if device == "cuda":
                with autocast_ctx():
                    pred = model(inp)
                    loss = loss_fn(pred, clean)
                if not torch.isfinite(loss):
                    print("[WARN] loss NaN/Inf, skip")
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(inp)
                loss = loss_fn(pred, clean)
                if not torch.isfinite(loss):
                    print("[WARN] loss NaN/Inf, skip")
                    continue
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            running += float(loss.item())

            if (step + 1) % log_every == 0:
                avg_so_far = running / (step + 1)
                elapsed = time.time() - start_t
                it_s = (step + 1) / max(elapsed, 1e-9)
                pbar.set_postfix({
                    "loss": f"{avg_so_far:.5f}",
                    "lr": f"{opt.param_groups[0]['lr']:.1e}",
                    "it/s": f"{it_s:.2f}"
                })

            pbar.update(1)

        pbar.close()
        avg = running / steps_per_epoch

        ckpt = os.path.join(out_dir, f"drunet_sigmap_epoch{epoch:02d}.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
        print(f"Epoch {epoch:02d} done | avg_loss={avg:.6f} | lr={opt.param_groups[0]['lr']:.1e}")

        # LR schedule
        if avg < best - 1e-7:
            best = avg
            stagnant = 0
        else:
            stagnant += 1

        if (not using_lr1) and stagnant >= plateau_epochs:
            for g in opt.param_groups:
                g["lr"] = lr1
            using_lr1 = True
            stagnant = 0
            print(f"Switch LR to {lr1}")

        if using_lr1 and stagnant >= plateau_epochs:
            print("Early stop: loss plateaued.")
            break

    final_path = os.path.join(out_dir, "drunet_sigmap_final.pth")
    torch.save({"model": model.state_dict()}, final_path)
    print(f"Saved: {final_path}")
    return final_path
