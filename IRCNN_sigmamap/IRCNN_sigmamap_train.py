import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
import time
from tqdm.auto import tqdm
from IRCNN_sigmamap import RandomPatchSigmaMapDataset, IRCNNSigmaMap


# Train
def train(
    clean_dir=r"./BSDS300/images/train",
    out_dir="weights_ircnn_sigmap",
    patch=35,
    sigma_min=0.01,
    sigma_max=50.0,
    batch_size=8,
    steps_per_epoch=1000,
    max_epochs=10,
    lr0=1e-3,
    lr1=1e-4,
    plateau_epochs=5,
    log_every=50
):
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = RandomPatchSigmaMapDataset(clean_dir, patch=patch, sigma_min=sigma_min, sigma_max=sigma_max)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,              
        pin_memory=(device == "cuda"),
        drop_last=True
    )

    model = IRCNNSigmaMap().to(device).train()
    opt = Adam(model.parameters(), lr=lr0)
    loss_fn = nn.MSELoss()
    scaler = GradScaler(enabled=(device == "cuda"))

    best = float("inf")
    stagnant = 0
    using_lr1 = False

    global_step = 0

    for epoch in range(1, max_epochs + 1):
        running = 0.0
        start_t = time.time()

        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}/{max_epochs}", leave=True)
        for step, (inp, clean) in enumerate(dl):
            if step >= steps_per_epoch:
                break

            inp = inp.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            noisy = inp[:, :3, :, :]
            target_noise = noisy - clean

            with autocast(enabled=(device == "cuda")):
                pred_clean = model(inp)
                pred_noise = noisy - pred_clean
                loss = loss_fn(pred_noise, target_noise)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            global_step += 1

            # logs
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

        avg = running / max(1, steps_per_epoch)
        ckpt = os.path.join(out_dir, f"ircnn_sigmap_epoch{epoch:02d}.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)

        print(f"Epoch {epoch:02d} done | avg_loss={avg:.6f} | lr={opt.param_groups[0]['lr']:.1e}")

        # LR schedule + early stop like before
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

    final_path = os.path.join(out_dir, "ircnn_sigmap_final.pth")
    torch.save({"model": model.state_dict()}, final_path)
    print(f"Saved: {final_path}")


train()