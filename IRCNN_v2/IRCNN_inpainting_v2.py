import os, math, random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from IRCNN_v2 import IRCNNModelManager
import matplotlib.pyplot as plt


def randn_like_compat(x: torch.Tensor, seed: int):
    try:
        g = torch.Generator(device=x.device).manual_seed(seed)
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g)
    except TypeError:
        torch.manual_seed(seed)
        return torch.randn_like(x)


def make_random_rect_mask(H: int, W: int, missing_ratio: float = 0.2, seed: int = 0,
                          min_rect: int = 2, max_rect: int = 7) -> torch.Tensor:
    rng = random.Random(seed)
    mask = np.ones((H, W), dtype=np.float32)
    target_missing = int(missing_ratio * H * W)
    missing = 0
    while missing < target_missing:
        rh = rng.randint(min_rect, min(max_rect, H))
        rw = rng.randint(min_rect, min(max_rect, W))
        y0 = rng.randint(0, H - rh)
        x0 = rng.randint(0, W - rw)

        before = mask[y0:y0+rh, x0:x0+rw].sum()
        mask[y0:y0+rh, x0:x0+rw] = 0.0
        after = mask[y0:y0+rh, x0:x0+rw].sum()
        missing += int(before - after)

    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


def shepard_initialize_rgb(y01: torch.Tensor, M01: torch.Tensor, window: int = 9, p: float = 2.0) -> torch.Tensor:
    assert window % 2 == 1
    y = y01.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()   # (H,W,3)
    m = M01.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)  # (H,W)
    H, W, _ = y.shape
    out = y.copy()
    wing = window // 2

    offsets, weights = [], []
    for dy in range(-wing, wing + 1):
        for dx in range(-wing, wing + 1):
            if dy == 0 and dx == 0:
                continue
            d = (abs(dy) ** p + abs(dx) ** p)
            if d == 0:
                continue
            offsets.append((dy, dx))
            weights.append(1.0 / d)
    weights = np.array(weights, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            if m[i, j] == 0:
                vals, wts = [], []
                for (dy, dx), w0 in zip(offsets, weights):
                    ii, jj = i + dy, j + dx
                    if 0 <= ii < H and 0 <= jj < W and m[ii, jj] == 1:
                        vals.append(y[ii, jj])
                        wts.append(w0)
                if len(wts) > 0:
                    wts = np.array(wts, dtype=np.float32)
                    wts = wts / (wts.sum() + 1e-12)
                    vals = np.stack(vals, axis=0)
                    out[i, j] = (vals * wts[:, None]).sum(axis=0)

    z0 = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
    return z0.clamp(0, 1)


def solve_fidelity_inpainting(y, z, mask, mu, eps=1e-8):
    return ((mask * y + mu * z) / (mask + mu + eps)).clamp(0,1)


def save_img01(t: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    TF.to_pil_image(t.squeeze(0).clamp(0, 1).cpu()).save(path)
    

# --- PSNR ---
def calculate_psnr(x, y, eps=1e-8):
    mse = torch.mean((x - y)**2).item()
    return 10 * math.log10(1.0 / (mse + eps))


def test_inpaint_pnp(
    image_path,
    model_dir=r"./IRCNN_v2/weights_ircnn_experts_colab",
    out_dir=r"./IRCNN_v2/tests_inpainting",
    missing_ratio=0.15, 
    seed=0,
    n_iter=15, 
    lambda_pnp = 3,
    noise_sigma=5.0, 
    sigma_n_pixels=2,
    add_small_noise_in_holes = 0.01,
    shepard_window=21, 
    shepard_p=2.0
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Device:", device)

    gt = TF.to_tensor(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device).clamp(0, 1)
    _, _, H, W = gt.shape
    
    M = make_random_rect_mask(H, W, missing_ratio=missing_ratio, seed=seed).to(device=device, dtype=gt.dtype)
    M3 = M.repeat(1, 3, 1, 1)
    y = (gt * M3).clamp(0, 1)

    sigma_obs = float(noise_sigma) / 255.0
    if sigma_obs > 0:
        y = (M3 * (y + randn_like_compat(y, seed=seed+123) * sigma_obs)).clamp(0, 1)
    
    # init Shepard
    x = shepard_initialize_rgb(y, M, window=shepard_window, p=shepard_p).to(device=device, dtype=y.dtype)
    if add_small_noise_in_holes > 0:
        x = (x + (1.0 - M3) * add_small_noise_in_holes * randn_like_compat(x, seed=seed)).clamp(0, 1)
    
    sigmas_k = np.logspace(np.log10(49), np.log10(sigma_n_pixels), n_iter)
    manager = IRCNNModelManager(model_dir, device=device)
    
    psnr_degraded = calculate_psnr(x, gt)
    best_psnr = -float("inf")
    best_x = x.clone()
    best_mu = 0
    psnr_list = []
    mu_list = []
    
    for sk in sigmas_k:
        mu = lambda_pnp/ (sk**2)
        # Solve fidelity via CG
        x = solve_fidelity_inpainting(y, x, M3, mu)
        # Prior CNN
        expert = manager.get_expert(sk)
        with torch.no_grad():
            x = expert.denoise(x).clamp(0,1) 
        # hard enforce known pixels
        x = (M3 * y + (1.0 - M3) * x).clamp(0, 1)
        
        # PSNR intermédiaire
        p = calculate_psnr(x, gt)
        psnr_list.append(p)
        mu_list.append(mu)
        if p > best_psnr:
            best_psnr = p
            best_x = x.clone()
            best_mu = mu

    improvement = best_psnr - psnr_degraded
    print(f"PSNR Degraded : {psnr_degraded:.2f} dB")
    print(f"PSNR Restored : {best_psnr:.2f} dB for mu = {best_mu:.2f}")
    print(f"Gain         : {improvement:.2f} dB")

    # 4. Sauvegarde et résultats
    img_id = os.path.splitext(os.path.basename(image_path))[0]
    TF.to_pil_image(gt.squeeze(0)).save(os.path.join(out_dir, f"{img_id}_cleaned.png"))
    TF.to_pil_image(M3.squeeze(0)).save(os.path.join(out_dir, f"{img_id}_masked.png"))
    TF.to_pil_image(y.squeeze(0)).save(os.path.join(out_dir, f"{img_id}_masked_noisy.png"))
    TF.to_pil_image(best_x.squeeze(0)).save(os.path.join(out_dir, f"{img_id}_restaured.png"))
    print(f"Terminé ! Images sauvegardées dans le dossier '{out_dir}'.")
    plt.plot(mu_list,psnr_list,'o')
    plt.plot([0],[psnr_degraded], color='orange', marker='x')
    plt.axvline(x=best_mu, color='red', linestyle='--', linewidth=2)
    plt.show()


import random

# Choix aléatoire de l'image test
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
clean_dir = r"./BSDS300/images/test" 
paths = [
            os.path.join(clean_dir, f) for f in os.listdir(clean_dir)
            if f.lower().endswith(IMG_EXT)
        ]

assert len(paths) > 0, f"Aucune image trouvée dans {clean_dir}"
image_path = random.choice(paths)
print("Image choisie:", image_path)
print(test_inpaint_pnp(image_path=image_path))