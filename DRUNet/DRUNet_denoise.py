import os, time, math, random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from PIL import Image
import torchvision.transforms.functional as TF
from DRUNet import DRUNetSigmaMap

def psnr_torch(x, y, eps=1e-8):
    mse = torch.mean((x - y) ** 2).item()
    return 10.0 * math.log10(1.0 / (mse + eps))

@torch.no_grad()
def denoise_image_pil(model, img_pil, sigma, device):
    """
    sigma: noise level en 'pixel space' (0..50)
    """
    y = TF.to_tensor(img_pil)  # [0,1]
    sigma_map = torch.full((1, y.shape[1], y.shape[2]), sigma / 255.0)
    inp = torch.cat([y, sigma_map], dim=0).unsqueeze(0).to(device)
    out = model(inp).squeeze(0).clamp(0.0, 1.0).cpu()
    return TF.to_pil_image(out)

@torch.no_grad()
def test_mode_A_clean_to_noisy(clean_path, ckpt_path, out_dir="test_outputs_drunet", sigma=25.0, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DRUNetSigmaMap(in_nc=4, out_nc=3, nc=(64,128,256,512), nb=4).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    clean_pil = Image.open(clean_path).convert("RGB")
    clean = TF.to_tensor(clean_pil).unsqueeze(0)  # (1,3,H,W)

    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(clean.shape, generator=g) * (sigma / 255.0)
    noisy = clean + noise  # PAS de clamp ici non plus (cohérent avec train)

    sigma_map = torch.full((1, 1, clean.shape[2], clean.shape[3]), sigma / 255.0)
    inp = torch.cat([noisy, sigma_map], dim=1).to(device)
    den = model(inp).clamp(0.0, 1.0).cpu()

    noisy_clamped = noisy.clamp(0.0, 1.0)  # seulement pour sauvegarde/PSNR
    psnr_noisy = psnr_torch(noisy_clamped, clean)
    psnr_den = psnr_torch(den, clean)

    TF.to_pil_image(clean.squeeze(0)).save(os.path.join(out_dir, "clean.png"))
    TF.to_pil_image(noisy_clamped.squeeze(0)).save(os.path.join(out_dir, f"noisy_sigma{int(sigma)}.png"))
    TF.to_pil_image(den.squeeze(0)).save(os.path.join(out_dir, f"denoised_sigma{int(sigma)}.png"))

    print("Saved to:", out_dir)
    print(f"PSNR noisy   : {psnr_noisy:.2f} dB")
    print(f"PSNR denoised: {psnr_den:.2f} dB")

test_mode_A_clean_to_noisy(
        clean_path=r"./BSDS300/images/test/102061.jpg",
        ckpt_path=r"./weights_drunet_sigmap/drunet_sigmap_final.pth",
        out_dir="results_DRUNET_denoise",
        sigma=20,
        seed=0
    )