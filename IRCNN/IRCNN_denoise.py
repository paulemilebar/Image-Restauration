import os, math, random
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
import torch.fft
from IRCNN_final import IRCNNModelManager
import matplotlib.pyplot as plt
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_fn
from typing import List, Tuple, Optional, Dict

# Utils
def psnr_torch(x01: torch.Tensor, y01: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((x01 - y01) ** 2).item()
    return 10.0 * math.log10(1.0 / (mse + eps))

def ssim_torch(x, y, data_range=1.0):
    return float(ssim_fn(x, y, data_range=data_range).item())

def list_images(folder: str, exts=(".png", ".jpg", ".jpeg")) -> List[str]:
    paths = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(exts):
                paths.append(os.path.join(root, fn))
    return sorted(paths)


def add_awgn(clean01: torch.Tensor, sigma: float, seed: int = 0, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    clean01: (1,3,H,W) in [0,1]
    returns noisy = clean + N(0, (sigma/255)^2)
    """
    if device is None:
        device = clean01.device
    sigma01 = sigma / 255.0
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(clean01.shape, generator=g, device=device, dtype=clean01.dtype) * sigma01
    return clean01.to(device) + noise


def solve_fidelity_denoise(y, z, mu):
     return (y + mu * z) / (1 + mu)


def run_single(
    clean_path:str,
    ckpt_path:str = "./weights_ircnn",
    out_dir:str = "./results_IRCNN/denoise_single",
    sigma: float = 20.0,
    seed: int = 0,
) -> Dict[str, float]:
    
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_pil = Image.open(clean_path).convert("RGB")
    clean = TF.to_tensor(img_pil).unsqueeze(0).to(device)
    
    manager = IRCNNModelManager(ckpt_path, device=device)
    expert = manager.get_expert(sigma)
    
    noisy = add_awgn(clean, sigma=sigma, seed=seed, device=device)
    den = expert.denoise(noisy).clamp(0, 1)
    noisy_clamped = noisy.clamp(0.0, 1.0)
    
    psnr_noisy = psnr_torch(noisy_clamped, clean)
    psnr_den = psnr_torch(den, clean)
    
    ssim_noisy = ssim_torch(noisy_clamped, clean)
    ssim_den = ssim_torch(den, clean)

    TF.to_pil_image(clean.squeeze(0)).save(os.path.join(out_dir, "clean.png"))
    TF.to_pil_image(noisy_clamped.squeeze(0)).save(os.path.join(out_dir, f"noisy_sigma{int(sigma)}.png"))
    TF.to_pil_image(den.squeeze(0)).save(os.path.join(out_dir, f"denoised_sigma{int(sigma)}.png"))

    print("Saved to:", out_dir)
    print(f"PSNR noisy   : {psnr_noisy:.2f} dB")
    print(f"PSNR denoised: {psnr_den:.2f} dB")
    print(f"SSIM noisy   : {ssim_noisy:.2f}")
    print(f"SSIM denoised: {ssim_den:.2f}")

    return {"psnr_noisy": psnr_noisy, "psnr_denoised": psnr_den, "ssim_noisy": ssim_noisy, "ssim_denoised": ssim_den}

# Plane
'''image_path =  "./BSDS300/images/test/37073.jpg"
run_single(
    clean_path=image_path,
    ckpt_path = "./weights_ircnn",
    out_dir = "./results_IRCNN/denoise_single/plane",
    sigma=20.0
)'''


@torch.no_grad()
def benchmark_ircnn(
    test_dir: str,
    ckpt_path: str,
    out_dir: str,
    sigma: float = 20.0,
    n_images: int = 20,
    seed: int = 0,
    save_examples: bool = False,
) -> pd.DataFrame:
    """
    Randomly sample n_images from test_dir, add AWGN, denoise, compute PSNR per image.
    Saves CSV. Optionally saves clean/noisy/den for each image.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IRCNNModelManager(ckpt_path, device=device)

    rows = []
    for i, path in enumerate(test_dir):
        clean_pil = Image.open(path).convert("RGB")
        clean = TF.to_tensor(clean_pil).unsqueeze(0)  # (1,3,H,W)

        expert = model.get_expert(sigma)
    
        noisy = add_awgn(clean, sigma=sigma, seed=seed, device=device)
        den = expert.denoise(noisy).clamp(0, 1)
        noisy_clamped = noisy.clamp(0.0, 1.0)

        psnr_noisy = psnr_torch(noisy_clamped, clean)
        psnr_den = psnr_torch(den, clean)
        
        ssim_noisy = ssim_torch(noisy_clamped, clean)
        ssim_den = ssim_torch(den, clean)

        rows.append({
            "idx": i,
            "filename": os.path.basename(path),
            "path": path,
            "sigma": sigma,
            "psnr_noisy_db": psnr_noisy,
            "psnr_denoised_db": psnr_den,
            "gain_db": psnr_den - psnr_noisy,
            "ssim_noisy": ssim_noisy,
            "ssim_denoised": ssim_den,
            "gain_ssim": ssim_den - ssim_noisy,
        })

        if save_examples:
            base = os.path.splitext(os.path.basename(path))[0]
            TF.to_pil_image(clean.squeeze(0)).save(os.path.join(out_dir, f"{base}_clean.png"))
            TF.to_pil_image(noisy_clamped.squeeze(0)).save(os.path.join(out_dir, f"{base}_noisy_sigma{int(sigma)}.png"))
            TF.to_pil_image(den.squeeze(0)).save(os.path.join(out_dir, f"{base}_den_sigma{int(sigma)}.png"))

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c.startswith("psnr") or c.startswith("gain") or c.startswith("ssim")]
    mean_row = {"filename": "MEAN"}
    std_row  = {"filename": "STD"}
    for c in metric_cols:
        mean_row[c] = float(df[c].mean())
        std_row[c]  = float(df[c].std())
    df2 = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    mean_noisy_psnr = df["psnr_noisy_db"].mean()
    mean_den_psnr = df["psnr_denoised_db"].mean()
    mean_gain_psnr = df["gain_db"].mean()
    
    mean_noisy_ssim = df["ssim_noisy"].mean()
    mean_den_ssim = df["ssim_denoised"].mean()
    mean_gain_ssim = df["gain_ssim"].mean()

    csv_path = os.path.join(out_dir, f"ircnn_benchmark_{n_images}imgs_sigma{int(sigma)}_seed{seed}.csv")
    df2.to_csv(csv_path, index=False)

    print("\n=== Résultats IRCNN (benchmark random) ===")
    print(f"test_dir : {test_dir}")
    print(f"ckpt     : {ckpt_path}")
    print(f"sigma    : {sigma} (pixel)")
    print(f"seed     : {seed}")
    print(f"CSV saved: {csv_path}")

    print("\nMoyennes:")
    print(f"  PSNR noisy    : {mean_noisy_psnr:.2f} dB")
    print(f"  PSNR denoised : {mean_den_psnr:.2f} dB")
    print(f"  Gain PSNR         : {mean_gain_psnr:.2f} dB")
    
    print(f"  SSIM noisy    : {mean_noisy_ssim:.2f}")
    print(f"  SSIM denoised : {mean_den_ssim:.2f}")
    print(f"  Gain SSIM         : {mean_gain_ssim:.2f}")

    show_cols = ["idx", "filename", "sigma", "psnr_noisy_db", "psnr_denoised_db", "gain_db", "ssim_noisy", "ssim_denoised", "gain_ssim"]
    print("\nTableau (par image):")
    print(df2[show_cols].to_string(index=False, justify="left", float_format=lambda x: f"{x:0.2f}"))

    return df2


'''if __name__ == "__main__":
    benchmark_ircnn(test_dir="./BSDS300/images/images_benchmark/benchmark_10_images", ckpt_path="./weights_ircnn", out_dir="./results_IRCNN/denoise_benchmark", sigma=25.0, n_images=10, seed=0, save_examples=False)'''

