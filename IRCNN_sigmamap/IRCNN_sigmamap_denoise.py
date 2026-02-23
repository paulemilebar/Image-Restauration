import os, math, random
import pandas as pd
import torch
from IRCNN_sigmamap import IRCNNSigmaMap
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_fn
from typing import Optional, Tuple, Dict, List
from pool_images_test.generate_pool_images import load_paths_from_file

def psnr_torch(x, y, eps=1e-8):
    mse = torch.mean((x - y) ** 2).item()
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

def load_ircnn(ckpt_path: str, device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, torch.device]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IRCNNSigmaMap().to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    return model, device

def ircnn_infer(model: torch.nn.Module, inp: torch.Tensor) -> torch.Tensor:
    return model(inp)

@torch.no_grad()
def denoise_noisy_tensor(model: torch.nn.Module, noisy01: torch.Tensor, sigma: float, device: torch.device) -> torch.Tensor:
    """
    Denoise a tensor already noisy.
    noisy01: (1,3,H,W)
    returns: (1,3,H,W) clamped to [0,1]
    """
    assert noisy01.ndim == 4 and noisy01.shape[1] == 3, "noisy01 must be (1,3,H,W)"
    _, _, H, W = noisy01.shape
    sigma01 = sigma / 255.0
    sigma_map = torch.full((1, 1, H, W), sigma01, device=noisy01.device, dtype=noisy01.dtype)
    inp = torch.cat([noisy01, sigma_map], dim=1).to(device)
    den = ircnn_infer(model, inp).clamp(0.0, 1.0)
    return den
    
    
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


@torch.no_grad()

def run_single_image_demo(
    clean_path: str,
    ckpt_path: str = r"./weights_ircnn_sigmap/ircnn_sigmap_final.pth",
    out_dir: str = r"./results_IRCNN_sigmamap/denoise_single",
    sigma: float = 50,
    seed: int = 0
) -> Dict[str, float]:
    os.makedirs(out_dir, exist_ok=True)
    model, device = load_ircnn(ckpt_path)

    clean_pil = Image.open(clean_path).convert("RGB")
    clean = TF.to_tensor(clean_pil).unsqueeze(0)

    noisy = add_awgn(clean, sigma=sigma, seed=seed, device=device)
    den = denoise_noisy_tensor(model, noisy, sigma=sigma, device=device).cpu()

    noisy_clamped = noisy.cpu().clamp(0.0, 1.0)
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



'''run_single_image_demo(
    clean_path=r"./BSDS300/images/test/37073.jpg",
    sigma=50
)'''


@torch.no_grad()
def benchmark_ircnn_sigmap(
    test_dir: str,
    ckpt_path: str,
    out_dir: str = r"./results_IRCNN_sigmamap/denoise_benchmark",
    sigma: float = 20.0,
    n_images: int = 10,
    seed: int = 0,
    save_examples: bool = False
) -> pd.DataFrame:
    """
    Randomly sample n_images from test_dir, add AWGN, denoise, compute PSNR per image.
    Saves CSV. Optionally saves clean/noisy/den for each image.
    """
    os.makedirs(out_dir, exist_ok=True)
    model, device = load_ircnn(ckpt_path)

    '''all_paths = list_images(test_dir)
    if len(all_paths) == 0:
        raise RuntimeError(f"Aucune image trouvée dans {test_dir}")
    if len(all_paths) < n_images:
        raise RuntimeError(f"Pas assez d'images: {len(all_paths)} < {n_images}")

    rng = random.Random(seed)
    chosen = rng.sample(all_paths, n_images)'''
    
    chosen = load_paths_from_file("./pool_images_test/benchmark_10_images.txt")

    rows = []
    for i, path in enumerate(chosen):
        clean_pil = Image.open(path).convert("RGB")
        clean = TF.to_tensor(clean_pil).unsqueeze(0)  # (1,3,H,W)

        noisy = add_awgn(clean, sigma=sigma, seed=seed + i, device=device)  # NO CLAMP
        den = denoise_noisy_tensor(model, noisy, sigma=sigma, device=device).cpu()

        noisy_clamped = noisy.cpu().clamp(0.0, 1.0)
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

    mean_noisy_psnr = df["psnr_noisy_db"].mean()
    mean_den_psnr = df["psnr_denoised_db"].mean()
    mean_gain_psnr = df["gain_db"].mean()
    
    mean_noisy_ssim = df["ssim_noisy"].mean()
    mean_den_ssim = df["ssim_denoised"].mean()
    mean_gain_ssim = df["gain_ssim"].mean()

    csv_path = os.path.join(out_dir, f"ircnn_benchmark_{n_images}imgs_sigma{int(sigma)}_seed{seed}.csv")
    df.to_csv(csv_path, index=False)

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
    print(df[show_cols].to_string(index=False, justify="left", float_format=lambda x: f"{x:0.2f}"))

    return df


if __name__ == "__main__":
    benchmark_ircnn_sigmap(test_dir="./BSDS300/images/test", ckpt_path="./weights_ircnn_sigmap/ircnn_sigmap_final.pth", out_dir="results_IRCNN_sigmamap/denoise_benchmark_v2", sigma=20.0, n_images=10, seed=0, save_examples=False)

