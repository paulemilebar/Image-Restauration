import os, math, random
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
from DRUNet import DRUNetSigmaMap
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_fn


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


@torch.no_grad()
def drunet_infer(model: torch.nn.Module, inp: torch.Tensor, modulo: int = 8) -> torch.Tensor:
    b, c, h, w = inp.shape
    pad_h = (modulo - h % modulo) % modulo
    pad_w = (modulo - w % modulo) % modulo
    if pad_h or pad_w:
        inp2 = F.pad(inp, (0, pad_w, 0, pad_h), mode="reflect")
        out = model(inp2)
        return out[..., :h, :w]
    return model(inp)


def load_drunet(ckpt_path: str, device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, torch.device]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRUNetSigmaMap(in_nc=4, out_nc=3, nc=(64, 128, 256, 512), nb=4).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=True)
    return model, device

@torch.no_grad()
def denoise_noisy_tensor(model: torch.nn.Module, noisy01: torch.Tensor, sigma: float, device: torch.device,
                         modulo: int = 8) -> torch.Tensor:
    """
    Denoise a *tensor* already noisy.
    noisy01: (1,3,H,W) (not necessarily clamped)
    returns: (1,3,H,W) clamped to [0,1]
    """
    assert noisy01.ndim == 4 and noisy01.shape[1] == 3, "noisy01 must be (1,3,H,W)"
    _, _, H, W = noisy01.shape
    sigma01 = sigma / 255.0
    sigma_map = torch.full((1, 1, H, W), sigma01, device=noisy01.device, dtype=noisy01.dtype)
    inp = torch.cat([noisy01, sigma_map], dim=1).to(device)
    den = drunet_infer(model, inp, modulo=modulo).clamp(0.0, 1.0)
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
    ckpt_path: str,
    out_dir: str = "results_DRUNET_denoise_single",
    sigma: float = 20.0,
    seed: int = 0,
    modulo: int = 8,
    ircnn_model_dir: str = "./weights_ircnn",
    ircnn_lambda: float = 1.0,
) -> Dict[str, float]:
    """
    Load clean image, add AWGN, denoise, save clean/noisy/denoised, print PSNR and SSIM.
    """
    os.makedirs(out_dir, exist_ok=True)
    model, device = load_drunet(ckpt_path)

    clean_pil = Image.open(clean_path).convert("RGB")
    clean = TF.to_tensor(clean_pil).unsqueeze(0)  # (1,3,H,W)

    noisy = add_awgn(clean, sigma=sigma, seed=seed, device=device)  # NO CLAMP
    den = denoise_noisy_tensor(model, noisy, sigma=sigma, device=device, modulo=modulo).cpu()

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


@torch.no_grad()
def benchmark_drunet_random(
    test_dir: str,
    ckpt_path: str,
    out_dir: str = "results_DRUNET_denoise_benchmark",
    sigma: float = 20.0,
    n_images: int = 20,
    seed: int = 0,
    modulo: int = 8,
    save_examples: bool = False,
) -> pd.DataFrame:
    """
    Randomly sample n_images from test_dir, add AWGN, denoise, compute PSNR per image.
    Saves CSV. Optionally saves clean/noisy/den for each image.
    """
    os.makedirs(out_dir, exist_ok=True)
    model, device = load_drunet(ckpt_path)

    all_paths = list_images(test_dir)
    if len(all_paths) == 0:
        raise RuntimeError(f"Aucune image trouvée dans {test_dir}")
    if len(all_paths) < n_images:
        raise RuntimeError(f"Pas assez d'images: {len(all_paths)} < {n_images}")

    rng = random.Random(seed)
    chosen = rng.sample(all_paths, n_images)

    rows = []
    for i, path in enumerate(chosen):
        clean_pil = Image.open(path).convert("RGB")
        clean = TF.to_tensor(clean_pil).unsqueeze(0)  # (1,3,H,W)

        noisy = add_awgn(clean, sigma=sigma, seed=seed + i, device=device)  # NO CLAMP
        den = denoise_noisy_tensor(model, noisy, sigma=sigma, device=device, modulo=modulo).cpu()

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
    
    metric_cols = [c for c in df.columns if c.startswith("psnr") or c.startswith("gain") or c.startswith("ssim")]
    mean_row = {"filename": "MEAN"}
    std_row  = {"filename": "STD"}
    for c in metric_cols:
        mean_row[c] = float(df[c].mean())
        std_row[c]  = float(df[c].std())
    df2 = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)


    csv_path = os.path.join(out_dir, f"drunet_benchmark_{n_images}imgs_sigma{int(sigma)}_seed{seed}.csv")
    df2.to_csv(csv_path, index=False)

    print("\n=== Résultats DRUNet (benchmark random) ===")
    print(f"test_dir : {test_dir}")
    print(f"ckpt     : {ckpt_path}")
    print(f"sigma    : {sigma} (pixel)")
    print(f"seed     : {seed}")
    print(f"CSV saved: {csv_path}")


    show_cols = ["idx", "filename", "sigma", "psnr_noisy_db", "psnr_denoised_db", "gain_db", "ssim_noisy", "ssim_denoised", "gain_ssim"]
    
    
    print("\nTableau (par image):")
    print(df2[show_cols].to_string(index=False, justify="left", float_format=lambda x: f"{x:0.2f}"))

    return df2


if __name__ == "__main__":
    # 1) One image (qualitative + PSNR + saves)
    run_single_image_demo(clean_path="./BSDS300/images/test/102061.jpg", ckpt_path="./weights_drunet_sigmap/drunet_sigmap_final.pth", out_dir="results_DRUNET/results_DRUNET_denoise_single/castel", sigma=20.0, seed=0)

    # 2) Benchmark N images (table + CSV)
    #benchmark_drunet_random(test_dir="./BSDS300/images/images_benchmark/benchmark_10_images", ckpt_path="./weights_drunet_sigmap/drunet_sigmap_final.pth", out_dir="results_DRUNET/results_DRUNET_denoise_benchmark", sigma=25.0, n_images=10, seed=0, save_examples=False)

