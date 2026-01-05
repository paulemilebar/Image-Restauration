import os, random
from DRUNet_denoise import psnr_torch

import torch
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
from DRUNet import DRUNetSigmaMap

import numpy as np

# Robust Levin09 loader (handles many .npy formats)
def load_levin09_kernel(npy_path: str, kernel_index: int) -> np.ndarray:
    """
    Robust loader for Levin09.npy where kernels may be stored as:
      - list/tuple of 2D arrays (variable sizes)
      - object ndarray (e.g. shape (1,8), (8,), etc.) containing 2D arrays
      - stacked float ndarray (K,H,W)
      - dict wrappers
    Returns a float32 2D kernel normalized to sum=1.
    """
    obj = np.load(npy_path, allow_pickle=True)

    # unwrap common wrappers
    while True:
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
            obj = obj.item()
            continue
        if isinstance(obj, dict):
            # try typical keys, else take first value
            for key in ["kernels", "kernel", "psf", "k", "data"]:
                if key in obj:
                    obj = obj[key]
                    break
            else:
                obj = next(iter(obj.values()))
            continue
        break

    # Case A: stacked numeric array (K,H,W)
    if isinstance(obj, np.ndarray) and obj.dtype != object and obj.ndim == 3:
        K = obj.shape[0]
        if not (0 <= kernel_index < K):
            raise IndexError(f"kernel_index={kernel_index} out of range, K={K}")
        k = obj[kernel_index].astype(np.float32)

    # Case B: single numeric kernel (H,W)
    elif isinstance(obj, np.ndarray) and obj.dtype != object and obj.ndim == 2:
        if kernel_index != 0:
            raise IndexError(f"Only 1 kernel in file; use kernel_index=0 (got {kernel_index}).")
        k = obj.astype(np.float32)

    # Case C: list/tuple of kernels
    elif isinstance(obj, (list, tuple)):
        K = len(obj)
        if not (0 <= kernel_index < K):
            raise IndexError(f"kernel_index={kernel_index} out of range, K={K}")
        k = np.asarray(obj[kernel_index], dtype=np.float32)

    # Case D: object ndarray container (common for Levin09: shape (1,8) or (8,))
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        flat = list(obj.ravel())
        # sometimes flat contains a single list/tuple of kernels
        if len(flat) == 1 and isinstance(flat[0], (list, tuple, np.ndarray)):
            obj2 = flat[0]
            if isinstance(obj2, np.ndarray) and obj2.dtype == object:
                flat = list(obj2.ravel())
            elif isinstance(obj2, (list, tuple)):
                flat = list(obj2)
            elif isinstance(obj2, np.ndarray) and obj2.dtype != object and obj2.ndim == 3:
                K = obj2.shape[0]
                if not (0 <= kernel_index < K):
                    raise IndexError(f"kernel_index={kernel_index} out of range, K={K}")
                k = obj2[kernel_index].astype(np.float32)
                s = float(k.sum())
                if s != 0:
                    k /= s
                return k

        K = len(flat)
        if not (0 <= kernel_index < K):
            raise IndexError(f"kernel_index={kernel_index} out of range, K={K}")
        k = np.asarray(flat[kernel_index], dtype=np.float32)

    else:
        raise ValueError(f"Unsupported Levin09.npy content type: {type(obj)}, "
                         f"ndim={getattr(obj,'ndim',None)}, dtype={getattr(obj,'dtype',None)}")

    # Normalize
    s = float(k.sum())
    if s != 0:
        k /= s
    return k

# FFT circular convolution utils (uniform blur, circular BC)
def psf_to_otf(psf: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """
    psf: (kh,kw) real
    returns otf: (H,W) complex
    """
    H, W = out_hw
    kh, kw = psf.shape
    pad = torch.zeros((H, W), device=psf.device, dtype=psf.dtype)
    pad[:kh, :kw] = psf
    # shift center to (0,0)
    pad = torch.roll(pad, shifts=(-(kh // 2), -(kw // 2)), dims=(0, 1))
    return torch.fft.fft2(pad)

def circ_conv_fft(x: torch.Tensor, otf: torch.Tensor) -> torch.Tensor:
    """
    x: (B,C,H,W) real
    otf: (H,W) complex
    """
    X = torch.fft.fft2(x, dim=(-2, -1))
    Y = X * otf[None, None, :, :]
    return torch.fft.ifft2(Y, dim=(-2, -1)).real

# DPIR-like HQS deblurring: x-step FFT closed-form, z-step DRUNet
@torch.no_grad()
def dpir_hqs_deblur(
    y: torch.Tensor,                 # (B,3,H,W) in [0,1]
    otf: torch.Tensor,               # (H,W) complex
    denoiser: torch.nn.Module,       # DRUNetSigmaMap
    sigma_img: float,                # noise std in pixel space (0..255), e.g. 2.55
    lam: float = 0.23,
    n_iter: int = 8,
    sigma_max: float = 49.0,
):
    device = y.device
    B, C, H, W = y.shape

    sigma_img_n = sigma_img / 255.0
    sigma_min = max(sigma_img, 0.1)

    # log schedule sigma_d: sigma_max -> sigma_min
    sigmas_d = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), n_iter)).astype(np.float32)

    Y = torch.fft.fft2(y, dim=(-2, -1))
    Hc = torch.conj(otf)
    H2 = (otf.real ** 2 + otf.imag ** 2)  # |H|^2 reel

    z = y.clone()

    for k in range(n_iter):
        sigma_d = float(sigmas_d[k])
        sigma_d_n = sigma_d / 255.0

        # mu = lambda / sigma_d^2
        mu = lam / (sigma_d_n ** 2)

        # alpha = mu * sigma_img^2 (matches paper scaling in x-step)
        alpha = mu * (sigma_img_n ** 2)

        Z = torch.fft.fft2(z, dim=(-2, -1))
        denom = (H2 + alpha).to(torch.complex64)
        numer = Hc[None, None, :, :] * Y + alpha * Z
        X = numer / denom
        x = torch.fft.ifft2(X, dim=(-2, -1)).real

        # z-step = denoise(x, sigma_d)
        sigma_map = torch.full((B, 1, H, W), sigma_d_n, device=device)
        inp = torch.cat([x, sigma_map], dim=1)
        z = denoiser(inp).clamp(0.0, 1.0)

    return z


# Main test function: blur + noise, then DPIR deblur
@torch.no_grad()
def test_deblurring_dpir_with_levin09(
    clean_path: str,
    ckpt_path: str,
    levin09_path: str = "kernels/Levin09.npy",
    kernel_index: int = 0,
    sigma_img: float = 2.55,
    n_iter: int = 8,
    lam: float = 0.23,
    out_dir: str = "test_outputs_dpir_deblur",
    seed: int = 0,
):
    assert os.path.isfile(levin09_path), f"File not found: {levin09_path}"
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model and weights
    model = DRUNetSigmaMap(in_nc=4, out_nc=3, nc=(64,128,256,512), nb=4).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=True)

    # load clean image
    clean_pil = Image.open(clean_path).convert("RGB")
    x = TF.to_tensor(clean_pil).unsqueeze(0).to(device)  # (1,3,H,W)
    B, C, H, W = x.shape

    # load kernel Levin09 for blurring
    k_np = load_levin09_kernel(levin09_path, kernel_index)
    k = torch.from_numpy(k_np).to(device)

    # blurring (circular) + add Gaussian noise
    otf = psf_to_otf(k, (H, W))
    blurry = circ_conv_fft(x, otf)

    sigma_n = sigma_img / 255.0
    try:
        g = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(blurry.shape, device=blurry.device, dtype=blurry.dtype, generator=g) * sigma_n
    except TypeError:
        # fallback si generator n'est pas supporté
        torch.manual_seed(seed)
        noise = torch.randn(blurry.shape, device=blurry.device, dtype=blurry.dtype) * sigma_n

    y = (blurry + noise).clamp(0.0, 1.0)

    # ----- deblur (DPIR/HQS) -----
    x_hat = dpir_hqs_deblur(
        y=y, otf=otf, denoiser=model,
        sigma_img=sigma_img, lam=lam, n_iter=n_iter, sigma_max=49.0
    )

    # ----- metrics -----
    psnr_blur = psnr_torch(y.detach().cpu(), x.detach().cpu())
    psnr_rec  = psnr_torch(x_hat.detach().cpu(), x.detach().cpu())

    # ----- save results -----
    TF.to_pil_image(x.squeeze(0).cpu()).save(os.path.join(out_dir, "clean.png"))
    TF.to_pil_image(y.squeeze(0).cpu()).save(os.path.join(out_dir, f"blurry_k{kernel_index}_sigma{sigma_img:.2f}.png"))
    TF.to_pil_image(x_hat.squeeze(0).cpu()).save(os.path.join(out_dir, f"restored_k{kernel_index}_sigma{sigma_img:.2f}.png"))

    # kernel visualization
    k_vis = (k / (k.max() + 1e-12)).detach().cpu().numpy()
    k_vis = (k_vis * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(k_vis).save(os.path.join(out_dir, f"kernel_k{kernel_index}.png"))

    print("Saved to:", out_dir)
    print(f"Kernel index     : {kernel_index}")
    print(f"Noise sigma_img  : {sigma_img:.2f} (pixel space)")
    print(f"DPIR n_iter      : {n_iter}, lambda={lam}")
    print(f"PSNR blurry/noisy: {psnr_blur:.2f} dB")
    print(f"PSNR restored    : {psnr_rec:.2f} dB")
    

@torch.no_grad()
def run_deblur_one_return_metrics(
    clean_path: str,
    ckpt_path: str,
    levin09_path: str = "kernels/Levin09.npy",
    kernel_index: int = 0,
    sigma_img: float = 2.55,
    n_iter: int = 8,
    lam: float = 0.23,
    seed: int = 0,
    save_outputs: bool = False,
    out_dir: str = "results_DRUNET_deblur_batch",
):
    assert os.path.isfile(levin09_path), f"File not found: {levin09_path}"
    if save_outputs:
        os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- load model -----
    model = DRUNetSigmaMap(in_nc=4, out_nc=3, nc=(64,128,256,512), nb=4).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=True)

    # ----- load clean image -----
    clean_pil = Image.open(clean_path).convert("RGB")
    x = TF.to_tensor(clean_pil).unsqueeze(0).to(device)  # (1,3,H,W)
    _, _, H, W = x.shape

    # ----- load kernel Levin09 -----
    k_np = load_levin09_kernel(levin09_path, kernel_index)
    k = torch.from_numpy(k_np).to(device)

    # ----- blur (circular) + add Gaussian noise -----
    otf = psf_to_otf(k, (H, W))
    blurry = circ_conv_fft(x, otf)

    sigma_n = sigma_img / 255.0
    try:
        g = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(blurry.shape, device=blurry.device, dtype=blurry.dtype, generator=g) * sigma_n
    except TypeError:
        torch.manual_seed(seed)
        noise = torch.randn(blurry.shape, device=blurry.device, dtype=blurry.dtype) * sigma_n

    y = (blurry + noise).clamp(0.0, 1.0)

    # ----- deblur (DPIR/HQS) -----
    x_hat = dpir_hqs_deblur(
        y=y, otf=otf, denoiser=model,
        sigma_img=sigma_img, lam=lam, n_iter=n_iter, sigma_max=49.0
    )

    # ----- metrics -----
    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()
    xhat_cpu = x_hat.detach().cpu()

    psnr_blur = psnr_torch(y_cpu, x_cpu)
    psnr_rec  = psnr_torch(xhat_cpu, x_cpu)

    # ----- optional save -----
    if save_outputs:
        base = os.path.splitext(os.path.basename(clean_path))[0]
        TF.to_pil_image(x_cpu.squeeze(0)).save(os.path.join(out_dir, f"{base}_clean.png"))
        TF.to_pil_image(y_cpu.squeeze(0)).save(os.path.join(out_dir, f"{base}_blurry_k{kernel_index}_sig{sigma_img:.2f}.png"))
        TF.to_pil_image(xhat_cpu.squeeze(0)).save(os.path.join(out_dir, f"{base}_restored_k{kernel_index}_sig{sigma_img:.2f}.png"))

    return {
        "filename": os.path.basename(clean_path),
        "path": clean_path,
        "kernel_index": kernel_index,
        "sigma_img": float(sigma_img),
        "n_iter": int(n_iter),
        "lambda": float(lam),
        "seed": int(seed),
        "psnr_blurry_db": float(psnr_blur),
        "psnr_restored_db": float(psnr_rec),
        "gain_db": float(psnr_rec - psnr_blur),
    }


def list_images(folder, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
    paths = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(exts):
                paths.append(os.path.join(root, fn))
    return sorted(paths)


@torch.no_grad()
def benchmark_dpir_deblur_to_csv(
    test_dir: str,
    ckpt_path: str,
    levin09_path: str = "kernels/Levin09.npy",
    kernel_index: int = 0,
    sigma_img: float = 2.55,
    n_iter: int = 8,
    lam: float = 0.23,
    n_images: int = 10,
    seed: int = 0,
    out_dir: str = "results_DRUNET_deblur_benchmark",
    save_examples: bool = False,
):
    """
    Sélectionne n_images au hasard dans test_dir, lance blur+noise+DPIR deblur,
    et sauvegarde un CSV + affiche les moyennes.
    """
    os.makedirs(out_dir, exist_ok=True)

    all_paths = list_images(test_dir)
    if len(all_paths) == 0:
        raise RuntimeError(f"Aucune image trouvée dans {test_dir}")
    if len(all_paths) < n_images:
        raise RuntimeError(f"Pas assez d'images: {len(all_paths)} < {n_images}")

    rng = random.Random(seed)
    chosen = rng.sample(all_paths, n_images)

    rows = []
    for i, p in enumerate(chosen):
        # seed différent par image pour le bruit
        row = run_deblur_one_return_metrics(
            clean_path=p,
            ckpt_path=ckpt_path,
            levin09_path=levin09_path,
            kernel_index=kernel_index,
            sigma_img=sigma_img,
            n_iter=n_iter,
            lam=lam,
            seed=seed + i,
            save_outputs=save_examples,
            out_dir=out_dir,
        )
        rows.append(row)
        print(f"[{i+1}/{n_images}] {row['filename']} | PSNR blurry {row['psnr_blurry_db']:.2f} -> restored {row['psnr_restored_db']:.2f} dB")

    df = pd.DataFrame(rows)

    mean_blur = df["psnr_blurry_db"].mean()
    mean_rest = df["psnr_restored_db"].mean()
    mean_gain = df["gain_db"].mean()

    csv_path = os.path.join(
        out_dir,
        f"dpir_deblur_benchmark_{n_images}imgs_k{kernel_index}_sig{sigma_img:.2f}_K{n_iter}_lam{lam}_seed{seed}.csv"
    )
    df.to_csv(csv_path, index=False)

    print("\n=== Moyennes (DPIR deblur) ===")
    print(f"PSNR blurry   : {mean_blur:.2f} dB")
    print(f"PSNR restored : {mean_rest:.2f} dB")
    print(f"Gain          : {mean_gain:.2f} dB")
    print("CSV saved:", csv_path)

    return df, csv_path

if __name__ == "__main__":
    test_deblurring_dpir_with_levin09(
         clean_path="./BSDS300/images/test/37073.jpg",
         ckpt_path="./weights_drunet_sigmap/drunet_sigmap_final.pth",
         levin09_path="kernels/Levin09.npy",
         kernel_index=0,
         sigma_img=2.55,
         n_iter=8,
         lam=0.23,
         out_dir="results_DRUNET/results_DRUNET_deblur",
 )

 #   benchmark_dpir_deblur_to_csv(
  #      test_dir="./BSDS300/images/test",
  #      ckpt_path="./weights_drunet_sigmap/drunet_sigmap_final.pth",
  #      levin09_path="kernels/Levin09.npy",
  #      kernel_index=0,
  #      sigma_img=2.55,
  #      n_iter=8,
  #      lam=0.23,
  #      n_images=10,
  #      seed=0,
  #      out_dir="results_DRUNET/results_DRUNET_deblur_benchmark",
  #      save_examples=False,
  #  )
