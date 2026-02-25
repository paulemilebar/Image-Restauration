import os, random
from IRCNN_denoise import psnr_torch, ssim_torch, list_images
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
from IRCNN_final import IRCNNModelManager
import numpy as np

def load_levin09_kernel(npy_path: str = "kernels/Levin09.npy", kernel_index: int = 0) -> np.ndarray:
    arr = np.load(npy_path, allow_pickle=True)
    k = np.asarray(arr[0, kernel_index], dtype=np.float32)
    s = float(k.sum())
    k /= s
    return k


def psf_to_otf(psf: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """
    psf: (kh,kw) real
    returns otf: (H,W) complex
    """
    H, W = out_hw
    kh, kw = psf.shape
    pad = torch.zeros((H, W), device=psf.device, dtype=psf.dtype)
    pad[:kh, :kw] = psf
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


@torch.no_grad()
def hqs_deblur(
    y: torch.Tensor,
    otf: torch.Tensor,
    manager: torch.nn.Module,
    sigma_img: float,
    lam: float = 0.23,
    n_iter: int = 8,
    sigma_max: float = 49.0,
):
    device = y.device
    B, C, H, W = y.shape

    sigma_img_n = sigma_img / 255.0
    sigma_min = max(sigma_img, 0.1)

    # log schedule sigma_d: sigma_max -> sigma_min
    sigmas_d = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), n_iter))

    Y = torch.fft.fft2(y, dim=(-2, -1))
    Hc = torch.conj(otf)
    H2 = (otf.real ** 2 + otf.imag ** 2)

    z = y.clone()

    for k in range(n_iter):
        sigma_d = float(sigmas_d[k])
        sigma_d_n = sigma_d / 255.0
        mu = lam / (sigma_d_n ** 2)
        alpha = mu * (sigma_img_n ** 2)

        # x-step
        Z = torch.fft.fft2(z, dim=(-2, -1))
        denom = (H2 + alpha).to(torch.complex64)
        numer = Hc[None, None, :, :] * Y + alpha * Z
        X = numer / denom
        x = torch.fft.ifft2(X, dim=(-2, -1)).real

        # z-step
        expert = manager.get_expert(sigma_d)
        z = expert.denoise(x).clamp(0, 1)

    return z

@torch.no_grad()
def test_deblurring_with_levin09(
    clean_path: str,
    ckpt_path: str,
    levin09_path: str = "kernels/Levin09.npy",
    kernel_index: int = 0,
    sigma_img: float = 2.55,
    n_iter: int = 8,
    lam: float = 0.23,
    out_dir: str = "test_outputs_deblur",
    seed: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model and weights
    model = IRCNNModelManager(ckpt_path, device=device)

    # load clean image
    clean_pil = Image.open(clean_path).convert("RGB")
    x = TF.to_tensor(clean_pil).unsqueeze(0).to(device)
    B, C, H, W = x.shape

    # load kernel Levin09 for blurring
    k_np = load_levin09_kernel(levin09_path, kernel_index)
    k = torch.from_numpy(k_np).to(device)

    # blurring + add Gaussian noise
    otf = psf_to_otf(k, (H, W))
    blurry = circ_conv_fft(x, otf)

    sigma_n = sigma_img / 255.0
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(blurry.shape, device=blurry.device, dtype=blurry.dtype, generator=g) * sigma_n

    y = (blurry + noise).clamp(0.0, 1.0)

    # deblur
    x_hat = hqs_deblur(
        y=y, otf=otf, manager=model,
        sigma_img=sigma_img, lam=lam, n_iter=n_iter, sigma_max=49.0
    )

    # metrics
    psnr_blur = psnr_torch(y.detach().cpu(), x.detach().cpu())
    psnr_rec  = psnr_torch(x_hat.detach().cpu(), x.detach().cpu())
    
    ssim_blur = ssim_torch(y.detach().cpu(), x.detach().cpu())
    ssim_rec  = ssim_torch(x_hat.detach().cpu(), x.detach().cpu())

    # save results
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
    print(f"n_iter      : {n_iter}, lambda={lam}")
    print(f"PSNR blurry/noisy: {psnr_blur:.2f} dB")
    print(f"PSNR restored    : {psnr_rec:.2f} dB")
    print(f"SSIM blurry/noisy: {ssim_blur:.2f}")
    print(f"SSIM restored    : {ssim_rec:.2f}")
    
#Plane
'''test_deblurring_with_levin09(
    clean_path="./BSDS300/images/test/37073.jpg",
    ckpt_path="./weights_ircnn",
    levin09_path="kernels/Levin09.npy",
    kernel_index=0,
    sigma_img=2.55,
    n_iter=8,
    lam=0.23,
    out_dir="./results_IRCNN/deblur_single/plane",
 )'''
 
#Castel
'''test_deblurring_with_levin09(
    clean_path="./BSDS300/images/test/102061.jpg",
    ckpt_path="./weights_ircnn",
    levin09_path="kernels/Levin09.npy",
    kernel_index=0,
    sigma_img=2.55,
    n_iter=8,
    lam=0.23,
    out_dir="./results_IRCNN/deblur_single/castel",
 )'''

 
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
    out_dir: str = "./results_IRCNN/deblur_benchmark",
):
    if save_outputs:
        os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = IRCNNModelManager(ckpt_path, device=device)

    # load clean image
    clean_pil = Image.open(clean_path).convert("RGB")
    x = TF.to_tensor(clean_pil).unsqueeze(0).to(device)
    _, _, H, W = x.shape

    # load kernel Levin09
    k_np = load_levin09_kernel(levin09_path, kernel_index)
    k = torch.from_numpy(k_np).to(device)

    # blur + add Gaussian noise
    otf = psf_to_otf(k, (H, W))
    blurry = circ_conv_fft(x, otf)

    sigma_n = sigma_img / 255.0
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(blurry.shape, device=blurry.device, dtype=blurry.dtype, generator=g) * sigma_n

    y = (blurry + noise).clamp(0.0, 1.0)

    # deblur
    x_hat = hqs_deblur(
        y=y, otf=otf, manager=model,
        sigma_img=sigma_img, lam=lam, n_iter=n_iter, sigma_max=49.0
    )

    # metrics
    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()
    xhat_cpu = x_hat.detach().cpu()

    psnr_blur = psnr_torch(y_cpu, x_cpu)
    psnr_rec  = psnr_torch(xhat_cpu, x_cpu)
    
    ssim_blur = ssim_torch(y_cpu, x_cpu)
    ssim_rec  = ssim_torch(xhat_cpu, x_cpu)

    # save
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
        "ssim_blurry": float(ssim_blur),
        "ssim_restored": float(ssim_rec),
        "ssim_gain": float(ssim_rec - ssim_blur),
    }


@torch.no_grad()
def benchmark_deblur_to_csv(
    test_dir: str,
    ckpt_path: str,
    levin09_path: str = "kernels/Levin09.npy",
    kernel_index: int = 0,
    sigma_img: float = 2.55,
    n_iter: int = 8,
    lam: float = 0.23,
    n_images: int = 10,
    seed: int = 0,
    out_dir: str = "./results_IRCNN/deblur_benchmark",
    save_examples: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    all_paths = list_images(test_dir)
    if len(all_paths) == 0:
        raise RuntimeError(f"No images found in {test_dir}")
    if len(all_paths) < n_images:
        raise RuntimeError(f"Not enough images: {len(all_paths)} < {n_images}")

    rng = random.Random(seed)
    chosen = rng.sample(all_paths, n_images)

    rows = []
    for i, p in enumerate(chosen):
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
        print(f"[{i+1}/{n_images}] {row['filename']} | PSNR blurry {row['psnr_blurry_db']:.2f} -> restored {row['psnr_restored_db']:.2f} dB | SSIM blurry {row['ssim_blurry']:.2f} -> restored {row['ssim_restored']:.2f}")
    
    df = pd.DataFrame(rows)

    metric_cols = [c for c in df.columns if c.startswith("psnr") or c.startswith("gain") or c.startswith("ssim")]
    mean_row = {"filename": "MEAN"}
    std_row  = {"filename": "STD"}
    for c in metric_cols:
        mean_row[c] = float(df[c].mean())
        std_row[c]  = float(df[c].std())
    df2 = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    csv_path = os.path.join(
        out_dir,
        f"deblur_benchmark_{n_images}imgs_k{kernel_index}_sig{sigma_img:.2f}_K{n_iter}_lam{lam}_seed{seed}.csv"
    )
    df2.to_csv(csv_path, index=False)

    return df2, csv_path

'''if __name__ == "__main__":
    benchmark_deblur_to_csv(
        test_dir="./BSDS300/images/images_benchmark/benchmark_10_images",
        ckpt_path="./weights_ircnn",
        levin09_path="kernels/Levin09.npy",
        kernel_index=0,
        sigma_img=5,
        n_iter=20,
        lam=0.23,
        n_images=10,
        seed=0,
        out_dir="./results_IRCNN/deblur_benchmark",
        save_examples=False,
    )'''
