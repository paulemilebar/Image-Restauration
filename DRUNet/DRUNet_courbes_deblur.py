import os, time, math, random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from DRUNet import DRUNetSigmaMap
from DRUNet_deblur import load_levin09_kernel, psf_to_otf, circ_conv_fft, psnr_torch
import numpy as np
import os, math
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

def _rmse(a: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(a**2)).item())

@torch.no_grad()
def dpir_hqs_deblur_with_trace(
    y: torch.Tensor,                 # (B,3,H,W) in [0,1]
    otf: torch.Tensor,               # (H,W) complex
    denoiser: torch.nn.Module,       # DRUNetSigmaMap
    sigma_img: float,                # pixel space
    lam: float = 0.23,
    n_iter: int = 8,
    sigma_max: float = 49.0,
    x_gt: torch.Tensor | None = None,
    save_dir: str | None = None,
    save_every: int = 1,
):
    device = y.device
    B, C, H, W = y.shape

    sigma_img_n = sigma_img / 255.0
    sigma_min = max(sigma_img, 0.1)

    # log schedule sigma_d: sigma_max -> sigma_min
    sigmas_d = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), n_iter)).astype(np.float32)

    Y = torch.fft.fft2(y, dim=(-2, -1))
    Hc = torch.conj(otf)
    H2 = (otf.real**2 + otf.imag**2)  # |H|^2

    z = y.clone()
    x_prev, z_prev = None, None

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    logs = {
        "k": [],
        "sigma_d": [],
        "alpha": [],
        "data_rmse": [],   # rmse(y - Hx)
        "xz_rmse": [],     # rmse(x - z)
        "dx_rel": [],      # rmse(x-x_prev)/rmse(x_prev)
        "dz_rel": [],      # rmse(z-z_prev)/rmse(z_prev)
        "psnr_z": [],      # optional
    }

    for k in range(n_iter):
        sigma_d = float(sigmas_d[k])
        sigma_d_n = sigma_d / 255.0

        mu = lam / (sigma_d_n**2)
        alpha = mu * (sigma_img_n**2)

        # x-step
        Z = torch.fft.fft2(z, dim=(-2, -1))
        denom = (H2 + alpha).to(torch.complex64)
        numer = Hc[None, None, :, :] * Y + alpha * Z
        X = numer / denom
        x = torch.fft.ifft2(X, dim=(-2, -1)).real

        # z-step (denoise)
        sigma_map = torch.full((B, 1, H, W), sigma_d_n, device=device)
        inp = torch.cat([x, sigma_map], dim=1)
        z = denoiser(inp).clamp(0.0, 1.0)

        # metrics
        Hx = circ_conv_fft(x, otf)
        data_rmse = _rmse(y - Hx)
        xz_rmse = _rmse(x - z)

        if x_prev is None:
            dx_rel = float("nan")
            dz_rel = float("nan")
        else:
            dx_rel = _rmse(x - x_prev) / (_rmse(x_prev) + 1e-12)
            dz_rel = _rmse(z - z_prev) / (_rmse(z_prev) + 1e-12)

        psnr = psnr_torch(z.detach().cpu(), x_gt.detach().cpu()) if x_gt is not None else float("nan")

        logs["k"].append(k)
        logs["sigma_d"].append(sigma_d)
        logs["alpha"].append(float(alpha))
        logs["data_rmse"].append(data_rmse)
        logs["xz_rmse"].append(xz_rmse)
        logs["dx_rel"].append(dx_rel)
        logs["dz_rel"].append(dz_rel)
        logs["psnr_z"].append(psnr)

        if save_dir is not None and (k % save_every == 0 or k == n_iter - 1):
            TF.to_pil_image(z.squeeze(0).detach().cpu()).save(os.path.join(save_dir, f"z_iter{k:02d}.png"))

        x_prev, z_prev = x, z

    return z, logs


def print_logs_table(logs):
    header = f"{'k':>2} | {'sigma_d':>7} | {'data_rmse':>10} | {'xz_rmse':>10} | {'dx_rel':>9} | {'dz_rel':>9} | {'PSNR(z)':>7}"
    print(header)
    print("-" * len(header))
    for i in range(len(logs["k"])):
        k = logs["k"][i]
        print(f"{k:2d} | {logs['sigma_d'][i]:7.2f} | {logs['data_rmse'][i]:10.3e} | {logs['xz_rmse'][i]:10.3e} | "
              f"{logs['dx_rel'][i]:9.3e} | {logs['dz_rel'][i]:9.3e} | {logs['psnr_z'][i]:7.2f}")


# -------------------------
# TEST: same as your test_deblurring_dpir_with_levin09 but with convergence trace
# -------------------------
@torch.no_grad()
def test_deblurring_dpir_with_levin09_convergence(
    clean_path: str,
    ckpt_path: str,
    levin09_path: str = "kernels/Levin09.npy",
    kernel_index: int = 0,
    sigma_img: float = 2.55,
    n_iter: int = 8,
    lam: float = 0.23,
    out_dir: str = "test_outputs_dpir_deblur_conv",
    seed: int = 0,
    save_iters: bool = True,
):
    assert os.path.isfile(levin09_path), f"File not found: {levin09_path}"
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = DRUNetSigmaMap(in_nc=4, out_nc=3, nc=(64,128,256,512), nb=4).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=True)

    # load clean image
    clean_pil = Image.open(clean_path).convert("RGB")
    x = TF.to_tensor(clean_pil).unsqueeze(0).to(device)  # GT
    B, C, H, W = x.shape

    # kernel + otf
    k_np = load_levin09_kernel(levin09_path, kernel_index)
    k = torch.from_numpy(k_np).to(device)
    otf = psf_to_otf(k, (H, W))

    # generate y = Hx + n
    blurry = circ_conv_fft(x, otf)

    sigma_n = sigma_img / 255.0
    try:
        g = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(blurry.shape, device=blurry.device, dtype=blurry.dtype, generator=g) * sigma_n
    except TypeError:
        torch.manual_seed(seed)
        noise = torch.randn(blurry.shape, device=blurry.device, dtype=blurry.dtype) * sigma_n

    y = (blurry + noise).clamp(0.0, 1.0)

    # run DPIR + trace
    it_dir = os.path.join(out_dir, "iters") if save_iters else None
    x_hat, logs = dpir_hqs_deblur_with_trace(
        y=y, otf=otf, denoiser=model,
        sigma_img=sigma_img, lam=lam, n_iter=n_iter, sigma_max=49.0,
        x_gt=x,
        save_dir=it_dir,
        save_every=1
    )

    # metrics
    psnr_blur = psnr_torch(y.detach().cpu(), x.detach().cpu())
    psnr_rec  = psnr_torch(x_hat.detach().cpu(), x.detach().cpu())

    # save outputs
    TF.to_pil_image(x.squeeze(0).cpu()).save(os.path.join(out_dir, "clean.png"))
    TF.to_pil_image(y.squeeze(0).cpu()).save(os.path.join(out_dir, f"blurry_k{kernel_index}_sigma{sigma_img:.2f}.png"))
    TF.to_pil_image(x_hat.squeeze(0).cpu()).save(os.path.join(out_dir, f"restored_k{kernel_index}_sigma{sigma_img:.2f}.png"))

    # save kernel vis
    k_vis = (k / (k.max() + 1e-12)).detach().cpu().numpy()
    k_vis = (k_vis * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(k_vis).save(os.path.join(out_dir, f"kernel_k{kernel_index}.png"))

    # print convergence table
    print_logs_table(logs)

    print("\nFinal:")
    print(f"PSNR blurry/noisy: {psnr_blur:.2f} dB")
    print(f"PSNR restored    : {psnr_rec:.2f} dB")
    if save_iters:
        print(f"Intermediate z_k images saved in: {it_dir}")

    return logs

logs = test_deblurring_dpir_with_levin09_convergence(
    clean_path="./BSDS300/images/test/102061.jpg",
    ckpt_path="./weights_drunet_sigmap/drunet_sigmap_final.pth",
    levin09_path="kernels/Levin09.npy",
    kernel_index=0,
    sigma_img=2.55,
    n_iter=8,
    lam=0.23,
    out_dir="results_DRUNET_deblur_courbes",
    seed=0,
    save_iters=True
)


import matplotlib.pyplot as plt

def plot_convergence_curves(logs, title="DPIR/HQS convergence"):
    ks = np.array(logs["k"])
    sigma_d = np.array(logs["sigma_d"])
    data_rmse = np.array(logs["data_rmse"])
    xz_rmse = np.array(logs["xz_rmse"])
    dx_rel = np.array(logs["dx_rel"])
    dz_rel = np.array(logs["dz_rel"])
    psnr_z = np.array(logs["psnr_z"])

    # 1) data fidelity + x-z consistency
    plt.figure()
    plt.plot(ks, data_rmse, marker="o", label="RMSE(y - Hx_k)")
    plt.plot(ks, xz_rmse, marker="o", label="RMSE(x_k - z_k)")
    plt.xlabel("iteration k")
    plt.ylabel("RMSE")
    plt.title(title + " | data & consistency")
    plt.legend()
    plt.grid(True)

    # 2) relative changes
    plt.figure()
    plt.plot(ks, dx_rel, marker="o", label="rel change x_k")
    plt.plot(ks, dz_rel, marker="o", label="rel change z_k")
    plt.xlabel("iteration k")
    plt.ylabel("relative change")
    plt.title(title + " | relative updates")
    plt.legend()
    plt.grid(True)

    # 3) PSNR(z_k) (if available)
    if np.isfinite(psnr_z).any():
        plt.figure()
        plt.plot(ks, psnr_z, marker="o")
        plt.xlabel("iteration k")
        plt.ylabel("PSNR(z_k) [dB]")
        plt.title(title + " | PSNR(z_k)")
        plt.grid(True)

    # 4) sigma schedule
    plt.figure()
    plt.plot(ks, sigma_d, marker="o")
    plt.xlabel("iteration k")
    plt.ylabel("sigma_d (pixel space)")
    plt.title(title + " | sigma schedule")
    plt.grid(True)

    plt.show()

plot_convergence_curves(logs, title="k0 sigma=2.55")
