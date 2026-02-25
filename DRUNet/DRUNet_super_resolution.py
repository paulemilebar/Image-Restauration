import os
from DRUNet_denoise import psnr_torch, ssim_torch, list_images
import numpy as np
from PIL import Image
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import pandas as pd
from DRUNet import DRUNetSigmaMap
from scipy.io import loadmat

def modcrop_tensor(x: torch.Tensor, sf: int) -> torch.Tensor:
    _, _, H, W = x.shape
    H2 = (H // sf) * sf
    W2 = (W // sf) * sf
    return x[..., :H2, :W2]

def save_img01(t: torch.Tensor, path: str):
    TF.to_pil_image(t.squeeze(0).clamp(0, 1).cpu()).save(path)

def load_kernel12(path="kernels/kernels_12.mat", k_index=0) -> np.ndarray:
    k = loadmat(path)["kernels"][0, k_index]
    k = np.asarray(k, dtype=np.float32)
    k /= k.sum()
    return k

# DPIR SISR closed-form
def splits(a: torch.Tensor, sf: int) -> torch.Tensor:
    # a: N x C x H x W -> N x C x (H/sf) x (W/sf) x (sf^2)
    # sf: scale factor
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b

def p2o(psf: torch.Tensor, shape) -> torch.Tensor:
    # psf: N x C x h x w (real)
    otf = torch.zeros(psf.shape[:-2] + shape, dtype=psf.dtype, device=psf.device)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
    return torch.fft.fftn(otf, dim=(-2, -1))

def upsample_zeros(x: torch.Tensor, sf: int) -> torch.Tensor:
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf),
                    device=x.device, dtype=x.dtype)
    z[..., st::sf, st::sf].copy_(x)
    return z

def downsample_decimate(x: torch.Tensor, sf: int) -> torch.Tensor:
    st = 0
    return x[..., st::sf, st::sf]

def pre_calculate(img_L: torch.Tensor, k: torch.Tensor, sf: int):
    h, w = img_L.shape[-2:]
    FB  = p2o(k, (h*sf, w*sf)) # blurring operatour B in Fourrier domain
    FBC = torch.conj(FB)
    F2B = torch.pow(torch.abs(FB), 2)
    STy = upsample_zeros(img_L, sf=sf)
    FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))
    return FB, FBC, F2B, FBFy

def data_solution_closed_form(z_prev, FB, FBC, F2B, FBFy, alpha, sf):
    """
    closed-form x-step
    """
    FR = FBFy + torch.fft.fftn(alpha * z_prev, dim=(-2, -1))
    x1 = FB.mul(FR)
    FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
    invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
    invWBR = FBR.div(invW + alpha)
    FCBinvWBR = FBC * invWBR.repeat(1, 1, sf, sf)
    FX = (FR - FCBinvWBR) / alpha
    x_est = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))
    return x_est

def shift_pixel_torch(x: torch.Tensor, sf: int, upper_left: bool = True) -> torch.Tensor:
    # shift = (sf-1)/2, bilinear grid interpolation (handles "upper-left" downsampler shift)
    B, C, H, W = x.shape
    shift = (sf - 1) * 0.5
    if not upper_left:
        shift = -shift

    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=torch.float32),
        torch.arange(W, device=x.device, dtype=torch.float32),
        indexing="ij"
    )
    xx = (xx + shift).clamp(0, W - 1)
    yy = (yy + shift).clamp(0, H - 1)

    gx = (xx / (W - 1)) * 2 - 1
    gy = (yy / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1,H,W,2)

    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

def get_rho_sigma(noise_level_model: float, iter_num: int, modelSigma1: float, modelSigma2: float, w: float = 1.0):
    # sigma schedule (pixel space): modelSigma1 -> modelSigma2 (log)
    sigmas = np.exp(np.linspace(np.log(modelSigma1), np.log(modelSigma2), iter_num)).astype(np.float32) / 255.0

    # safeguard for "sigma=0" case
    sigma = max(0.255/255.0, float(noise_level_model))

    # rho schedule
    rhos = (w * (sigma**2) / (sigmas**2 + 1e-12)).astype(np.float32)
    return rhos, sigmas


@torch.no_grad()
def drunet_infer(model, inp, modulo: int = 8):
    b, c, h, w = inp.shape
    pad_h = (modulo - h % modulo) % modulo
    pad_w = (modulo - w % modulo) % modulo
    if pad_h or pad_w:
        inp2 = F.pad(inp, (0, pad_w, 0, pad_h), mode="reflect")
        out = model(inp2)
        return out[..., :h, :w]
    return model(inp)


# Main DPIR SISR (minimal outputs + PSNR plot)
@torch.no_grad()
def run_one(
    clean_path: str,
    ckpt_path: str,
    out_dir: str,
    scale: int = 3,
    sigma_img: float = 7.65,# LR noise in pixel space (0..255)
    iter_num: int = 24,
    kernels_mat_path: str = "kernels/kernels_12.mat",
    k_index: int = 2,
    modelSigma1: float = 49.0,
    seed: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load model ----
    model = DRUNetSigmaMap(in_nc=4, out_nc=3, nc=(64,128,256,512), nb=4).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=True)

    # ---- load HR ----
    x = TF.to_tensor(Image.open(clean_path).convert("RGB")).unsqueeze(0).to(device)
    x = modcrop_tensor(x, scale)
    B, C, H, W = x.shape

    # ---- load kernel ----
    if not os.path.isfile(kernels_mat_path):
        raise FileNotFoundError(
            f"Missing {kernels_mat_path}. Put kernels_12.mat in ./kernels/."
        )
    k_np = load_kernel12(kernels_mat_path, k_index=k_index)
    k = torch.from_numpy(k_np.astype(np.float32)).to(device=device, dtype=x.dtype)
    k = k.unsqueeze(0).unsqueeze(0).repeat(1, C, 1, 1)  # (1,3,kh,kw)

    # ---- classical degradation ----
    FB = p2o(k, (H, W))  # complex OTF
    xb = torch.real(torch.fft.ifftn(torch.fft.fftn(x, dim=(-2, -1)) * FB, dim=(-2, -1)))
    y = downsample_decimate(xb, scale)

    sigma_n = float(sigma_img) / 255.0
    if sigma_n > 0:
        try:
            g = torch.Generator(device=device).manual_seed(seed)
            y = y + torch.randn_like(y, generator=g) * sigma_n
        except TypeError:
            torch.manual_seed(seed)
            y = y + torch.randn_like(y) * sigma_n

    # ---- save only the 3 images
    save_img01(x, os.path.join(out_dir, "clean.png"))
    save_img01(y, os.path.join(out_dir, "lr.png"))

    noise_level_model = sigma_n
    modelSigma2 = max(float(scale), float(noise_level_model * 255.0))
    rhos, sigmas = get_rho_sigma(noise_level_model, iter_num, modelSigma1, modelSigma2, w=1.0)
    rhos_t = torch.tensor(rhos, device=device, dtype=x.dtype)
    sigmas_t = torch.tensor(sigmas, device=device, dtype=x.dtype)

    # ---- pre-calc for Eq.(14) ----
    FB2, FBC, F2B, FBFy = pre_calculate(y, k, scale)

    # ---- init z0 = bicubic(y) + shift correction ----
    z = F.interpolate(y, scale_factor=scale, mode="bicubic", align_corners=False)
    z = shift_pixel_torch(z, sf=scale, upper_left=True).clamp(0, 1)
    save_img01(z, os.path.join(out_dir, "bicubic.png"))

    psnr_bic = psnr_torch(z.cpu(), x.cpu())
    print(f"PSNR bicubic (z0): {psnr_bic:.2f} dB")
    
    ssim_bic = ssim_torch(z.cpu(), x.cpu())
    print(f"SSIM bicubic (z0): {ssim_bic:.2f}")


    # ---- iterations: x_k (closed-form) then z_k (DRUNet) ----
    psnr_x = []
    psnr_z = []
    
    ssim_x = []
    ssim_z = []

    for i in range(iter_num):
        alpha = rhos_t[i].view(1, 1, 1, 1)

        xk = data_solution_closed_form(z, FB2, FBC, F2B, FBFy, alpha, scale).clamp(0, 1)

        sigma_i = float(sigmas_t[i].item())  # normalized [0,1]
        sigma_map = torch.full((B, 1, H, W), sigma_i, device=device, dtype=xk.dtype)
        inp = torch.cat([xk, sigma_map], dim=1)

        z = drunet_infer(model, inp, modulo=8).clamp(0, 1)

        psnr_x.append(psnr_torch(xk.cpu(), x.cpu()))
        psnr_z.append(psnr_torch(z.cpu(),  x.cpu()))
        ssim_x.append(ssim_torch(xk.cpu(), x.cpu()))
        ssim_z.append(ssim_torch(z.cpu(),  x.cpu()))

    # ---- save restored ----
    save_img01(z, os.path.join(out_dir, "restored.png"))

    print("Saved to:", out_dir)
    print("Shapes HR / LR / Restored:", tuple(x.shape), tuple(y.shape), tuple(z.shape))
    print(f"Final PSNR (z_K): {psnr_z[-1]:.2f} dB")
    print(f"Final SSIM (z_K): {ssim_z[-1]:.2f}")

    # ---- plot PSNR curves ----
    it = np.arange(1, iter_num + 1)
    plt.figure()
    plt.plot(it, psnr_x, label="PSNR(x_k)")
    plt.plot(it, psnr_z, label="PSNR(z_k)")
    plt.xlabel("Iteration k")
    plt.ylabel("PSNR (dB)")
    plt.title(f"DRUNET SISR PSNR curves (sf={scale}, sigma={sigma_img}, k_index={k_index})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "psnr_curves.png"), dpi=200)
    plt.show()
    
    # ---- plot SSIM curves ----
    it = np.arange(1, iter_num + 1)
    plt.figure()
    plt.plot(it, ssim_x, label="SSIM(x_k)")
    plt.plot(it, ssim_z, label="SSIM(z_k)")
    plt.xlabel("Iteration k")
    plt.ylabel("SSIM")
    plt.title(f"DRUNET SISR SSIM curves (sf={scale}, sigma={sigma_img}, k_index={k_index})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ssim_curves.png"), dpi=200)
    plt.show()


@torch.no_grad()
def run_one_metrics_sisr(
    clean_path: str,
    model: torch.nn.Module,
    device: torch.device,
    out_dir: str | None,
    scale: int = 2,
    sigma_img: float = 0.0,
    iter_num: int = 24,
    kernels_mat_path: str = "kernels/kernels_12.mat",
    k_index: int = 2,
    modelSigma1: float = 49.0,
    seed: int = 0,
    save_images: bool = False,
):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    # ---- load HR ----
    x = TF.to_tensor(Image.open(clean_path).convert("RGB")).unsqueeze(0).to(device)
    x = modcrop_tensor(x, scale)
    B, C, H, W = x.shape

    # ---- load kernel ----
    if not os.path.isfile(kernels_mat_path):
        raise FileNotFoundError(f"Missing {kernels_mat_path}. Put kernels_12.mat in ./kernels/.")
    k_np = load_kernel12(kernels_mat_path, k_index=k_index)
    k = torch.from_numpy(k_np.astype(np.float32)).to(device=device, dtype=x.dtype)
    k = k.unsqueeze(0).unsqueeze(0).repeat(1, C, 1, 1)  # (1,3,kh,kw)

    # ---- degradation: y = (x ⊗ k)↓s + n ----
    FB = p2o(k, (H, W))
    xb = torch.real(torch.fft.ifftn(torch.fft.fftn(x, dim=(-2, -1)) * FB, dim=(-2, -1)))
    y = downsample_decimate(xb, scale)

    sigma_n = float(sigma_img) / 255.0
    if sigma_n > 0:
        try:
            g = torch.Generator(device=device).manual_seed(seed)
            y = y + torch.randn_like(y, generator=g) * sigma_n
        except TypeError:
            torch.manual_seed(seed)
            y = y + torch.randn_like(y) * sigma_n

    # ---- init z0 = bicubic + shift ----
    z = F.interpolate(y, scale_factor=scale, mode="bicubic", align_corners=False)
    z = shift_pixel_torch(z, sf=scale, upper_left=True).clamp(0, 1)

    psnr_bic = psnr_torch(z.detach().cpu(), x.detach().cpu())
    ssim_bic = ssim_torch(z.detach().cpu(), x.detach().cpu())

    # ---- DPIR params ----
    noise_level_model = sigma_n
    modelSigma2 = max(float(scale), float(noise_level_model * 255.0))
    rhos, sigmas = get_rho_sigma(noise_level_model, iter_num, modelSigma1, modelSigma2, w=1.0)
    rhos_t = torch.tensor(rhos, device=device, dtype=x.dtype)
    sigmas_t = torch.tensor(sigmas, device=device, dtype=x.dtype)

    # ---- pre-calc ----
    FB2, FBC, F2B, FBFy = pre_calculate(y, k, scale)

    # ---- iterations ----
    for i in range(iter_num):
        alpha = rhos_t[i].view(1, 1, 1, 1)
        xk = data_solution_closed_form(z, FB2, FBC, F2B, FBFy, alpha, scale).clamp(0, 1)

        sigma_i = float(sigmas_t[i].item())
        sigma_map = torch.full((B, 1, H, W), sigma_i, device=device, dtype=xk.dtype)
        inp = torch.cat([xk, sigma_map], dim=1)

        z = drunet_infer(model, inp, modulo=8).clamp(0, 1)

    psnr_final = psnr_torch(z.detach().cpu(), x.detach().cpu())
    ssim_final = ssim_torch(z.detach().cpu(), x.detach().cpu())

    # ---- optional save ----
    if save_images and out_dir is not None:
        base = os.path.splitext(os.path.basename(clean_path))[0]
        save_img01(x, os.path.join(out_dir, f"{base}_clean.png"))
        save_img01(y, os.path.join(out_dir, f"{base}_lr.png"))
        save_img01(F.interpolate(y, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0,1),
                   os.path.join(out_dir, f"{base}_lr_bicubic_upsampled.png"))
        save_img01(z, os.path.join(out_dir, f"{base}_restored.png"))

    return {
        "filename": os.path.basename(clean_path),
        "path": clean_path,
        "scale": int(scale),
        "sigma_img": float(sigma_img),
        "k_index": int(k_index),
        "iter_num": int(iter_num),
        "psnr_bicubic_db": float(psnr_bic),
        "psnr_restored_db": float(psnr_final),
        "gain_db": float(psnr_final - psnr_bic),
        "ssim_bicubic": float(ssim_bic),
        "ssim_restored": float(ssim_final),
        "ssim_gain": float(ssim_final - ssim_bic),
    }


@torch.no_grad()
def benchmark_sisr_10_random_to_csv(
    test_dir: str,
    ckpt_path: str,
    out_dir: str = "results_DRUNET_superresolution_benchmark",
    n_images: int = 10,
    seed: int = 0,
    scale: int = 2,
    sigma_img: float = 0.0,
    iter_num: int = 24,
    kernels_mat_path: str = "kernels/kernels_12.mat",
    k_index: int = 2,
    modelSigma1: float = 49.0,
    save_examples: bool = False,
):
    """
    Pick 10 test images, run DPIR-SISR, write CSV and print means.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model once
    model = DRUNetSigmaMap(in_nc=4, out_nc=3, nc=(64,128,256,512), nb=4).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=True)

    all_paths = list_images(test_dir)
    if len(all_paths) < n_images:
        raise RuntimeError(f"Pas assez d'images dans {test_dir}: {len(all_paths)} < {n_images}")

    rng = random.Random(seed)
    chosen = rng.sample(all_paths, n_images)

    rows = []
    for i, p in enumerate(chosen):
        row = run_one_metrics_sisr(
            clean_path=p,
            model=model,
            device=device,
            out_dir=out_dir,
            scale=scale,
            sigma_img=sigma_img,
            iter_num=iter_num,
            kernels_mat_path=kernels_mat_path,
            k_index=k_index,
            modelSigma1=modelSigma1,
            seed=seed + i,
            save_images=save_examples,
        )
        rows.append(row)
        print(f"[{i+1}/{n_images}] {row['filename']} | bic {row['psnr_bicubic_db']:.2f} -> restored {row['psnr_restored_db']:.2f} dB")
        print(f"[{i+1}/{n_images}] {row['filename']} | bic {row['ssim_bicubic']:.2f} -> restored {row['ssim_restored']:.2f}")
    
    df = pd.DataFrame(rows)
    
    metric_cols = [c for c in df.columns if c.startswith("psnr") or c.startswith("gain") or c.startswith("ssim")]
    mean_row = {"filename": "MEAN"}
    std_row  = {"filename": "STD"}
    for c in metric_cols:
        mean_row[c] = float(df[c].mean())
        std_row[c]  = float(df[c].std())
    df2 = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    csv_path = os.path.join(out_dir, f"sisr_benchmark_{n_images}imgs_sf{scale}_sig{sigma_img}_k{k_index}_K{iter_num}_seed{seed}.csv")
    df2.to_csv(csv_path, index=False)


    return df2, csv_path

if __name__ == "__main__":
    # test for 1 image
    run_one(
        clean_path="./BSDS300/images/test/102061.jpg",
        ckpt_path="./weights_drunet_sigmap/drunet_sigmap_final.pth",
        out_dir="results_DRUNET/results_DRUNET_superresolution/castel",
        scale=2,
        sigma_img=0,
        iter_num=20,
    )

    # benchmark for 10 images
   # benchmark_sisr_10_random_to_csv(
   #     test_dir="./BSDS300/images/images_benchmark/benchmark_10_images",
   #     ckpt_path="./weights_drunet_sigmap/drunet_sigmap_final.pth",
   #     out_dir="results_DRUNET/results_DRUNET_superresolution_benchmark",
   #     n_images=10,
   #     seed=0,
   #     scale=2,
   #     sigma_img=0,
   #     iter_num=20,
   #     kernels_mat_path="kernels/kernels_12.mat",
   #     k_index=2,
   #     save_examples=False,
   # )