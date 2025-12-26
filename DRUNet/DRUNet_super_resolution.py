import os, math
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

from DRUNet import DRUNetSigmaMap


# ============================================================
# Utils
# ============================================================
def psnr_torch(x, y, eps=1e-12):
    mse = torch.mean((x - y) ** 2).item()
    return 10.0 * math.log10(1.0 / (mse + eps))

def modcrop_tensor(x: torch.Tensor, sf: int) -> torch.Tensor:
    _, _, H, W = x.shape
    H2 = (H // sf) * sf
    W2 = (W // sf) * sf
    return x[..., :H2, :W2]

def save_img01(t: torch.Tensor, path: str):
    TF.to_pil_image(t.squeeze(0).clamp(0, 1).cpu()).save(path)


# ============================================================
# Load kernels_12.mat (paper kernels)
# ============================================================
def loadmat_any(path: str):
    try:
        import hdf5storage
        return hdf5storage.loadmat(path)
    except Exception:
        from scipy.io import loadmat
        return loadmat(path)

def load_kernel12(kernels_mat_path: str, k_index: int) -> np.ndarray:
    md = loadmat_any(kernels_mat_path)
    kernels = md["kernels"]              # expected (1,8) object
    k = kernels[0, k_index]
    k = np.asarray(k, dtype=np.float64)
    k = k / (k.sum() + 1e-12)
    return k


# ============================================================
# DPIR SISR closed-form (Eq. 14) helpers
# ============================================================
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
    FB  = p2o(k, (h*sf, w*sf)) # FB = FFT(k) : opérateur de floutage B dans Fourrier
    FBC = torch.conj(FB)
    F2B = torch.pow(torch.abs(FB), 2)
    STy = upsample_zeros(img_L, sf=sf)
    FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))
    return FB, FBC, F2B, FBFy

def data_solution_closed_form(z_prev, FB, FBC, F2B, FBFy, alpha, sf):
    """
    closed-form x-step (Eq.14) as implemented in DPIR utils_sisr.data_solution
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
    # sigma schedule (pixel space): modelSigma1 -> modelSigma2 (log), then /255
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


# ============================================================
# Main DPIR SISR (minimal outputs + PSNR plot)
# ============================================================
@torch.no_grad()
def run_one(
    clean_path: str,
    ckpt_path: str,
    out_dir: str,
    scale: int = 3,
    sigma_img: float = 7.65,     # LR noise in pixel space (0..255)
    iter_num: int = 24,
    kernels_mat_path: str = "kernels/kernels_12.mat",
    k_index: int = 2,           # choose 0..7
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

    # ---- load paper kernel ----
    if not os.path.isfile(kernels_mat_path):
        raise FileNotFoundError(
            f"Missing {kernels_mat_path}. Put kernels_12.mat in ./kernels/."
        )
    k_np = load_kernel12(kernels_mat_path, k_index=k_index)
    k = torch.from_numpy(k_np.astype(np.float32)).to(device=device, dtype=x.dtype)
    k = k.unsqueeze(0).unsqueeze(0).repeat(1, C, 1, 1)  # (1,3,kh,kw)

    # ---- classical degradation: y = (x ⊗ k)↓s + n  with circular BC ----
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

    # ---- DPIR params (K=24, sigma_K=max(sigma,s)) ----
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


    # ---- iterations: x_k (closed-form) then z_k (DRUNet) ----
    psnr_x = []
    psnr_z = []

    for i in range(iter_num):
        alpha = rhos_t[i].view(1, 1, 1, 1)

        xk = data_solution_closed_form(z, FB2, FBC, F2B, FBFy, alpha, scale).clamp(0, 1)

        sigma_i = float(sigmas_t[i].item())  # normalized [0,1]
        sigma_map = torch.full((B, 1, H, W), sigma_i, device=device, dtype=xk.dtype)
        inp = torch.cat([xk, sigma_map], dim=1)

        z = drunet_infer(model, inp, modulo=8).clamp(0, 1)

        psnr_x.append(psnr_torch(xk.cpu(), x.cpu()))
        psnr_z.append(psnr_torch(z.cpu(),  x.cpu()))

    # ---- save restored ----
    save_img01(z, os.path.join(out_dir, "restored.png"))

    print("Saved to:", out_dir)
    print("Shapes HR / LR / Restored:", tuple(x.shape), tuple(y.shape), tuple(z.shape))
    print(f"Final PSNR (z_K): {psnr_z[-1]:.2f} dB")

    # ---- plot PSNR curves ----
    it = np.arange(1, iter_num + 1)
    plt.figure()
    plt.plot(it, psnr_x, label="PSNR(x_k)  (data step)")
    plt.plot(it, psnr_z, label="PSNR(z_k)  (after DRUNet)")
    plt.xlabel("Iteration k")
    plt.ylabel("PSNR (dB)")
    plt.title(f"DPIR SISR PSNR curves (sf={scale}, sigma={sigma_img}, k_index={k_index})")
    plt.grid(True)
    plt.legend()
    plt.show()

run_one(
        clean_path="./BSDS300/images/test/37073.jpg",
        ckpt_path="./weights_drunet_sigmap/drunet_sigmap_final.pth",
        out_dir="test_outputs_dpir_sisr",
        scale=2,
        sigma_img=0,
        iter_num=24,
    )
