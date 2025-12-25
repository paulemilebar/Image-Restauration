from IRCNN import IRCNNSigmaMap
import os, random, math
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

def psf_to_otf(psf: torch.Tensor, H: int, W: int):
    """
    psf: (kh, kw) float tensor, sum=1
    returns OTF: (H, W) complex tensor
    """
    kh, kw = psf.shape
    pad = torch.zeros((H, W), device=psf.device, dtype=psf.dtype)
    pad[:kh, :kw] = psf

    pad = torch.roll(pad, shifts=(-(kh//2), -(kw//2)), dims=(0, 1))
    otf = torch.fft.fft2(pad)
    return otf

@torch.no_grad()
def ircnn_denoise_sigma(model, x01, sigma_pixels, device):
    """
    x01: (1,3,H,W) in [0,1]
    sigma_pixels: float in pixel scale (0..50)
    """
    H, W = x01.shape[-2], x01.shape[-1]
    sigma_map = torch.full((1, 1, H, W), float(sigma_pixels) / 255.0, device=device, dtype=x01.dtype)
    inp = torch.cat([x01.to(device), sigma_map], dim=1)
    out = model(inp).clamp(0.0, 1.0)
    return out

@torch.no_grad()
def deblur_pnp_hqs(
    model,
    y01,                  # (1,3,H,W) in [0,1], blurred (+ noise)
    psf,                  # (kh,kw) tensor, sum=1
    sigma_n_pixels=2.0,   # noise level in pixel units (0..)
    sigmas_pixels=None,   # list/tuple of denoiser sigmas (pixels), decreasing
    device="cuda"
):
    """
    Plug-and-Play HQS for non-blind deblurring with circular boundary (FFT).
    """
    model.eval()

    y = y01.to(device)
    B, C, H, W = y.shape
    assert B == 1 and C == 3

    if sigmas_pixels is None:
        # schedule typique: fort prior -> plus doux
        sigmas_pixels = [9.00, 8.5, 8, 7.5, 7, 6.5,
  6, 5.5,  5,   4.5,  4,  3.5,
  3,  2.5,  2]

    psf = psf.to(device, dtype=y.dtype)
    K = psf_to_otf(psf, H, W)                 # (H,W) complex
    Kc = torch.conj(K)
    K2 = (Kc * K).real                        # |K|^2 real

    # FFT de y (par canal)
    Y = torch.fft.fft2(y, dim=(-2, -1))       # (1,3,H,W) complex

    # init
    x = y.clone()
    z = x.clone()

    sigma_n = float(sigma_n_pixels) / 255.0
    sigma_n2 = sigma_n * sigma_n + 1e-12

    for s in sigmas_pixels:
        # relie sigma_denoise à beta : sigma_denoise ~= 1/sqrt(beta) (en [0,1])
        sigma_d = float(s) / 255.0
        beta = 1.0 / (sigma_d * sigma_d + 1e-12)

        # x-update (data fidelity) en FFT:
        # x = argmin (1/2σn^2)||Kx - y||^2 + (β/2)||x - z||^2
        # => X = (K*Y/σn^2 + β Z) / (|K|^2/σn^2 + β)
        Z = torch.fft.fft2(z, dim=(-2, -1))
        numerator = (Kc[None, None, :, :] * Y) / sigma_n2 + beta * Z
        denom = (K2[None, None, :, :] / sigma_n2 + beta)
        X = numerator / denom
        x = torch.fft.ifft2(X, dim=(-2, -1)).real.clamp(0.0, 1.0)

        # z-update via denoiser (Plug-and-Play proximal)
        z = ircnn_denoise_sigma(model, x, sigma_pixels=s, device=device)

    return z.clamp(0.0, 1.0)

def gaussian_psf(ksize=15, sigma=3.0, device="cpu", dtype=torch.float32):
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    psf = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf = psf / psf.sum()
    return psf

def blur_circular(x01, psf):
    """
    x01: (1,3,H,W)
    psf: (kh,kw)
    circular blur via FFT
    """
    device = x01.device
    B, C, H, W = x01.shape
    K = psf_to_otf(psf.to(device, dtype=x01.dtype), H, W)
    X = torch.fft.fft2(x01, dim=(-2, -1))
    Y = X * K[None, None, :, :]
    y = torch.fft.ifft2(Y, dim=(-2, -1)).real
    return y

def psnr_torch01(x, y, eps=1e-8):
    mse = torch.mean((x - y) ** 2).item()
    return 10.0 * math.log10(1.0 / (mse + eps))

@torch.no_grad()
def test_deblur_mode_A(
    clean_path,
    ckpt_path,
    out_dir="test_deblur",
    ksize=15,
    blur_sigma=10.0,
    sigma_n_pixels=2.0,
    seed=0
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    model = IRCNNSigmaMap().to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    clean_pil = Image.open(clean_path).convert("RGB")
    clean = TF.to_tensor(clean_pil).unsqueeze(0).to(device)

    psf = gaussian_psf(ksize=ksize, sigma=blur_sigma, device=device, dtype=clean.dtype)

    # blur
    blurred = blur_circular(clean, psf).clamp(0,1)

    # ajouter bruit
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(blurred.shape, device=device, dtype=blurred.dtype, generator=g) * (sigma_n_pixels / 255.0)

    y = (blurred + noise).clamp(0,1)

    # deblur PnP
    xhat = deblur_pnp_hqs(
        model,
        y01=y,
        psf=psf,
        sigma_n_pixels=sigma_n_pixels,
        sigmas_pixels=[9.00, 8.5, 8, 7.5, 7, 6.5,
  6, 5.5,  5,   4.5,  4,  3.5,
  3,  2.5,  2],
        device=device
    )

    # metrics
    psnr_blur = psnr_torch01(blurred, clean)
    psnr_y = psnr_torch01(y, clean)
    psnr_hat = psnr_torch01(xhat, clean)

    # save
    TF.to_pil_image(clean.squeeze(0).cpu()).save(os.path.join(out_dir, "clean.png"))
    TF.to_pil_image(blurred.squeeze(0).cpu()).save(os.path.join(out_dir, "blurred.png"))
    TF.to_pil_image(y.squeeze(0).cpu()).save(os.path.join(out_dir, f"blurred_noisy_sigN{int(sigma_n_pixels)}.png"))
    TF.to_pil_image(xhat.squeeze(0).cpu()).save(os.path.join(out_dir, "deblurred_pnp.png"))

    print("Saved to:", out_dir)
    print(f"PSNR blurred     : {psnr_blur:.2f} dB")
    print(f"PSNR blurred+noise: {psnr_y:.2f} dB")
    print(f"PSNR deblurred   : {psnr_hat:.2f} dB")

@torch.no_grad()
def deblur_real(
    noisy_blur_path,
    ckpt_path,
    out_dir="test_deblur",
    ksize=15,
    blur_sigma=3.0,
    sigma_n_pixels=2.0
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = IRCNNSigmaMap().to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    y_pil = Image.open(noisy_blur_path).convert("RGB")
    y = TF.to_tensor(y_pil).unsqueeze(0).to(device)

    psf = gaussian_psf(ksize=ksize, sigma=blur_sigma, device=device, dtype=y.dtype)

    xhat = deblur_pnp_hqs(
        model,
        y01=y,
        psf=psf,
        sigma_n_pixels=sigma_n_pixels,
        sigmas_pixels=[9.00, 8.5, 8, 7.5, 7, 6.5,
  6, 5.5,  5,   4.5,  4,  3.5,
  3,  2.5,  2],
        device=device
    )

    base = os.path.splitext(os.path.basename(noisy_blur_path))[0]
    TF.to_pil_image(xhat.squeeze(0).cpu()).save(os.path.join(out_dir, f"{base}_deblurred_pnp.png"))
    print("Saved to:", out_dir)

import glob

clean_dir = r"./BSDS300/images/test" 
candidates = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    candidates += glob.glob(os.path.join(clean_dir, "**", ext), recursive=True)

assert len(candidates) > 0, f"Aucune image trouvée dans {clean_dir}"
clean_path = random.choice(candidates)
print("Image choisie:", clean_path)

ckpt_path = r"weights_ircnn_sigmap/ircnn_sigmap_final.pth" 

# 3) Paramètres de flou + bruit pour le test
out_dir = "test_deblur_IRCNN"
ksize = 19
blur_sigma = 1.6       # flou gaussien 
sigma_n_pixels = 2   # bruit rajouté

test_deblur_mode_A(
    clean_path=clean_path,
    ckpt_path=ckpt_path,
    out_dir=out_dir,
    ksize=ksize,
    blur_sigma=blur_sigma,
    sigma_n_pixels=sigma_n_pixels,
    seed=0
)

print("Fichiers générés dans:", out_dir)
print(" - clean.png")
print(" - blurred.png")
print(" - blurred_noisy_sigN*.png")
print(" - deblurred_pnp.png")
