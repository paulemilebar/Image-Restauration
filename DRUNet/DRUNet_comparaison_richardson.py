import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from DRUNet import DRUNetSigmaMap
from DRUNet_deblur import load_levin09_kernel, psf_to_otf, circ_conv_fft, dpir_hqs_deblur
from DRUNet_deblur import psnr_torch
# -------------------------
# Baselines WITHOUT DPIR (no denoiser prior)
# -------------------------

@torch.no_grad()
def deblur_inverse_filter(y: torch.Tensor, otf: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Naive inverse filtering: X = Y / (H + eps)
    Very sensitive to zeros in H and noise.
    """
    Y = torch.fft.fft2(y, dim=(-2, -1))
    H = otf[None, None, :, :]
    X = Y / (H + eps)
    x = torch.fft.ifft2(X, dim=(-2, -1)).real
    return x.clamp(0.0, 1.0)

@torch.no_grad()
def deblur_tikhonov_l2(y: torch.Tensor, otf: torch.Tensor, beta: float = 2e-3) -> torch.Tensor:
    """
    L2/Tikhonov (a.k.a. Wiener with flat prior):
      min_x ||y - Hx||^2 + beta ||x||^2
    Closed-form in Fourier:
      X = conj(H)Y / (|H|^2 + beta)
    """
    Y = torch.fft.fft2(y, dim=(-2, -1))
    Hc = torch.conj(otf)[None, None, :, :]
    H2 = (otf.real**2 + otf.imag**2)[None, None, :, :]
    X = (Hc * Y) / (H2 + beta)
    x = torch.fft.ifft2(X, dim=(-2, -1)).real
    return x.clamp(0.0, 1.0)

@torch.no_grad()
def richardson_lucy_circular(y: torch.Tensor, otf: torch.Tensor, iters: int = 30, eps: float = 1e-8) -> torch.Tensor:
    """
    Richardson-Lucy for circular convolution (Poisson-likelihood style).
    Not great with Gaussian noise unless you stop early or denoise.
    Update:
      x <- x * (H^T (y / (H x + eps)))
    with H^T implemented by conj(H) in Fourier.
    """
    # init
    x = y.clamp(0.0, 1.0).clone()

    H = otf
    Hc = torch.conj(otf)

    for _ in range(iters):
        # Hx
        Hx = circ_conv_fft(x, H).clamp_min(eps)
        ratio = (y / Hx).clamp(0.0, 10.0)  # limit crazy values
        corr = circ_conv_fft(ratio, Hc)
        x = (x * corr).clamp(0.0, 1.0)

    return x

# -------------------------
# Test runner: compare baselines vs DPIR output
# -------------------------
@torch.no_grad()
def test_deblur_compare_baselines_vs_dpir(
    clean_path: str,
    levin09_path: str,
    kernel_index: int,
    sigma_img: float,
    seed: int,
    # baseline params
    inv_eps: float = 1e-3,
    beta_l2: float = 2e-3,
    rl_iters: int = 30,
    out_dir: str = "test_outputs_deblur_compare",
    # DPIR params (call your existing DPIR function)
    dpir_fn=None,   # pass a function like: lambda y, otf: dpir_hqs_deblur(...)
):
    """
    dpir_fn should be a callable that takes (y, otf) and returns restored (B,3,H,W).
    Example:
      dpir_fn = lambda y, otf: dpir_hqs_deblur(y, otf, model, sigma_img=2.55, lam=0.23, n_iter=8)
    """
    assert dpir_fn is not None, "Please pass dpir_fn (your DPIR deblurring function wrapper)."
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load clean
    clean_pil = Image.open(clean_path).convert("RGB")
    x = TF.to_tensor(clean_pil).unsqueeze(0).to(device)
    B, C, H, W = x.shape

    # load kernel + otf
    k_np = load_levin09_kernel(levin09_path, kernel_index)
    k = torch.from_numpy(k_np).to(device)
    otf = psf_to_otf(k, (H, W))

    # synthesize y = Hx + n
    blurry = circ_conv_fft(x, otf)
    sigma_n = sigma_img / 255.0
    try:
        g = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(blurry.shape, device=blurry.device, dtype=blurry.dtype, generator=g) * sigma_n
    except TypeError:
        torch.manual_seed(seed)
        noise = torch.randn(blurry.shape, device=blurry.device, dtype=blurry.dtype) * sigma_n

    y = (blurry + noise).clamp(0.0, 1.0)

    # baselines
    x_inv = deblur_inverse_filter(y, otf, eps=inv_eps)
    x_l2  = deblur_tikhonov_l2(y, otf, beta=beta_l2)
    x_rl  = richardson_lucy_circular(y, otf, iters=rl_iters)

    # dpir (your function)
    x_dpir = dpir_fn(y, otf).clamp(0.0, 1.0)

    # PSNRs
    psnr_y    = psnr_torch(y.detach().cpu(), x.detach().cpu())
    psnr_inv  = psnr_torch(x_inv.detach().cpu(), x.detach().cpu())
    psnr_l2   = psnr_torch(x_l2.detach().cpu(), x.detach().cpu())
    psnr_rl   = psnr_torch(x_rl.detach().cpu(), x.detach().cpu())
    psnr_dpir = psnr_torch(x_dpir.detach().cpu(), x.detach().cpu())

    # save
    TF.to_pil_image(x.squeeze(0).cpu()).save(os.path.join(out_dir, "clean.png"))
    TF.to_pil_image(y.squeeze(0).cpu()).save(os.path.join(out_dir, f"blurry_k{kernel_index}_sigma{sigma_img:.2f}.png"))
    TF.to_pil_image(x_inv.squeeze(0).cpu()).save(os.path.join(out_dir, f"invfilter_eps{inv_eps}.png"))
    TF.to_pil_image(x_l2.squeeze(0).cpu()).save(os.path.join(out_dir, f"tikhonov_beta{beta_l2}.png"))
    TF.to_pil_image(x_rl.squeeze(0).cpu()).save(os.path.join(out_dir, f"richardsonlucy_it{rl_iters}.png"))
    TF.to_pil_image(x_dpir.squeeze(0).cpu()).save(os.path.join(out_dir, f"dpir.png"))

    # kernel vis
    k_vis = (k / (k.max() + 1e-12)).detach().cpu().numpy()
    k_vis = (k_vis * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(k_vis).save(os.path.join(out_dir, f"kernel_k{kernel_index}.png"))

    print("Saved to:", out_dir)
    print(f"PSNR blurry/noisy : {psnr_y:.2f} dB")
    print(f"PSNR inverse filt : {psnr_inv:.2f} dB")
    print(f"PSNR Tikhonov L2  : {psnr_l2:.2f} dB   (beta={beta_l2})")
    print(f"PSNR RichardsonLucy: {psnr_rl:.2f} dB  (iters={rl_iters})")
    print(f"PSNR DPIR (PnP)   : {psnr_dpir:.2f} dB")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DRUNetSigmaMap(in_nc=4, out_nc=3, nc=(64,128,256,512), nb=4).to(device).eval()
state = torch.load("./weights_drunet_sigmap/drunet_sigmap_final.pth", map_location=device)
sd = state["model"] if isinstance(state, dict) and "model" in state else state
model.load_state_dict(sd, strict=True)

dpir_fn = lambda y, otf: dpir_hqs_deblur(
    y=y, otf=otf, denoiser=model,
    sigma_img=2.55, lam=0.23, n_iter=8, sigma_max=49.0
)

test_deblur_compare_baselines_vs_dpir(
    clean_path="./BSDS300/images/test/102061.jpg",
    levin09_path="kernels/Levin09.npy",
    kernel_index=0,
    sigma_img=2.55,
    seed=0,
    beta_l2=2e-3,
    rl_iters=30,
    out_dir="results_DRUNET/results_DRUNET_comparaison_deblur",
    dpir_fn=dpir_fn
)
