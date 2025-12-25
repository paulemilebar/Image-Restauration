from IRCNN import IRCNNSigmaMap
import os, random, math, glob
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np

# PnP Super-Resolution (IRCNN sigma-map denoiser prior) + test on 1 random image from BSDS300
# - Back-projection step repeated 5x per iter
# - 30 main iterations
# - alpha=1.75
# - sigma schedule: exponential from 12*sf to 1*sf (in "0..255" scale like the paper)
# Saves: hr.png, lr.png, bicubic.png, sr_pnp.png (+ optional blurred LR)


clean_dir = r"./BSDS300/images/test"
ckpt_path = r"weights_ircnn_sigmap/ircnn_sigmap_final.pth"
out_dir  = "test_sisr_pnp"
os.makedirs(out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

sf = 2
n_iter = 30
n_bp = 5               # repeat back-projection 5 times
alpha = 1.75
paper_bp_sign = +1     # +1 is the common IBP form; set to -1 to match the sign in the text literally

# Degradation settings (choose one)
use_gaussian_blur = False
gauss_ksize = 7
gauss_sigma = 1.6


def to_torch_rgb01(pil_img: Image.Image) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  # (1,3,H,W)

def to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().clamp(0,1)[0].permute(1,2,0).cpu().numpy()
    return Image.fromarray((t*255.0 + 0.5).astype(np.uint8))

def save_img(t: torch.Tensor, path: str):
    to_pil(t).save(path)

def psnr(x: torch.Tensor, ref: torch.Tensor, eps=1e-12) -> float:
    mse = torch.mean((x - ref) ** 2).item()
    return 10.0 * math.log10(1.0 / max(mse, eps))

def geometric_sigma_schedule(sigma_start: float, sigma_end: float, n: int) -> torch.Tensor:
    if n < 2:
        return torch.tensor([sigma_end], device=device, dtype=torch.float32)
    r = (sigma_end / sigma_start) ** (1.0 / (n - 1))
    return torch.tensor([sigma_start * (r ** k) for k in range(n)], device=device, dtype=torch.float32)

def make_gaussian_kernel2d(ksize=7, sigma=1.6, device=None, dtype=torch.float32):
    assert ksize % 2 == 1
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize // 2)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (k / k.sum())

def blur_rgb(x: torch.Tensor, kernel2d: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    k = kernel2d.shape[-1]
    ker = kernel2d.view(1,1,k,k).to(device=x.device, dtype=x.dtype).repeat(C,1,1,1)
    pad = k // 2
    return F.conv2d(x, ker, padding=pad, groups=C)

def downsample_bicubic(x: torch.Tensor, sf: int) -> torch.Tensor:
    return F.interpolate(x, scale_factor=1.0/sf, mode="bicubic", align_corners=False)

def upsample_bicubic(x: torch.Tensor, sf: int) -> torch.Tensor:
    return F.interpolate(x, scale_factor=sf, mode="bicubic", align_corners=False)

@torch.no_grad()
def denoise_ircnn(model: nn.Module, x: torch.Tensor, sigma_255: float) -> torch.Tensor:
    # x in [0,1], sigma_255 in [0..50] style (0-255 scale); convert to [0,1] for sigma map
    B, C, H, W = x.shape
    sigma_map = torch.full((B,1,H,W), float(sigma_255) / 255.0, device=x.device, dtype=x.dtype)
    inp = torch.cat([x, sigma_map], dim=1)  # (B,4,H,W)
    return model(inp).clamp(0,1)

# Pick a random test image

candidates = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    candidates += glob.glob(os.path.join(clean_dir, "**", ext), recursive=True)
assert len(candidates) > 0, f"Aucune image trouvée dans {clean_dir}"
clean_path = random.choice(candidates)
print("Image choisie:", clean_path)

img_hr_pil = Image.open(clean_path).convert("RGB")
x_hr = to_torch_rgb01(img_hr_pil).to(device)

# crop to multiples of sf (so down/up are clean)
_, _, H, W = x_hr.shape
H2, W2 = (H // sf) * sf, (W // sf) * sf
x_hr = x_hr[:, :, :H2, :W2].contiguous()

# -----------------------------
# Load denoiser weights
# -----------------------------
den = IRCNNSigmaMap(features=64).to(device).eval()
ckpt = torch.load(ckpt_path, map_location=device)

# robust loading
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    sd = ckpt["state_dict"]
elif isinstance(ckpt, dict) and "model" in ckpt:
    sd = ckpt["model"]
else:
    sd = ckpt

# strip possible "module." prefixes
sd2 = {}
for k, v in sd.items():
    sd2[k.replace("module.", "")] = v
den.load_state_dict(sd2, strict=True)
print("Loaded weights:", ckpt_path)

# -----------------------------
# Create LR observation y (synthetic, for testing)
# -----------------------------
kernel = make_gaussian_kernel2d(gauss_ksize, gauss_sigma, device=device) if use_gaussian_blur else None

def degrade(x):
    z = x
    if kernel is not None:
        z = blur_rgb(z, kernel)
    z = downsample_bicubic(z, sf)
    return z

y_lr = degrade(x_hr).clamp(0,1)
x_bic = upsample_bicubic(y_lr, sf).clamp(0,1)

# -----------------------------
# PnP SISR loop (Back-projection + Denoise)
# -----------------------------
sigmas = geometric_sigma_schedule(sigma_start=12.0*sf, sigma_end=1.0*sf, n=n_iter)

x = x_bic.clone()
with torch.no_grad():
    for k in range(n_iter):
        # back-projection repeated n_bp times
        for _ in range(n_bp):
            r = (y_lr - degrade(x))                         # LR residual
            x = (x + paper_bp_sign * alpha * upsample_bicubic(r, sf)).clamp(0,1)
        # denoise step with decaying sigma
        x = denoise_ircnn(den, x, float(sigmas[k].item()))

x_sr = x

save_img(x_hr,  os.path.join(out_dir, "hr.png"))
save_img(y_lr,  os.path.join(out_dir, "lr.png"))
save_img(x_bic, os.path.join(out_dir, "bicubic.png"))
save_img(x_sr,  os.path.join(out_dir, "sr_pnp.png"))
if kernel is not None:
    # purely for debug/visualization: LR already includes blur effect via degrade()
    pass

print(f"Saved outputs to: {out_dir}")
print(f"PSNR bicubic vs HR: {psnr(x_bic, x_hr):.2f} dB")
print(f"PSNR PnP-SR  vs HR: {psnr(x_sr,  x_hr):.2f} dB")

fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(1,3,1); ax1.set_title("HR (ref)");           ax1.imshow(to_pil(x_hr));  ax1.axis("off")
ax2 = fig.add_subplot(1,3,2); ax2.set_title("Bicubic upsample");    ax2.imshow(to_pil(x_bic)); ax2.axis("off")
ax3 = fig.add_subplot(1,3,3); ax3.set_title("PnP-SR (IRCNN prior)");ax3.imshow(to_pil(x_sr));  ax3.axis("off")
plt.show()
