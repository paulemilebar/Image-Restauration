import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import math
from IRCNN_v2 import IRCNNModelManager
import matplotlib.pyplot as plt

# --- Fonctions de dégradation ---
def gaussian_psf(ksize=15, sigma=1.6, device='cpu'):
    coords = torch.arange(ksize, device=device) - ksize // 2
    g = torch.exp(-(coords**2)/(2*sigma**2))
    g = g / g.sum()
    psf = g[:, None] @ g[None, :]
    psf = psf.unsqueeze(0).unsqueeze(0)
    return psf

def downsample(x, scale=2, mode='bilinear'):
    if x.dim()==3:
        x = x.unsqueeze(0)
    H, W = x.shape[2], x.shape[3]
    y = F.interpolate(x, size=(H//scale, W//scale), mode=mode, align_corners=False)
    return y.squeeze(0) if x.dim()==3 else y

def add_gaussian_noise(x, noise_sigma=5):
    noise = torch.randn_like(x) * (noise_sigma/255.0)
    return (x + noise).clamp(0,1)

# --- Solve fidelity via Conjugate Gradient ---
def solve_fidelity_cg(y, z, psf, mu, scale=2, n_iter_cg=10):
    _, C, H, W = z.shape
    ksize = psf.shape[-1]
    pad = ksize // 2

    # Opérateur H: blur + downsample
    def H_op(x):
        x_blur = F.conv2d(x, psf.expand(C,1,ksize,ksize), padding=pad, groups=C)
        x_ds = F.interpolate(x_blur, scale_factor=1/scale, mode='bilinear', align_corners=False)
        return x_ds

    # Opérateur H^T: upsample + conv transpose
    def HT_op(x):
        x_up = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        psf_flip = torch.flip(psf, [2,3])
        x_blur = F.conv2d(x_up, psf_flip.expand(C,1,ksize,ksize), padding=pad, groups=C)
        return x_blur

    def A(x):
        return HT_op(H_op(x)) + mu * x

    y = y.unsqueeze(0) if y.dim()==3 else y
    b = HT_op(y) + mu * z

    # CG initialization
    x_cg = z.clone()
    r = b - A(x_cg)
    p = r.clone()
    rsold = torch.sum(r*r)

    for i in range(n_iter_cg):
        Ap = A(p)
        alpha = rsold / (torch.sum(p*Ap) + 1e-8)
        x_cg = x_cg + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(r*r)
        if torch.sqrt(rsnew) < 1e-6:
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew

    return x_cg

# --- PSNR ---
def calculate_psnr(x, y, eps=1e-8):
    mse = torch.mean((x - y)**2).item()
    return 10 * math.log10(1.0 / (mse + eps))

# --- Fonction principale ---
def test_deblur_pnp(
    image_path,
    model_dir=r"./IRCNN_v2/weights_ircnn_experts_colab",
    out_dir=r"./IRCNN_v2/tests_super_resolution",
    sigma_n_pixels=2,
    lambda_pnp=3,
    ksize=7,
    n_iter=10,
    blur_sigma=1.6,
    scale=5,
    noise_sigma=5
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cpu")

    # Charger image HR
    img_pil = Image.open(image_path).convert("RGB")
    clean = TF.to_tensor(img_pil).unsqueeze(0).to(device)
    _, C, H, W = clean.shape

    # Créer PSF
    psf = gaussian_psf(ksize=ksize, sigma=blur_sigma, device=device)

    # Dégradée LR
    y_blur = F.conv2d(clean, psf.expand(C,1,ksize,ksize), padding=ksize//2, groups=C)
    y_lr = downsample(y_blur, scale=scale)
    y = add_gaussian_noise(y_lr, noise_sigma=noise_sigma)

    # Préparer sigmas
    sigmas_k = np.logspace(np.log10(49), np.log10(sigma_n_pixels), n_iter)
    manager = IRCNNModelManager(model_dir, device=device)

    # Initialisation x0
    x = F.interpolate(y, size=(H,W), mode='bicubic', align_corners=False)
    psnr_degraded = calculate_psnr(x, clean)

    best_psnr = -float("inf")
    best_x = x.clone()
    best_mu = 0
    psnr_list = []
    mu_list = []

    print(f"Début du défloutage HQS ({n_iter} itérations)...")
    for i, sk in enumerate(sigmas_k):
        mu = lambda_pnp*scale**2 / (sk**2)
        # Solve fidelity via CG
        x = solve_fidelity_cg(y, x, psf, mu, scale=scale, n_iter_cg=10)
        # Prior CNN
        expert = manager.get_expert(sk)
        with torch.no_grad():
            x = expert.denoise(x).clamp(0,1)

        # PSNR intermédiaire
        p = calculate_psnr(x, clean)
        psnr_list.append(p)
        mu_list.append(mu)
        if p > best_psnr:
            best_psnr = p
            best_x = x.clone()
            best_mu = mu

    improvement = best_psnr - psnr_degraded
    print(f"PSNR Degraded : {psnr_degraded:.2f} dB")
    print(f"PSNR Restored : {best_psnr:.2f} dB for mu = {best_mu:.2f}")
    print(f"Gain         : {improvement:.2f} dB")

    # 4. Sauvegarde et résultats
    img_id = os.path.splitext(os.path.basename(image_path))[0]
    TF.to_pil_image(clean.squeeze(0)).save(os.path.join(out_dir, f"{img_id}_input_cleaned.png"))
    TF.to_pil_image(y.squeeze(0)).save(os.path.join(out_dir, f"{img_id}_input_degraded.png"))
    TF.to_pil_image(best_x.squeeze(0)).save(os.path.join(out_dir, f"{img_id}_output_restaured.png"))
    print(f"Terminé ! Images sauvegardées dans le dossier '{out_dir}'.")
    plt.plot(mu_list,psnr_list,'o')
    plt.plot([0],[psnr_degraded], color='orange', marker='x')
    plt.axvline(x=best_mu, color='red', linestyle='--', linewidth=2)
    plt.show()


import random

# Choix aléatoire de l'image test
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
clean_dir = r"./BSDS300/images/test" 
paths = [
            os.path.join(clean_dir, f) for f in os.listdir(clean_dir)
            if f.lower().endswith(IMG_EXT)
        ]

assert len(paths) > 0, f"Aucune image trouvée dans {clean_dir}"
image_path = random.choice(paths)
print("Image choisie:", image_path)
print(test_deblur_pnp(image_path=image_path))