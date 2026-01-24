import os, math
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch.fft
from IRCNN import IRCNNFixed, IRCNNModelManager
import matplotlib.pyplot as plt
import torch.nn.functional as F



# --- Fonctions de dégradation (déjà définies) ---
def gaussian_psf(ksize=15, sigma=1.6, device="cpu"):
    # On crée une grille de coordonnées centrée
    coords = torch.arange(ksize, device=device).float() - (ksize - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = g.view(-1, 1) * g.view(1, -1)
    kernel /= kernel.sum() # Normalisation cruciale pour garder la luminosité
    return kernel.view(1, 1, ksize, ksize)

def downsample(x, scale=2, mode='bicubic'):
    H, W = x.shape[2], x.shape[3]
    y = F.interpolate(x, size=(H//scale, W//scale), mode=mode, align_corners=False)
    return y

def add_gaussian_noise(x, noise_sigma=5):
    noise = torch.randn_like(x) * (noise_sigma/255.0)
    return (x + noise).clamp(0.0, 1.0)


# --- Solve fidelity ---
def solve_fidelity(y, z, psf, mu):
    """
    Résout x_{k+1} = argmin ||y - Hx||^2 + mu ||x - z||^2
    """
    _,_,H,W = z.shape
    
    # Upsample y pour correspondre à HR
    if y.dim() == 3:
        y = y.unsqueeze(0)
    y_up = F.interpolate(y, size=(H,W), mode='bicubic', align_corners=False).squeeze(0)

    ksize = psf.shape[-1]
    p = ksize // 2

    # 1. Padding de la PSF à la taille de l'image
    psf_padded = torch.zeros((1, 1, H, W), device=z.device, dtype=z.dtype)
    psf_padded[:, :, :ksize, :ksize] = psf

    # 2. Centrage du noyau (Roll) : INDISPENSABLE pour l'alignement
    psf_padded = torch.roll(psf_padded, shifts=(-p, -p), dims=(-2, -1))

    H_f = torch.fft.fft2(psf_padded)
    H_f_conj = torch.conj(H_f)
    z_f = torch.fft.fft2(z)
    y_f = torch.fft.fft2(y_up)

    x_f = (H_f_conj * y_f + mu * z_f) / (H_f_conj * H_f + mu)
    x = torch.fft.ifft2(x_f).real
    return x

# --- PSNR robuste ---
def calculate_psnr(x, y, eps=1e-8):
    """
    Calcule le PSNR entre deux images x et y (tensors [0,1])
        y = y.squeeze(0)"""
    # Mettre y à la même taille que x si nécessaire
    if x.shape != y.shape:
        y = F.interpolate(y, size=(x.shape[1], x.shape[2]), mode='bicubic', align_corners=False)
    mse = torch.mean((x - y) ** 2).item()
    return 10.0 * math.log10(1.0 / (mse + eps))

# --- Pipeline IRCNN PnP ---
def test_deblur_pnp(
    image_path,
    model_dir=r"./IRCNN_v2/weights_ircnn_experts_colab",
    out_dir=r"./IRCNN_v2/tests_super_resolution",
    sigma_n_pixels=2,
    lambda_pnp=3,
    ksize=15,
    n_iter=10,
    blur_sigma=1.6,
    scale=5,
    noise_sigma=5
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cpu")

    # 1. Charger l'image et préparer le flou
    img_pil = Image.open(image_path).convert("RGB")
    clean = TF.to_tensor(img_pil).unsqueeze(0).to(device)
    _, _, H, W = clean.shape

    ksize = 15
    p = ksize // 2
    psf = gaussian_psf(ksize=ksize, sigma=blur_sigma, device=device)

    psf_padded = torch.zeros((1, 1, H, W), device=device)
    psf_padded[:, :, :ksize, :ksize] = psf
    psf_padded = torch.roll(psf_padded, shifts=(-p, -p), dims=(2, 3))

    # Flou
    clean_fft = torch.fft.fft2(clean)
    psf_fft = torch.fft.fft2(psf_padded)
    y_blurred = torch.real(torch.fft.ifft2(clean_fft * psf_fft))
    y_lr = downsample(y_blurred, scale=scale)
    y = add_gaussian_noise(y_lr, noise_sigma=noise_sigma)
    
    # Préparer les sigmas
    sigmas_k = np.logspace(np.log10(49), np.log10(sigma_n_pixels), n_iter)
    manager = IRCNNModelManager(model_dir, device=device)
    
    x=y.clone()
    x = F.interpolate(x, size=(H,W), mode='bicubic', align_corners=False)
    psnr_degraded = calculate_psnr(x, clean)
    best_psnr = -float("inf")
    best_x = y.clone()
    best_mu = 0
    psnr_list = []
    mu_list = []

    print(f"Début du défloutage HQS ({n_iter} itérations)...")
    for i, sk in enumerate(sigmas_k):
        mu = lambda_pnp / (sk**2)
        '''print(f"etape {i}")
        print(f"0 shape : {x.shape}")'''
        # Étape A : fidélité
        x = solve_fidelity(y, x, psf, mu)
        
        # Étape B : prior (CNN)
        expert = manager.get_expert(sk)
        with torch.no_grad():
            x  = expert.denoise(x).clamp(0,1)
        
        # PSNR intermédiaire
        p = calculate_psnr(x, clean)
        psnr_list.append(p)
        mu_list.append(mu)
        if p > best_psnr:
            best_psnr = p
            best_x = x.clone()
            best_mu = mu

    improvement = best_psnr - psnr_degraded

    print(f"\n--- Résultats ---")
    print(f"PSNR Degraded  : {psnr_degraded:.2f} dB")
    print(f"PSNR Restaured : {best_psnr:.2f} dB for mu = {best_mu:.2f}")
    print(f"Gain           : {improvement:.2f} dB")

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
'''IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
clean_dir = r"./BSDS300/images/test" 
paths = [
            os.path.join(clean_dir, f) for f in os.listdir(clean_dir)
            if f.lower().endswith(IMG_EXT)
        ]

assert len(paths) > 0, f"Aucune image trouvée dans {clean_dir}"
image_path = random.choice(paths)'''
image_path = r"./BSDS300/images/test\119082.jpg"
print("Image choisie:", image_path)
test_deblur_pnp(image_path=image_path)
