import os, math
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch.fft
from IRCNN import IRCNNFixed
import matplotlib.pyplot as plt

# --- Fonctions Utilitaires ---

def gaussian_psf(ksize=15, sigma=1.6, device="cpu"):
    # On crée une grille de coordonnées centrée
    coords = torch.arange(ksize, device=device).float() - (ksize - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = g.view(-1, 1) * g.view(1, -1)
    kernel /= kernel.sum() # Normalisation cruciale pour garder la luminosité
    return kernel.view(1, 1, ksize, ksize)


def solve_fidelity_fft(y, z, psf, mu, eps=1e-8):
    # y: (1,3,H,W), z: (1,3,H,W), psf: (1,1,k,k)
    _, _, H, W = y.shape
    device = y.device
    ksize = psf.shape[-1]
    p = ksize // 2

    # 1. Padding de la PSF à la taille de l'image
    psf_padded = torch.zeros((1, 1, H, W), device=device, dtype=y.dtype)
    psf_padded[:, :, :ksize, :ksize] = psf

    # 2. Centrage du noyau (Roll) : INDISPENSABLE pour l'alignement
    psf_padded = torch.roll(psf_padded, shifts=(-p, -p), dims=(-2, -1))

    # 3. FFT
    Y = torch.fft.fft2(y, dim=(-2, -1))
    Z = torch.fft.fft2(z, dim=(-2, -1))
    H_fft = torch.fft.fft2(psf_padded, dim=(-2, -1))

    # 4. Formule de Wiener (Fidélité)
    H_conj = torch.conj(H_fft)
    # On utilise mu directement comme poids du prior
    denominator = torch.abs(H_fft)**2 + mu + eps
    x_hat = (H_conj * Y + mu * Z) / denominator

    return torch.real(torch.fft.ifft2(x_hat, dim=(-2, -1))).clamp(0,1)


class IRCNNModelManager:
    """ Gère le chargement dynamique des 10 experts. """
    def __init__(self, model_dir, device="cpu"):
        self.model_dir = model_dir
        self.device = device
        self.available_sigmas = [2*i for i in range(1,26)]
        self.model = IRCNNFixed(n_filters=64).to(device)
        self.current_sigma = None

    def get_expert(self, target_sigma):
        # Trouver l'expert le plus proche (ex: target 12.5 -> expert 15)
        closest = min(self.available_sigmas, key=lambda x: abs(x - target_sigma))
        if closest != self.current_sigma:
            path = os.path.join(self.model_dir, f"ircnn_sigma_{closest}_final.pth")
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state["model"])
            self.model.eval()
            self.current_sigma = closest
        return self.model


def calculate_psnr(x, y, eps=1e-8):
    """
    Calcule le PSNR entre deux images img1 et img2 (tensors entre 0 et 1).
    """
    mse = torch.mean((x - y) ** 2).item()
    return 10.0 * math.log10(1.0 / (mse + eps))


# --- Fonction principale de Défloutage ---

def test_deblur_pnp(
    image_path,
    model_dir=r"./IRCNN_v2/weights_ircnn_experts_colab",
    out_dir=r"./IRCNN_v2/tests_deblur",
    sigma_n_pixels=2, # Bruit réel de l'image (estimé)
    lambda_pnp=3,     # Paramètre de régularisation (à augmenter avec niveau de flou+bruit) -> trouver méthode pour l'estimer en cas d'image inconnue, entre 2 et 5 d'après mon petit test
    ksize=15,
    n_iter=20,
    blur_sigma=1.6,
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

    # Ajout du bruit
    noise = torch.randn_like(y_blurred) * (noise_sigma / 255.0)
    y = (y_blurred + noise).clamp(0, 1)

    # 2. Préparer les itérations (30 itérations de sigma=49 à sigma=sigma_n)
    manager = IRCNNModelManager(model_dir, device=device)
    sigmas_k = np.logspace(np.log10(49), np.log10(sigma_n_pixels), n_iter)

    # 3. Calcul des PSNR
    psnr_blurred = calculate_psnr(y_blurred, clean)
    psnr_blurred_noisy = calculate_psnr(y, clean)
    
    x = y.clone()
    best_psnr = -float("inf")
    best_x = x.clone()
    best_mu = 0
    mus=[0]
    psnr1, psnr2=[psnr_blurred_noisy], [psnr_blurred_noisy]
    
    #print(f"Début du défloutage HQS ({n_iter} itérations)...")
    for i, sk in enumerate(sigmas_k):
        #sigma_d = sk / 255.0
        mu = lambda_pnp / (sk**2)

        # Étape A : Fidélité (FFT)
        x = solve_fidelity_fft(y, x, psf, mu)
        psnr1.append(calculate_psnr(x, clean))

        # Étape B : Prior (Expert CNN)
        expert = manager.get_expert(sk)
        with torch.no_grad():
            x = expert.denoise(x).clamp(0, 1)
        mus.append(mu)
        p = calculate_psnr(x, clean)
        psnr2.append(p)
        if p > best_psnr:
            best_psnr = p
            best_x = x.clone()
            best_mu = mu

    # psnr_deblurred = calculate_psnr(x, clean)
    improvement = best_psnr - psnr_blurred_noisy

    print(f"\n--- Résultats ---")
    print(f"PSNR Blurred       : {psnr_blurred:.2f} dB")
    print(f"PSNR Blurred+Noisy : {psnr_blurred_noisy:.2f} dB")
    print(f"PSNR Deblurred     : {best_psnr:.2f} dB for mu = {best_mu:.2f}")
    print(f"Gain               : {improvement:.2f} dB")

    # 4. Sauvegarde et résultats
    img_id = os.path.splitext(os.path.basename(image_path))[0]
    TF.to_pil_image(clean.squeeze(0)).save(os.path.join(out_dir, f"{img_id}_input_cleaned.png"))
    TF.to_pil_image(y.squeeze(0)).save(os.path.join(out_dir, f"{img_id}_input_blurred_noisy.png"))
    TF.to_pil_image(best_x.squeeze(0)).save(os.path.join(out_dir, f"{img_id}_output_deblurred.png"))
    print(f"Terminé ! Images sauvegardées dans le dossier '{out_dir}'.")
    '''plt.plot(mus,psnr1,'o')
    plt.plot(mus,psnr2,'x')
    plt.axvline(x=best_mu, color='red', linestyle='--', linewidth=2)
    plt.show()'''

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
test_deblur_pnp(image_path=image_path)

# Importance de lambda selon niveau de flou et bruit

'''
scenarios = [
    {"blur_sigma": 1.0, "noise_sigma": 2,  "label": "Blur=1.0 / Noise=2"},
    {"blur_sigma": 1.6, "noise_sigma": 5,  "label": "Blur=1.6 / Noise=5"},
    {"blur_sigma": 2.2, "noise_sigma": 10, "label": "Blur=2.2 / Noise=10"},
]

lambda_list = [0.1, 0.2, 0.5, 1, 2, 5, 10]

best_psnrs = []

results = {}  # {label: [psnr_lambda1, psnr_lambda2, ...]}

for scen in scenarios:
    label = scen["label"]
    blur_sigma = scen["blur_sigma"]
    noise_sigma = scen["noise_sigma"]

    print(f"\n=== Scenario: {label} ===")
    psnrs = []

    for lam in lambda_list:
        print(f"  -> lambda = {lam}")

        best_psnr = test_deblur_pnp(
            image_path=image_path,
            lambda_pnp=lam,
            n_iter=10,
            blur_sigma=blur_sigma,
            noise_sigma=noise_sigma,
            sigma_n_pixels=2,
            ksize=15
        )

        psnrs.append(best_psnr)
        print(f"     Best PSNR = {best_psnr:.2f} dB")

    results[label] = psnrs
    
plt.figure(figsize=(8,6))

for label, psnrs in results.items():
    plt.semilogx(lambda_list, psnrs, marker='o', label=label)

plt.xlabel("lambda (regularization weight)")
plt.ylabel("Best PSNR over iterations (dB)")
plt.title("Influence of lambda for different blur / noise levels (IRCNN Deblurring)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
'''
