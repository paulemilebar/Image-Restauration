import os, math
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch.fft
try:
    # Tentative pour quand on lance depuis le Benchmark
    from IRCNN_v2.IRCNN_v2_final import IRCNNModelManager
except ModuleNotFoundError:
    # Repli pour quand on lance le fichier en direct
    from IRCNN_v2_final import IRCNNModelManager
import matplotlib.pyplot as plt

# --- Fonctions Utilitaires ---

def solve_fidelity_denoise(y, z, mu):
     return (y + mu * z) / (1 + mu)


def calculate_psnr(x, y, eps=1e-8):
    """
    Calcule le PSNR entre deux images img1 et img2 (tensors entre 0 et 1).
    """
    mse = torch.mean((x - y) ** 2).item()
    return 10.0 * math.log10(1.0 / (mse + eps))


# --- Fonction principale de Défloutage ---

def test_denoise_pnp(
    image_path,
    model_dir=r"./IRCNN_v2/weights_ircnn_experts_colab",
    out_dir=r"./IRCNN_v2/tests_denoise",
    noise_sigma=40,
    sigma_n_pixels=2, # Bruit réel de l'image (estimé)
    lambda_pnp=1,     # Paramètre de régularisation (à ajuster)
    n_iter=20
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cpu")

    # 1. Charger l'image et préparer le flou
    img_pil = Image.open(image_path).convert("RGB")
    clean = TF.to_tensor(img_pil).unsqueeze(0).to(device)

    # Ajout du bruit
    noise = torch.randn_like(clean) * (noise_sigma / 255.0)
    y = (clean + noise).clamp(0, 1)

    # 2. Préparer les itérations (30 itérations de sigma=49 à sigma=sigma_n)
    manager = IRCNNModelManager(model_dir, device=device)
    sigmas_k = np.logspace(np.log10(49), np.log10(sigma_n_pixels), n_iter)

    x = y.clone()
    best_psnr = -float("inf")
    best_x = x.clone()
    best_mu = 0
    mus=[]
    psnr=[]
    # print(f"Début du défloutage HQS ({n_iter} itérations)...")
    for i, sk in enumerate(sigmas_k):
        # sigma_d = sk / 255.0
        mu = lambda_pnp / (sk **2)

        # Étape A : Fidélité (FFT)
        x = solve_fidelity_denoise(y, x, mu)

        # Étape B : Prior (Expert CNN)
        expert = manager.get_expert(sk)
        with torch.no_grad():
            x = expert.denoise(x).clamp(0, 1)
        mus.append(mu)
        p = calculate_psnr(x, clean)
        psnr.append(p)
        if p > best_psnr:
            best_psnr = p
            best_x = x.clone()
            best_mu = mu

    # 3. Calcul des PSNR
    psnr_noisy = calculate_psnr(y, clean)
    # psnr_denoised = calculate_psnr(x, clean)
    improvement = best_psnr - psnr_noisy

    print(f"\n--- Résultats ---")
    print(f"PSNR Noisy    : {psnr_noisy:.2f} dB")
    print(f"PSNR Denoised : {best_psnr:.2f} dB for mu = {best_mu:.2f}")
    print(f"Gain          : {improvement:.2f} dB")

    # 4. Sauvegarde et résultats
    img_id = os.path.splitext(os.path.basename(image_path))[0]
    TF.to_pil_image(clean.squeeze(0)).save(os.path.join(out_dir, f"{img_id}_input_cleaned.png"))
    TF.to_pil_image(y.squeeze(0)).save(os.path.join(out_dir, f"{img_id}_input_noisy.png"))
    TF.to_pil_image(best_x.squeeze(0)).save(os.path.join(out_dir, f"{img_id}_output_denoised.png"))
    print(f"Terminé ! Images sauvegardées dans le dossier '{out_dir}'.")
    '''plt.plot(mus,psnr,'o')
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


# Importance de lambda selon niveau de bruit -> pas très important

'''
scenarios = [
    {"noise_sigma": 2,  "label": "Noise=2"},
    {"noise_sigma": 10,  "label": "Noise=10"},
    {"noise_sigma": 50, "label": "Noise=50"},
]

lambda_list = [0.1, 0.2, 0.5, 1, 2, 5, 10]

best_psnrs = []

results = {}  # {label: [psnr_lambda1, psnr_lambda2, ...]}

for scen in scenarios:
    label = scen["label"]
    noise_sigma = scen["noise_sigma"]

    print(f"\n=== Scenario: {label} ===")
    psnrs = []

    for lam in lambda_list:
        print(f"  -> lambda = {lam}")

        best_psnr = test_denoise_pnp(
            image_path=image_path,
            noise_sigma=noise_sigma,
            sigma_n_pixels=noise_sigma,
            lambda_pnp=lam,
            n_iter=10
        )

        psnrs.append(best_psnr)
        print(f"     Best PSNR = {best_psnr:.2f} dB")

    results[label] = psnrs
    
plt.figure(figsize=(8,6))

for label, psnrs in results.items():
    plt.semilogx(lambda_list, psnrs, marker='o', label=label)

plt.xlabel("lambda (regularization weight)")
plt.ylabel("Best PSNR over iterations (dB)")
plt.title("Influence of lambda for noise levels (IRCNN Denoising)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
'''



# --- Fonction pour comparer perf ---

def denoise_ircnnv2(
    image_path,
    model_dir=r"./IRCNN_v2/weights_ircnn_experts_colab",
    out_dir=r"./benchmark/denoise",
    noise_sigma=40,
    sigma_n_pixels=2, # Bruit réel de l'image (estimé)
    lambda_pnp=1,     # Paramètre de régularisation (à ajuster)
    n_iter=15
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cpu")

    # 1. Charger l'image et préparer le flou
    img_pil = Image.open(image_path).convert("RGB")
    clean = TF.to_tensor(img_pil).unsqueeze(0).to(device)

    # Ajout du bruit
    noise = torch.randn_like(clean) * (noise_sigma / 255.0)
    y = (clean + noise).clamp(0, 1)

    # 2. Préparer les itérations (30 itérations de sigma=49 à sigma=sigma_n)
    manager = IRCNNModelManager(model_dir, device=device)
    sigmas_k = np.logspace(np.log10(49), np.log10(sigma_n_pixels), n_iter)

    x = y.clone()
    best_psnr = -float("inf")
    best_x = x.clone()
    best_mu = 0
    mus=[]
    psnr=[]
    for i, sk in enumerate(sigmas_k):
        # sigma_d = sk / 255.0
        mu = lambda_pnp / (sk **2)

        # Étape A : Fidélité (FFT)
        x = solve_fidelity_denoise(y, x, mu)

        # Étape B : Prior (Expert CNN)
        expert = manager.get_expert(sk)
        with torch.no_grad():
            x = expert.denoise(x).clamp(0, 1)
        mus.append(mu)
        p = calculate_psnr(x, clean)
        psnr.append(p)
        if p > best_psnr:
            best_psnr = p
            best_x = x.clone()
            best_mu = mu

    # 3. Calcul des PSNR
    psnr_noisy = calculate_psnr(y, clean)
    improvement = best_psnr - psnr_noisy

    # 4. Sauvegarde et résultats
    img_id = os.path.splitext(os.path.basename(image_path))[0]
    img_dir = os.path.join(out_dir, img_id)
    os.makedirs(img_dir, exist_ok=True)
    TF.to_pil_image(clean.squeeze(0)).save(os.path.join(img_dir, f"clean.png"))
    TF.to_pil_image(y.squeeze(0)).save(os.path.join(img_dir, f"noisy.png"))
    TF.to_pil_image(best_x.squeeze(0)).save(os.path.join(img_dir, f"denoised_ircnnv2.png"))
    '''print(f"Terminé ! Images sauvegardées dans le dossier '{img_dir}'.")
    plt.plot(mus,psnr,'o')
    plt.show()'''
    return best_psnr


'''print("Image choisie:", r"./BSDS300/images/test\123456.jpg")
print(denoise_ircnnv2(image_path=image_path))'''