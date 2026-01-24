import sys
import os
from pathlib import Path
import random
import statistics

# Ajoute le dossier parent (Image-Restauration) au chemin de recherche de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Maintenant tu peux importer
from IRCNN.IRCNN_denoise import denoise_ircnn
from IRCNN.IRCNN_denoise import denoise_ircnnv2
from DRUNet.DRUNet_denoise import denoise_drunet

# Paramètres
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
clean_dir = r"./BSDS300/images/test"
benchmark_dir = r"./benchmark/denoise"

# Liste de toutes les images
all_paths = [
    os.path.join(clean_dir, f)
    for f in os.listdir(clean_dir)
    if f.lower().endswith(IMG_EXT)
]

assert len(all_paths) > 0, f"Aucune image trouvée dans {clean_dir}"

# Mélange aléatoire et sélection des 20 premières
random.shuffle(all_paths)
num_samples = min(20, len(all_paths))
selected_paths = all_paths[:num_samples]

# Paramètres du bruit appliqués aux images
sigma_noise = 20

# Stockage des psnr pour chaque modèle
psnr_degraded = []
psnr_ircnn = []
psnr_ircnnv2 = []
psnr_drunet = []

for img_path in selected_paths:
    print("Image choisie:", img_path) 
    # On crée un dossier pour l'image choisie pour visualiser les images restaurées des différents modèles
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    img_dir = os.path.join(benchmark_dir, img_id)
    os.makedirs(img_dir, exist_ok=True)
    
    deg, ir = denoise_ircnn(clean_path=img_path, sigma=sigma_noise)
    psnr_degraded.append(deg)
    psnr_ircnn.append(ir)
    psnr_ircnnv2.append(denoise_ircnnv2(image_path=img_path, noise_sigma=sigma_noise))
    psnr_drunet.append(denoise_drunet(clean_path=img_path, sigma=sigma_noise))

    # remplir pour drunet : psnr_drunet.append(...)
    

print(f"PSNR degraded : {statistics.mean(psnr_degraded):.2f} +- {statistics.pstdev(psnr_degraded):.2f}")
print(f"PSNR IRCNN : {statistics.mean(psnr_ircnn):.2f} +- {statistics.pstdev(psnr_ircnn):.2f}")
print(f"PSNR IRCNN_v2 : {statistics.mean(psnr_ircnnv2):.2f} +- {statistics.pstdev(psnr_ircnnv2):.2f}")
print(f"PSNR DRUNet : {statistics.mean(psnr_drunet):.2f} +- {statistics.pstdev(psnr_drunet):.2f}")
