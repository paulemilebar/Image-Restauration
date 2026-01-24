import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from IRCNN import FixedSigmaPatchDataset, IRCNNFixed

def train_all_experts(
    clean_dir=r"./BSDS300/images/train",
    base_out_dir=r"./IRCNN_v2/weights_ircnn_experts_colab",
    patch=35,
    sigmas=[2*i for i in range(1,26)],
    batch_size=8,
    steps_per_epoch=500,
    max_epochs=5, # Moins d'époques car on utilise le transfert d'apprentissage
    lr0=1e-3,     # LR plus bas pour stabiliser le transfert
    # lr1=1e-4,
    tolerance=0.015,
    log_every=50
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Démarrage de l'entraînement sur : {device}")

    # Initialisation du modèle unique (3 canaux -> 3 canaux)
    # On l'instancie une seule fois pour faire le transfert d'apprentissage
    model = IRCNNFixed().to(device)
    loss_fn = nn.MSELoss()
    scaler = GradScaler(enabled=(device == "cuda"))

    # Boucle sur chaque niveau de bruit
    for sigma in sigmas:
        print(f"\n{'='*20}")
        print(f" ENTRAÎNEMENT EXPERT SIGMA = {sigma}")
        print(f"{'='*20}")

        out_dir = os.path.join(base_out_dir, f"sigma_{sigma}")
        os.makedirs(out_dir, exist_ok=True)

        # Dataset fixe pour ce niveau de bruit
        ds = FixedSigmaPatchDataset(clean_dir, patch=patch, sigma=float(sigma))
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=(device == "cuda"), drop_last=True
        )

        opt = Adam(model.parameters(), lr=lr0)
        best = float("inf")
        stagnant = 0
        using_lr1 = False
        previous_loss = float('inf')

        for epoch in range(1, max_epochs + 1):
            model.train()
            running = 0.0
            start_t = time.time()

            pbar = tqdm(total=steps_per_epoch, desc=f"Sigma {sigma} - Ep {epoch}/{max_epochs}", leave=True)
            for step, (noisy, clean) in enumerate(dl):
                if step >= steps_per_epoch:
                    break

                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)

                # Cible résiduelle (le bruit)
                target_noise = noisy - clean

                with autocast(enabled=(device == "cuda")):
                    # Le modèle prédit le bruit (via la méthode forward ou get_noise)
                    pred_noise = model.forward(noisy)
                    loss = loss_fn(pred_noise, target_noise)

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                running += loss.item()

                if (step + 1) % log_every == 0:
                    avg_so_far = running / (step + 1)
                    pbar.set_postfix({"loss": f"{avg_so_far:.5f}", "lr": f"{opt.param_groups[0]['lr']:.1e}"})
                pbar.update(1)
            pbar.close()

            avg_loss = running / steps_per_epoch

            # Sauvegarde du point de passage
            ckpt_path = os.path.join(out_dir, f"ircnn_s{sigma}_epoch{epoch:02d}.pth")
            torch.save({"model": model.state_dict(), "sigma": sigma}, ckpt_path)

            # LOGIQUE D'ARRÊT ANTICIPÉ (Early Stopping)
            improvement = (previous_loss - avg_loss) / previous_loss if previous_loss != float('inf') else 1.0

            if improvement < tolerance:
                print(f"  [INFO] Convergence atteinte pour Sigma {sigma} (Amélioration: {improvement:.2%}). Passage au suivant.")
                break # On sort de la boucle d'époques pour ce sigma

            previous_loss = avg_loss

        # Sauvegarde finale pour cet expert avant de passer au sigma suivant
        final_sigma_path = os.path.join(base_out_dir, f"ircnn_sigma_{sigma}_final.pth")
        torch.save({"model": model.state_dict()}, final_sigma_path)
        print(f"Expert {sigma} terminé et sauvegardé.")

    print("\nFélicitations ! Les 25 experts ont été entraînés avec succès.")

train_all_experts()