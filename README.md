# PnP Image Restoration with Deep Denoiser Priors — IRCNN vs DRUNet

Projet de Deep Learning / Image Restoration (CentraleSupélec — Université Paris-Saclay)


**Objectif :** construire un framework **Plug-and-Play (PnP)** basé sur **HQS** utilisant des **priors de débruitage appris** (IRCNN / IRCNN+sigma map / DRUNet) pour résoudre plusieurs problèmes inverses : **débruitage**, **défloutage**, **super-résolution (SISR)** et **inpainting**.  
En plus des performances (PSNR), le projet inclut une étude de **convergence** (PSNR(x_k), pas relatifs, RMSE) et une analyse des **représentations latentes** internes de DRUNet (energy maps, top channels, PCA).

---

## 1) Contexte : Plug-and-Play + HQS

On considère un modèle de dégradation :

\[
y = Hx + n
\]

et un objectif MAP :

\[
\hat x = \arg\min_x \frac12\|y - Hx\|_2^2 + \lambda \phi(x)
\]

On introduit une variable auxiliaire `z` et on résout via **Half-Quadratic Splitting (HQS)** :

\[
\mathcal{L}_{\mu}(x,z)=\frac12\|y-Hx\|_2^2+\lambda\,\phi(z)+\frac{\mu}{2}\|z-x\|_2^2
\]

Mises à jour alternées :

\[
x^{k+1}=(H^\top H+\mu_k I)^{-1}(H^\top y+\mu_k z^k)
\]

\[
z^{k+1}=\arg\min_z \frac12\|z-x^{k+1}\|_2^2+\frac{\lambda}{\mu_k}\phi(z)
\]

Le **z-step** est équivalent à un **débruitage gaussien** de niveau :

\[
\sigma_k=\sqrt{\lambda/\mu_k}
\]

et s’écrit :

\[
z^{k+1} = D_{\sigma_k}(x^{k+1})
\]

où \(D_{\sigma}\) est un **débruiteur profond** (IRCNN / DRUNet).

---

## 2) Modèles entraînés

Nous comparons 3 débruiteurs CNN :

- **IRCNN** (prior CNN résiduel)
- **IRCNN + noise level map** (*sigma map*) : entrée 4 canaux (RGB + carte σ)
- **DRUNet** : **Residual U-Net** conditionné par une **sigma map** (entrée 4 canaux)

💡 La **sigma map** est une carte constante de même taille que l’image (valeur σ/255), concaténée à l’entrée RGB pour conditionner le réseau au niveau de bruit.

---

## 3) Dataset & entraînement

- **Dataset :** 400 images naturelles **BSDS300** (train sur patches aléatoires)
- **Augmentations :** flips, rotations 90°
- **Corruption :** AWGN avec bruit \(\sigma \sim \mathcal{U}([\sigma_{min}, \sigma_{max}])\)
- **Optimisation :** Adam
- **Loss :** \(\ell_1\) entre image débruitée et ground truth :

\[
\mathcal{L}(\theta)=\|f_\theta(y,\sigma)-x\|_1
\]

---

## 4) Résultats (poster)

PSNR moyen (dB) sur un pool de 20 images (BSDS300 test) :

| Method | Denoise σ=20 | Deblur | SISR | Inpaint |
|---|---:|---:|---:|---:|
| Degraded | 22.37 | 23.31 | 27.15 | 12.86 |
| IRCNN | 30.9 | 26.14 | — | — |
| IRCNN + | 29.91 | 25.50 | — | — |
| **DRUNet** | **32.8** | **33.63** | **32.22** | **29.22** |

Notes :
- SISR “Degraded” = bicubic SR (point de départ).
- Deblur : kernel Levin.
- Inpaint : missing ratio ≈ 0.42.

---

## 5) Convergence : métriques suivies

En plus du PSNR, on trace plusieurs diagnostics pendant les itérations HQS / DPIR-like :

- **PSNR(x_k)** vs itération  
- **Relative update** :
  \[
  \frac{\|x_{k+1}-x_k\|_2}{\|x_0\|_2}
  \]
  (souvent en log-scale)
- **RMSE** (selon le contexte), par ex :
  - RMSE(\(x_k - z_k\)) : cohérence data/denoise step
  - RMSE(\(y - Hx_k\)) : fidélité au modèle de dégradation

Ces courbes permettent de vérifier :
- stabilisation des itérés,
- impact des paramètres (\(\lambda\), \(\mu_k\), scheduling de σ),
- comportement comparé IRCNN vs DRUNet.

---

## 6) Analyse des représentations internes (DRUNet latent analysis)

Le repo inclut une analyse “interprétabilité” des activations internes de DRUNet :

### a) Feature maps & Energy maps
À un niveau interne \(F \in \mathbb{R}^{C \times H \times W}\) (feature maps), on définit une **energy map** :

\[
E(h,w) = \sqrt{\sum_{c=1}^{C} F_c(h,w)^2}
\]

Cela résume **où le réseau s’active** (tous canaux confondus).

### b) Top channels
On visualise des **channels** (feature maps) sélectionnés parmi les \(C\) canaux, typiquement ceux qui ont une forte **variance spatiale** (ils “bougent” beaucoup dans l’image) — pratique pour voir des canaux edge/texture/noise-like.

### c) PCA du bottleneck
On extrait un embedding du bottleneck (GAP sur `x4`) et on projette en 2D par PCA pour observer la sensibilité des représentations au niveau de bruit σ.

---

## 7) Installation

### Prérequis
- Python ≥ 3.9
- PyTorch + torchvision
- numpy, pillow, matplotlib
- (optionnel) scikit-learn pour PCA

Exemple :
```bash
pip install -r requirements.txt


