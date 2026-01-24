# PnP Image Restoration with Deep Denoiser Priors
### IRCNN vs. DRUNet (+ IRCNN+ conditioned on sigma map)

This repository implements a **Plug-and-Play (PnP)** image restoration framework based on **HQS** (Half-Quadratic Splitting) and **learned deep denoiser priors**.

We **re-implemented and trained**:
- **IRCNN** (paper-like: multiple specialized models),
- **DRUNet** (residual U-Net conditioned on a sigma map),
- **IRCNN+ (ours)**: an IRCNN variant **conditioned on a sigma map**, similar to DRUNet conditioning.

**Tasks covered:** denoising, deblurring, single-image super-resolution (SISR), and inpainting.

---

## Key idea (PnP + HQS)

We consider the degradation model:

- `y = Hx + n`

and a MAP-like objective:

- minimize: `0.5 * ||y - Hx||_2^2 + lambda * phi(x)`

Using HQS, we introduce an auxiliary variable `z` and solve:

- minimize over `(x, z)`:
  `0.5 * ||y - Hx||_2^2 + lambda * phi(z) + (mu/2) * ||z - x||_2^2`

We alternate between:

### 1) x-step (data fidelity)
Closed-form update:

- `x_{k+1} = (H^T H + mu_k I)^(-1) (H^T y + mu_k z_k)`

### 2) z-step (prior via denoiser)
Replace the proximal step by a learned denoiser:

- `z_{k+1} = D_{sigma_k}(x_{k+1})`
- with `sigma_k = sqrt(lambda / mu_k)`

---

## Models

### IRCNN
- Residual CNN denoiser prior.
- **Paper-like setup:** train **25 specialized models** for discrete noise levels  
  `sigma in {2, 4, ..., 50}`.
- Learns **noise prediction** (predicts the noise, not the clean image).

### IRCNN+ (ours)
- Same spirit as IRCNN but **conditioned on a sigma map**.
- Input becomes **4 channels**: RGB + constant sigma map.
- Goal: a more flexible IRCNN-style denoiser across noise levels.

### DRUNet
- Residual U-Net denoiser **conditioned on a sigma map** (RGB + sigma map).
- Learns **image prediction** (predicts the clean image).

---

## Inverse problems implemented

Same PnP solver, only the forward operator `H` changes:

- **Denoising:** `y = x + n`
- **Deblurring:** `y = k * x + n` (convolution kernel)
- **SISR:** `y = D(k * x) + n` (blur + downsampling)
- **Inpainting:** `y = M ⊙ x + n` (binary mask)

---

## Dataset & training

- **Dataset:** BSDS300 (400 natural images for training).
- Training uses random clean patches + augmentation (flips + 90° rotations).
- Noise: AWGN with `sigma ~ Uniform([sigma_min, sigma_max])` for conditioned models.
- **Optimizer:** Adam
- **Loss:** L1

Targets:
- IRCNN  → **noise target**
- DRUNet → **clean image target**

---

## Results (PSNR on 20 BSDS300 test images)

Average PSNR (dB) ± std:

| Method   | Denoise (σ=20) | Deblur (Levin kernel, σ=5) | SISR (Levin kernel, ×2) | Inpainting (missing=0.15, σ=2) |
|----------|------------------|----------------------------|--------------------------|--------------------------------|
| Degraded | 22.36 ± 0.27     | 22.47 ± 3.27               | 25.19 ± 2.81             | 15.53 ± 1.55                   |
| IRCNN    | 29.91 ± 1.11     | 24.80 ± 1.20               | 29.65 ± 3.71             | 30.12 ± 2.08                   |
| IRCNN+   | 29.11 ± 1.11     | 25.26 ± 3.05               | 29.83 ± 3.45             | 26.95 ± 1.59                   |
| DRUNet   | **31.70 ± 1.93** | **30.18 ± 3.77**           | **30.01 ± 3.71**         | **30.30 ± 2.08**               |

**Overall:** DRUNet performs best across all tasks in our experiments.

---

## Repository structure

- `BSDS300/` : dataset (train and test)
- `IRCNN/` : IRCNN implementation + training (multi-sigma models)
- `IRCNN+/` : IRCNN+ implementation (sigma-map conditioning)
- `DRUNet/` : DRUNet implementation + training + restoration tasks
- `benchmark/` : evaluation scripts (PSNR, comparisons, multi-image runs)
- `kernels/` : blur kernels (e.g., Levin kernels) used for image degradation
- `results_DRUNET/` : saved results / plots / reconstructions
- `weights_ircnn_sigmap/` : checkpoints (notably sigma-map conditioned models)
- `poster_project.png` : project poster

---

## References

- Kai Zhang, Wangmeng Zuo, Shuhang Gu, Lei Zhang. *Learning Deep CNN Denoiser Prior for Image Restoration* (2017)
- Kai Zhang, Yawei Li, Wangmeng Zuo. *Plug-and-Play Image Restoration with Deep Denoiser Prior* (2021)
