# Image Restoration PROJECT — IRCNN (Sigma-Map) & DRUNet (Sigma-Map)

This repository contains an **image restoration** project built around the **Plug-and-Play (PnP)** paradigm:
- a **CNN denoiser** acts as an image prior (IRCNN or DRUNet),
- combined with a **data-fidelity step** (FFT / closed-form solve) to handle:

  - **Denoising**
  - **Deblurring (deconvolution)** via **PnP-HQS**
  - **Super-Resolution** via **PnP-HQS**

> Key point: the IRCNN here is a **sigma-map conditioned variant** (4th input channel), so a **single model** can handle multiple noise levels (instead of training one model per σ).  
> DRUNet is used in the “paper-style” way: **input = noisy RGB + sigma map**.


### 1) IRCNN (sigma-map conditioned)
- IRCNN-like **7-layer dilated CNN** (dilations: 1–2–3–4–3–2–1).
- **Sigma-map conditioning** (4-channel input: RGB + σ-map).
- Training on random patches with AWGN where σ is sampled uniformly in a range.

Supported tasks:
- **Denoising**
- **PnP-HQS Deblurring** (FFT-based x-update + IRCNN denoiser step)
- **PnP-HQS Super-Resolution** (HQS loop with denoiser prior)

### 2) DRUNet (sigma-map conditioned)
- DRUNet-style U-Net denoiser conditioned on a sigma-map.
- **Denoising** (train + test/inference).
- The model is designed to be easily plugged into PnP loops (deblurring/SR) as a drop-in denoiser prior.

---

## Repository structure

`BSDS300/`  
  Clean training/test images (e.g., BSDS300).
- `models/`
  - `ircnn_sigmap.py` — IRCNN sigma-map network
  - `drunet_sigmap.py` — DRUNet sigma-map network
- `train/`
  - `train_ircnn.py`
  - `train_drunet.py`
- `pnp/`
  - `deblur_pnp_hqs.py`
  - `sr_pnp_hqs.py`
- `scripts/` or notebooks:
  - demo scripts / evaluation scripts
- `weights_*` (ignored by git)
  - model checkpoints (`.pth`) are **not pushed** to GitHub (file size limit)

---

## Sigma-map conditioning (how it works)
Each training sample is built as:
- `clean`: RGB patch in `[0, 1]` with shape `(3, H, W)`
- `noisy = clean + n`, where `n ~ N(0, (σ/255)^2)`
- `sigma_map`: constant map of shape `(1, H, W)` filled with `σ/255`
- `inp = cat([noisy, sigma_map], dim=0)` → shape `(4, H, W)`

So the network input is **(noisy RGB + noise level map)**.


## Setup

### Requirements
- Python 3.9+ (recommended)
- PyTorch
- torchvision
- numpy, pillow, tqdm

