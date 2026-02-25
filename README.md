# PnP Image Restoration with Deep Denoiser Priors
### IRCNN vs. DRUNet (+ IRCNN+ conditioned on sigma map)

This repository implements a **Plug-and-Play (PnP)** image restoration framework based on **HQS** (Half-Quadratic Splitting) and **learned deep denoiser priors**.

We **re-implemented from scratch and trained the 3 deep learned denoiser**:
- **IRCNN** (multiple CNN specialized models),
- **DRUNet** (residual U-Net conditioned on a sigma map),
- **IRCNN+ (ours)**: an IRCNN variant **conditioned on a sigma map**, similar to DRUNet conditioning.

**Image restauration Tasks covered:** denoising, deblurring, single-image super-resolution (SISR), and inpainting.

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

---

## Repository structure

- `BSDS300/` : dataset (train and test)
- `IRCNN/` : IRCNN implementation + training (multi-sigma models) + restauration tasks
- `IRCNN_sigmamap/` : IRCNN+ implementation (sigma-map conditioning of the IRCNN) + restauration tasks
- `DRUNet/` : DRUNet implementation + training + restauration tasks
- `benchmark/` : evaluation scripts (PSNR, comparisons, multi-image runs)
- `kernels/` : blur kernels (e.g., Levin kernels) used for image degradation
- `results_DRUNET/` : saved results / plots / reconstructions
- `weights_ircnn_sigmap/` : checkpoints (notably sigma-map conditioned models)
- `poster_project.png` : project poster

## NOTA BENE

The weights of our trained model could have not been pushed to the github because they are too big. We have it in local instead. Don't hesitate to email me, if you want the path.

---

## Tests & Reproducibility

The repository is structured around the two main folders:

- `DRUNet/`: all scripts for DRUNet-based PnP restoration (denoising, deblurring, SISR, inpainting)
- `IRCNN/`: all scripts for IRCNN-based PnP restoration
- `IRCNN_sigmamap/`: all scripts for IRCNN with conditionned sigma map PnP restoration

### How to run an experiment

For each task script, you will find **two entry points**:
- **Single-image run**: restores one image and saves the reconstruction.
- **Benchmark run**: restores a batch of test images and outputs average PSNR/SSIM.

To choose what to run, simply **comment / uncomment the corresponding function call** at the bottom of the script (function names are explicit).

### Required assets 

To reproduce the results, you need:

1) **Test images**
- e.g. `BSDS300/images/test/` (or your own images but don't forget to change the images path in the function arguments)

2) **Kernels (for deblurring / SISR)**
- for exemple `kernels/Levin09.npy` used for deblurring

3) **Model checkpoints (denoiser weights)**
- DRUNet sigma-map conditioned weights
- IRCNN bank of experts (multi-sigma checkpoints)
- IRCNN+ (sigma-map conditioned) weights

> **Nota bene:** model weights may not be included in this GitHub repository (large files).  
> If you need the trained checkpoints used in the reported numbers, please contact me.

### Paths to edit before running

Before running, make sure you set the correct paths in the function call you execute:
- `clean_path` (single image) or `test_dir` (benchmark folder)
- `ckpt_path` (path to denoiser weights)
- `levin09_path` / kernel paths
- `out_dir`

If dependencies are installed and paths are correct, the scripts should run end-to-end and save results automatically.

### Outputs

Results are saved into model-specific folders:

- `results_DRUNET/` for ALL DRUNet experiments
- `results_IRCNN/` for ALL IRCNN experiments
- `results_IRCNN_sigmamap/` for IRCNN+ sigma-map experiments

Each folder typically contains subfolders named after the **task** and whether it was a **single-image** run or a **benchmark** run.



## References

- Kai Zhang, Wangmeng Zuo, Shuhang Gu, Lei Zhang. *Learning Deep CNN Denoiser Prior for Image Restoration* (2017)
- Kai Zhang, Yawei Li, Wangmeng Zuo. *Plug-and-Play Image Restoration with Deep Denoiser Prior* (2021)
