import os, math, random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from IRCNN_sigmamap import IRCNNSigmaMap
from IRCNN_sigmamap_denoise import psnr_torch, ssim_torch, list_images, ircnn_infer
import matplotlib.pyplot as plt
import time
from glob import glob
import pandas as pd


# Utils
def save_img01(t: torch.Tensor, path: str):
    TF.to_pil_image(t.squeeze(0).clamp(0, 1).cpu()).save(path)
    
def l2norm_flat(x: torch.Tensor) -> torch.Tensor:
    return torch.norm(x.reshape(-1), p=2)

def save_convergence_plots(metrics: dict, out_dir: str, prefix: str = "ircnn"):
    os.makedirs(out_dir, exist_ok=True)
    K = len(metrics["psnr_x"])
    it = np.arange(K)
    it2 = np.arange(K-1)
    

    # PSNR
    plt.figure()
    plt.plot(it, metrics["psnr_x"], label="PSNR x_k")
    plt.plot(it, metrics["psnr_z"], label="PSNR z_k")
    plt.xlabel("itération k")
    plt.ylabel("PSNR [dB]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_psnr.png"), dpi=200)
    plt.close()
    
    plt.figure()
    plt.plot(it, metrics["ssim_x"], label="SSIM x_k")
    plt.plot(it, metrics["ssim_z"], label="SSIM z_k")
    plt.xlabel("itération k")
    plt.ylabel("SSIM")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_ssim.png"), dpi=200)
    plt.close()

    # relative change
    relx = np.array(metrics["rel_step_x"])
    relz = np.array(metrics["rel_step_z"])
    relx = np.maximum(relx, 1e-16)
    relz = np.maximum(relz, 1e-16)
    plt.figure()
    plt.semilogy(it2, relx, label=r"$(x_k-x_{k-1})/(x_0)$")
    plt.semilogy(it2, relz, label=r"$\|z_k-z_{k-1}\|/\|z_0\|$")    
    plt.xlabel("iteration k")
    plt.ylabel(r"relsafe")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_rel_step_log.png"), dpi=200)
    plt.close()

    # Cumulative sum
    cx = np.array(metrics["cumsum_x"])
    cz = np.array(metrics["cumsum_z"])
    plt.figure()
    plt.plot(it2, cx, label="cumsum x")
    plt.plot(it2, cz, label="cumsum z")    
    plt.xlabel("iteration k")
    plt.ylabel(r"cumsum")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_cumsum_rel_step.png"), dpi=200)
    plt.close()

def randn_like_compat(x: torch.Tensor, seed: int):
    g = torch.Generator(device=x.device).manual_seed(seed)
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g)


def make_random_rect_mask(H: int, W: int, missing_ratio: float = 0.2, seed: int = 0,
                          min_rect: int = 2, max_rect: int = 7) -> torch.Tensor:
    rng = random.Random(seed)
    mask = np.ones((H, W), dtype=np.float32)
    target_missing = int(missing_ratio * H * W)
    missing = 0
    while missing < target_missing:
        rh = rng.randint(min_rect, min(max_rect, H))
        rw = rng.randint(min_rect, min(max_rect, W))
        y0 = rng.randint(0, H - rh)
        x0 = rng.randint(0, W - rw)

        before = mask[y0:y0+rh, x0:x0+rw].sum()
        mask[y0:y0+rh, x0:x0+rw] = 0.0
        after = mask[y0:y0+rh, x0:x0+rw].sum()
        missing += int(before - after)

    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)


# Shepard init
def shepard_initialize_rgb(y01: torch.Tensor, M01: torch.Tensor, window: int = 9, p: float = 2.0) -> torch.Tensor:
    assert window % 2 == 1
    y = y01.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()   # (H,W,3)
    m = M01.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)  # (H,W)
    H, W, _ = y.shape
    out = y.copy()
    wing = window // 2

    offsets, weights = [], []
    for dy in range(-wing, wing + 1):
        for dx in range(-wing, wing + 1):
            if dy == 0 and dx == 0:
                continue
            d = (abs(dy) ** p + abs(dx) ** p)
            if d == 0:
                continue
            offsets.append((dy, dx))
            weights.append(1.0 / d)
    weights = np.array(weights, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            if m[i, j] == 0:
                vals, wts = [], []
                for (dy, dx), w0 in zip(offsets, weights):
                    ii, jj = i + dy, j + dx
                    if 0 <= ii < H and 0 <= jj < W and m[ii, jj] == 1:
                        vals.append(y[ii, jj])
                        wts.append(w0)
                if len(wts) > 0:
                    wts = np.array(wts, dtype=np.float32)
                    wts = wts / (wts.sum() + 1e-12)
                    vals = np.stack(vals, axis=0)
                    out[i, j] = (vals * wts[:, None]).sum(axis=0)

    z0 = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
    return z0.clamp(0, 1)


def get_rho_sigma(sigma_obs=2.55/255, iter_num=15, modelSigma2=2.55):
    modelSigma1 = 49.0
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    sigmas = modelSigmaS / 255.0
    mus = list(map(lambda x: (sigma_obs**2) / (x**2) / 3.0, sigmas))
    rhos = mus
    return np.array(rhos, np.float32), np.array(sigmas, np.float32)

@torch.no_grad()
def denoise_sigma_map(model_name: str, model, x3: torch.Tensor, sigma: float):
    B, C, H, W = x3.shape
    sigma_map = torch.full((B, 1, H, W), float(sigma), device=x3.device, dtype=x3.dtype)
    inp4 = torch.cat([x3, sigma_map], dim=1)
    return ircnn_infer(model, inp4).clamp(0, 1)

# HQS Inpainting
@torch.no_grad()
def hqs_inpaint(
    y: torch.Tensor,
    M: torch.Tensor,
    model_name: str,
    model,
    iter_num: int = 15,
    lambda_pnp: float = 0.23,
    sigma_obs_pix: float = 5.0,
    modelSigma2_pix: float = 2.55,
    shepard_window: int = 21,
    shepard_p: float = 2.0,
    add_small_noise_in_holes: float = 0.01,
    seed: int = 0,
    track_convergence: bool = False,
    gt: torch.Tensor = None,
):
    device = y.device
    M3 = M.repeat(1, 3, 1, 1)

    sigma_obs = float(sigma_obs_pix) / 255.0
    rhos, sigmas = get_rho_sigma(sigma_obs=sigma_obs, iter_num=iter_num, modelSigma2=modelSigma2_pix)
    rhos_t = torch.tensor(rhos, device=device, dtype=y.dtype)

    # init Shepard
    z = shepard_initialize_rgb(y, M, window=shepard_window, p=shepard_p).to(device=device, dtype=y.dtype)
    if add_small_noise_in_holes > 0:
        z = (z + (1.0 - M3) * add_small_noise_in_holes * randn_like_compat(z, seed=seed)).clamp(0, 1)

    # metrics
    metrics = None
    if track_convergence:
        metrics = {
            "psnr_x":   [0.0] * iter_num,
            "psnr_z":   [0.0] * iter_num,
            "ssim_x":   [0.0] * iter_num,
            "ssim_z":   [0.0] * iter_num,
            "rel_step_x": [0.0] * (iter_num-1),
            "rel_step_z": [0.0] * (iter_num-1),
            "cumsum_x":   [0.0] * (iter_num-1),
            "cumsum_z":   [0.0] * (iter_num-1),
        }
        x_prev = None
        x0_norm = None
        z0_norm = None
        csum_x = 0.0
        csum_z = 0.0

    for k in range(iter_num):
        sigma_k = sigmas[k]
        mu = lambda_pnp/ (sigma_k**2)
        xk = (M3 * y + mu * z) / (M3 + mu)
        xk = xk.clamp(0, 1)

        # PSNR and SSIM
        if track_convergence and (gt is not None):
            metrics["psnr_x"][k] = psnr_torch(xk, gt)
            metrics["ssim_x"][k] = ssim_torch(xk, gt)

        # z-step
        sigma_k = float(sigmas[k])
        z = denoise_sigma_map(model_name, model, xk, sigma=sigma_k)

        # enforce known pixels
        z = (M3 * y + (1.0 - M3) * z).clamp(0, 1)
        
        if track_convergence and (gt is not None):
            metrics["psnr_z"][k] = psnr_torch(z, gt)
            metrics["ssim_z"][k] = ssim_torch(z, gt)

            if k == 0:
                x_prev = xk.clone()
                z_prev = z.clone()
                x0_norm = float(l2norm_flat(x_prev).item()) + 1e-12
                z0_norm = float(l2norm_flat(z_prev).item()) + 1e-12
            else:
                step_x = float(l2norm_flat(xk - x_prev).item()) / x0_norm
                step_z = float(l2norm_flat(z  - z_prev).item()) / z0_norm

                metrics["rel_step_x"][k - 1] = step_x
                metrics["rel_step_z"][k - 1] = step_z

                csum_x += step_x
                csum_z += step_z
                metrics["cumsum_x"][k - 1] = csum_x
                metrics["cumsum_z"][k - 1] = csum_z

                x_prev = xk.clone()
                z_prev = z.clone()

    return (z, metrics) if track_convergence else z


def run_compare(
    clean_path,
    ircnn_ckpt="./weights_ircnn_sigmap/ircnn_sigmap_final.pth",
    out_dir="./results_IRCNN_sigmamap/inpainting_single_v2",
    missing_ratio=0.45, 
    seed=0,
    iter_num=20,
    lambda_pnp=0.23,
    sigma_obs_pix=5.0, 
    modelSigma2_pix=2.55,
    shepard_window=11, 
    shepard_p=2.0
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt = TF.to_tensor(Image.open(clean_path).convert("RGB")).unsqueeze(0).to(device).clamp(0, 1)
    _, _, H, W = gt.shape

    M = make_random_rect_mask(H, W, missing_ratio=missing_ratio, seed=seed).to(device=device, dtype=gt.dtype)
    M3 = M.repeat(1, 3, 1, 1)
    y = (gt * M3).clamp(0, 1)

    sigma_obs = float(sigma_obs_pix) / 255.0
    if sigma_obs > 0:
        y = (M3 * (y + randn_like_compat(y, seed=seed+123) * sigma_obs)).clamp(0, 1)

    save_img01(gt,  os.path.join(out_dir, "clean.png"))
    save_img01(M3, os.path.join(out_dir, "mask.png"))
    save_img01(y,  os.path.join(out_dir, "masked_noisy.png"))

    rec_sh = shepard_initialize_rgb(y, M, window=shepard_window, p=shepard_p).to(device=device, dtype=gt.dtype)
    rec_sh = (M3 * y + (1.0 - M3) * rec_sh).clamp(0, 1)
    
    # Load IRCNN+
    ircnn = IRCNNSigmaMap(features=64).to(device).eval()
    st = torch.load(ircnn_ckpt, map_location=device)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    ircnn.load_state_dict(sd, strict=True)

    # IRCNN+
    rec_d, metrics_d = hqs_inpaint(
        y=y, M=M, model_name="ircnn", model=ircnn,
        iter_num=iter_num, lambda_pnp=lambda_pnp, sigma_obs_pix=sigma_obs_pix, modelSigma2_pix=modelSigma2_pix,
        shepard_window=shepard_window, shepard_p=shepard_p, seed=seed,
        track_convergence=True, gt=gt
    )

    save_img01(rec_d, os.path.join(out_dir, "restored_hqs_ircnn.png"))
    save_convergence_plots(metrics_d, out_dir=out_dir, prefix="ircnn")

    miss = (1.0 - M3)

    def psnr_on_missing(a, b, missmask, eps=1e-12):
        num = missmask.sum().item()
        mse = (((a - b) * missmask) ** 2).sum().item() / (num + eps)
        return 10.0 * math.log10(1.0 / (mse + eps))

    print("Saved to:", out_dir)
    print("PSNR global:")
    print("  input        :", f"{psnr_torch(y, gt):.2f} dB")
    print("  shepard-only :", f"{psnr_torch(rec_sh, gt):.2f} dB")
    print("  IRCNN+:", f"{psnr_torch(rec_d, gt):.2f} dB")
    print("SSIM global:")
    print("  input        :", f"{ssim_torch(y, gt):.2f}")
    print("  shepard-only :", f"{ssim_torch(rec_sh, gt):.2f}")
    print("  IRCNN+:", f"{ssim_torch(rec_d, gt):.2f}")
    
    print("PSNR missing:")
    print("  input        :", f"{psnr_on_missing(y,      gt, miss):.2f} dB")
    print("  shepard-only :", f"{psnr_on_missing(rec_sh, gt, miss):.2f} dB")
    print("  IRCNN+:", f"{psnr_on_missing(rec_d,  gt, miss):.2f} dB")
    
    print("Convergence plots saved:")
    print(" ", os.path.join(out_dir, "ircnn_psnr_xk.png"))
    print(" ", os.path.join(out_dir, "ircnn_ssim_xk.png"))
    print(" ", os.path.join(out_dir, "ircnn_rel_step_log.png"))
    print(" ", os.path.join(out_dir, "ircnn_cumsum_rel_step.png"))
    

#castel
'''clean_path = "./BSDS300/images/test/37073.jpg"
print("Image choisie:", clean_path)
run_compare(
    clean_path = clean_path,
    ircnn_ckpt="./weights_ircnn_sigmap/ircnn_sigmap_final.pth",
    out_dir="./results_IRCNN_sigmamap/inpainting_single/plane",
    missing_ratio=0.33, 
    seed=0,
    iter_num=20,
    lambda_pnp=3,
    sigma_obs_pix=0,
    shepard_window=11
)'''


@torch.no_grad()
def run_compare_return_metrics(
    clean_path,
    out_dir,
    ircnn,
    missing_ratio=0.15,
    seed=0,
    iter_num=15,
    lambda_pnp=0.23,
    sigma_obs_pix=5.0,
    modelSigma2_pix=2.55,
    shepard_window=21,
    shepard_p=2.0,
    save_outputs=True,
    save_convergence=True
):
    os.makedirs(out_dir, exist_ok=True)
    device = next(ircnn.parameters()).device

    gt = TF.to_tensor(Image.open(clean_path).convert("RGB")).unsqueeze(0).to(device).clamp(0, 1)
    _, _, H, W = gt.shape

    M = make_random_rect_mask(H, W, missing_ratio=missing_ratio, seed=seed).to(device=device, dtype=gt.dtype)
    M3 = M.repeat(1, 3, 1, 1)
    y = (gt * M3).clamp(0, 1)

    # noise observation
    sigma_obs = float(sigma_obs_pix) / 255.0
    if sigma_obs > 0:
        y = (M3 * (y + randn_like_compat(y, seed=seed+123) * sigma_obs)).clamp(0, 1)

    # Shepard init
    rec_sh = shepard_initialize_rgb(y, M, window=int(shepard_window), p=shepard_p).to(device=device, dtype=gt.dtype)
    rec_sh = (M3 * y + (1.0 - M3) * rec_sh).clamp(0, 1)

    # HQS
    t0 = time.perf_counter()
    out = hqs_inpaint(
    y=y, M=M, model_name="ircnn", model=ircnn,
    iter_num=iter_num, lambda_pnp=lambda_pnp, sigma_obs_pix=sigma_obs_pix, modelSigma2_pix=modelSigma2_pix,
    shepard_window=int(shepard_window), shepard_p=shepard_p, seed=seed,
    track_convergence=save_convergence, gt=gt
    )

    if save_convergence:
        rec_d, metrics_d = out
    else:
        rec_d = out
        metrics_d = None

    t_d = time.perf_counter() - t0

    t0 = time.perf_counter()

    miss = (1.0 - M3)

    def psnr_on_missing(a, b, missmask, eps=1e-12):
        num = missmask.sum().item()
        mse = (((a - b) * missmask) ** 2).sum().item() / (num + eps)
        return 10.0 * math.log10(1.0 / (mse + eps))

    res = {
        "image": os.path.basename(clean_path),
        "H": H, "W": W,
        "missing_ratio": float(missing_ratio),
        "iter_num": int(iter_num),
        "sigma_obs_pix": float(sigma_obs_pix),
        "shepard_window": int(shepard_window),
        "psnr_input": psnr_torch(y, gt),
        "psnr_shepard": psnr_torch(rec_sh, gt),
        "psnr_ircnn": psnr_torch(rec_d, gt),
        "ssim_input": ssim_torch(y, gt),
        "ssim_shepard": ssim_torch(rec_sh, gt),
        "ssim_ircnn": ssim_torch(rec_d, gt),
        "psnr_miss_input": psnr_on_missing(y, gt, miss),
        "psnr_miss_shepard": psnr_on_missing(rec_sh, gt, miss),
        "psnr_miss_ircnn": psnr_on_missing(rec_d, gt, miss),
        "time_ircnn_s": t_d,
    }

    if save_outputs:
        save_img01(gt,  os.path.join(out_dir, "clean.png"))
        save_img01(M3, os.path.join(out_dir, "mask.png"))
        save_img01(y,  os.path.join(out_dir, "masked_noisy.png"))
        save_img01(rec_sh, os.path.join(out_dir, "restored_shepard_only.png"))
        save_img01(rec_d,  os.path.join(out_dir, "restored_hqs_ircnnt.png"))

        if save_convergence and (metrics_d is not None):
            save_convergence_plots(metrics_d, out_dir=out_dir, prefix="ircnn")

    return res


IMG_EXT = (".png", ".jpg", ".jpeg")

def list_images_in_dir(root: str):
    paths = []
    for ext in IMG_EXT:
        paths += glob(os.path.join(root, f"**/*{ext}"), recursive=True)
        paths += glob(os.path.join(root, f"**/*{ext.upper()}"), recursive=True)
    return sorted(list(set(paths)))

def pick_n_images(paths, n=10, seed=0):
    rng = random.Random(seed)
    if n >= len(paths):
        return paths
    return rng.sample(paths, n)

def load_models_once(device, ircnn_ckpt):
    model = IRCNNSigmaMap().to(device).eval()
    state = torch.load(ircnn_ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    return model


def run_pool_10_images(
    clean_dir,
    out_root,
    ircnn_ckpt,
    n_images=10,
    seed=0,
    missing_ratio=0.45,
    iter_num=20,
    lambda_pnp=0.23,
    sigma_obs_pix=5.0,
    modelSigma2_pix=2.55,
    shepard_window=21,
    shepard_p=2.0,
    save_outputs_per_image=False,
    save_convergence=False,
    csv_name="results_pool.csv"
):
    os.makedirs(out_root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = list_images_in_dir(clean_dir)
    chosen = pick_n_images(paths, n=n_images, seed=seed)

    ircnn = load_models_once(device, ircnn_ckpt)

    rows = []
    for p in chosen:
        name = os.path.splitext(os.path.basename(p))[0]
        out_dir = os.path.join(out_root, name) if save_outputs_per_image else out_root

        r = run_compare_return_metrics(
            clean_path=p,
            out_dir=out_dir,
            ircnn=ircnn,
            missing_ratio=missing_ratio,
            seed=seed,
            iter_num=iter_num,
            lambda_pnp=lambda_pnp,
            sigma_obs_pix=sigma_obs_pix,
            modelSigma2_pix=modelSigma2_pix,
            shepard_window=shepard_window,
            shepard_p=shepard_p,
            save_outputs=save_outputs_per_image,
            save_convergence=save_convergence
        )
        rows.append(r)
        print(r)
    df = pd.DataFrame(rows)

    metric_cols = [c for c in df.columns if c.startswith("psnr") or c.startswith("time_") or c.startswith("ssim")]
    mean_row = {"image": "MEAN"}
    std_row  = {"image": "STD"}
    for c in metric_cols:
        mean_row[c] = float(df[c].mean())
        std_row[c]  = float(df[c].std())
    df2 = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    csv_path = os.path.join(out_root, csv_name)
    df2.to_csv(csv_path, index=False)
    
    return df2
   
    
'''df = run_pool_10_images(
        clean_dir="./BSDS300/images/images_benchmark/benchmark_10_images",
        out_root="./results_IRCNN_sigmamap/inpainting_benchmark",
        ircnn_ckpt="./weights_ircnn_sigmap/ircnn_sigmap_final.pth",
        n_images=10,
        seed=0,
        missing_ratio=0.2,
        iter_num=20,
        lambda_pnp=0.23,
        sigma_obs_pix=2.5,
        modelSigma2_pix=2.55,
        shepard_window=11,
        shepard_p=2.0,
        save_outputs_per_image=False,
        save_convergence=False,
        csv_name="pool10_metrics.csv"
    )'''