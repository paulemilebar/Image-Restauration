import os, math, random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from DRUNet.DRUNet import DRUNetSigmaMap
from IRCNN.IRCNN import IRCNNSigmaMap
import matplotlib.pyplot as plt



# =========================
# Utils
# =========================

def l2norm_flat(x: torch.Tensor) -> torch.Tensor:
    return torch.norm(x.reshape(-1), p=2)

def save_convergence_plots(metrics: dict, out_dir: str, prefix: str = "drunet"):
    """
    metrics:
      psnr_x:   list length K
      rel_step: list length K (rel_step[k] = ||x_{k+1}-x_k|| / ||x0|| for k<=K-2, rel_step[K-1]=0)
      cumsum:   list length K (cumsum[k] = sum_{i<=k} rel_step[i])
    """
    os.makedirs(out_dir, exist_ok=True)
    K = len(metrics["psnr_x"])
    it = np.arange(K)

    # --- 1) PSNR(x_k)
    plt.figure()
    plt.plot(it, metrics["psnr_x"])
    plt.xlabel("itération k")
    plt.ylabel("PSNR(x_k, GT) [dB]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_psnr_xk.png"), dpi=200)
    plt.close()

    # --- 2) ||x_{k+1}-x_k|| / ||x0|| (log scale)
    rel = np.array(metrics["rel_step"], dtype=np.float64)
    rel_safe = np.maximum(rel, 1e-16)  # évite log(0)
    plt.figure()
    plt.semilogy(it, rel_safe)
    plt.xlabel("itération k")
    plt.ylabel(r"relsafe")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_rel_step_log.png"), dpi=200)
    plt.close()

    # --- 3) somme cumulée
    plt.figure()
    plt.plot(it, metrics["cumsum"])
    plt.xlabel("itération k")
    plt.ylabel(r"cumsum")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_cumsum_rel_step.png"), dpi=200)
    plt.close()

def psnr_torch(x, y, eps=1e-12):
    mse = torch.mean((x - y) ** 2).item()
    return 10.0 * math.log10(1.0 / (mse + eps))

def save_img01(t: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    TF.to_pil_image(t.squeeze(0).clamp(0, 1).cpu()).save(path)

def randn_like_compat(x: torch.Tensor, seed: int):
    try:
        g = torch.Generator(device=x.device).manual_seed(seed)
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g)
    except TypeError:
        torch.manual_seed(seed)
        return torch.randn_like(x)

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

    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


# =========================
# Shepard init (RGB) (ta version)
# =========================
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


# =========================
# Denoisers (sigma-map)
# =========================
@torch.no_grad()
def drunet_infer(model, inp, modulo: int = 8):
    b, c, h, w = inp.shape
    pad_h = (modulo - h % modulo) % modulo
    pad_w = (modulo - w % modulo) % modulo
    if pad_h or pad_w:
        inp2 = F.pad(inp, (0, pad_w, 0, pad_h), mode="reflect")
        out = model(inp2)
        return out[..., :h, :w]
    return model(inp)

@torch.no_grad()
def denoise_sigma_map(model_name: str, model, x3: torch.Tensor, sigma: float):
    # sigma: normalisé [0,1]
    B, C, H, W = x3.shape
    sigma_map = torch.full((B, 1, H, W), float(sigma), device=x3.device, dtype=x3.dtype)
    inp4 = torch.cat([x3, sigma_map], dim=1)
    if model_name == "drunet":
        return drunet_infer(model, inp4, modulo=8).clamp(0, 1)
    elif model_name == "ircnn":
        return model(inp4).clamp(0, 1)
    else:
        raise ValueError("model_name must be 'drunet' or 'ircnn'")


# =========================
# Schedule DPIR utils_inpaint (ton snippet)
# =========================
def get_rho_sigma_dpir(sigma_obs=2.55/255, iter_num=15, modelSigma2=2.55):
    modelSigma1 = 49.0
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    sigmas = modelSigmaS / 255.0
    mus = list(map(lambda x: (sigma_obs**2) / (x**2) / 3.0, sigmas))  # /3 pour RGB
    rhos = mus
    return np.array(rhos, np.float32), np.array(sigmas, np.float32)


# =========================
# DPIR-style PnP-HQS Inpainting
# =========================
@torch.no_grad()
def dpir_hqs_inpaint(
    y: torch.Tensor,     # (1,3,H,W) masked observation (trous=0)
    M: torch.Tensor,     # (1,1,H,W) 1 connu / 0 manquant
    model_name: str,
    model,
    iter_num: int = 15,
    sigma_obs_pix: float = 5.0,
    modelSigma2_pix: float = 2.55,
    shepard_window: int = 21,
    shepard_p: float = 2.0,
    add_small_noise_in_holes: float = 0.01,
    seed: int = 0,

    # --- NEW ---
    track_convergence: bool = False,
    gt: torch.Tensor = None,   # (1,3,H,W) ground truth dans [0,1]
):
    device = y.device
    M3 = M.repeat(1, 3, 1, 1)

    # schedule (DPIR)
    sigma_obs = float(sigma_obs_pix) / 255.0
    rhos, sigmas = get_rho_sigma_dpir(sigma_obs=sigma_obs, iter_num=iter_num, modelSigma2=modelSigma2_pix)
    rhos_t = torch.tensor(rhos, device=device, dtype=y.dtype)

    # init Shepard
    z = shepard_initialize_rgb(y, M, window=shepard_window, p=shepard_p).to(device=device, dtype=y.dtype)
    if add_small_noise_in_holes > 0:
        z = (z + (1.0 - M3) * add_small_noise_in_holes * randn_like_compat(z, seed=seed)).clamp(0, 1)

    # --- NEW: buffers convergence ---
    metrics = None
    if track_convergence:
        metrics = {
            "psnr_x":   [0.0] * iter_num,
            "rel_step": [0.0] * iter_num,
            "cumsum":   [0.0] * iter_num,
        }
        x_prev = None
        x0_norm = None
        csum = 0.0

    for k in range(iter_num):
        mu = rhos_t[k].view(1, 1, 1, 1)

        # x-step (fermé pixelwise)
        xk = (M3 * y + mu * z) / (M3 + mu)
        xk = xk.clamp(0, 1)

        # --- NEW: PSNR(x_k)
        if track_convergence and (gt is not None):
            metrics["psnr_x"][k] = psnr_torch(xk, gt)

        # z-step
        sigma_k = float(sigmas[k])  # déjà normalisé [0,1]
        z = denoise_sigma_map(model_name, model, xk, sigma=sigma_k)

        # hard enforce known pixels
        z = (M3 * y + (1.0 - M3) * z).clamp(0, 1)

        # --- NEW: ||x_{k+1}-x_k|| / ||x0|| + somme cumulée
        if track_convergence:
            if k == 0:
                x_prev = xk.clone()
                x0_norm = float(l2norm_flat(x_prev).item()) + 1e-12
            else:
                # ceci correspond à rel_step[k-1] = ||x_k - x_{k-1}|| / ||x0||
                step = float(l2norm_flat(xk - x_prev).item()) / x0_norm
                metrics["rel_step"][k - 1] = step
                csum += step
                metrics["cumsum"][k - 1] = csum
                x_prev = xk.clone()

            # on remplit "au fil de l'eau" : pour k=0, rel_step[0] sera rempli au k=1 etc.

    # termine les derniers points (rel_step[K-1]=0, cumsum[K-1]=csum)
    if track_convergence:
        metrics["rel_step"][-1] = 0.0
        metrics["cumsum"][-1] = csum

    return (z, metrics) if track_convergence else z



# =========================
# Main compare
# =========================
def run_compare(clean_path, out_dir, drunet_ckpt, ircnn_ckpt,
                missing_ratio=0.15, seed=0,
                iter_num=15, sigma_obs_pix=5.0, modelSigma2_pix=2.55,
                shepard_window=21, shepard_p=2.0):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

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
    save_img01(rec_sh, os.path.join(out_dir, "restored_shepard_only.png"))

    # Load DRUNet
    drunet = DRUNetSigmaMap(in_nc=4, out_nc=3, nc=(64,128,256,512), nb=4).to(device).eval()
    st = torch.load(drunet_ckpt, map_location=device)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    drunet.load_state_dict(sd, strict=True)

    # Load IRCNN
    ircnn = IRCNNSigmaMap(features=64).to(device).eval()
    st = torch.load(ircnn_ckpt, map_location=device)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    ircnn.load_state_dict(sd, strict=True)

    # --- DRUNet + tracking
    rec_d, metrics_d = dpir_hqs_inpaint(
        y=y, M=M, model_name="drunet", model=drunet,
        iter_num=iter_num, sigma_obs_pix=sigma_obs_pix, modelSigma2_pix=modelSigma2_pix,
        shepard_window=shepard_window, shepard_p=shepard_p, seed=seed,
        track_convergence=True, gt=gt
    )

    # --- IRCNN sans tracking (ou mets track_convergence=True si tu veux aussi)
    rec_i = dpir_hqs_inpaint(
        y=y, M=M, model_name="ircnn", model=ircnn,
        iter_num=iter_num, sigma_obs_pix=sigma_obs_pix, modelSigma2_pix=modelSigma2_pix,
        shepard_window=shepard_window, shepard_p=shepard_p, seed=seed
    )

    save_img01(rec_d, os.path.join(out_dir, "restored_dpir_hqs_drunet.png"))
    save_img01(rec_i, os.path.join(out_dir, "restored_dpir_hqs_ircnn.png"))

    # --- NEW: save plots dans le même dossier out_dir
    save_convergence_plots(metrics_d, out_dir=out_dir, prefix="drunet")

    # Metrics finaux
    miss = (1.0 - M3)

    def psnr_on_missing(a, b, missmask, eps=1e-12):
        num = missmask.sum().item()
        mse = (((a - b) * missmask) ** 2).sum().item() / (num + eps)
        return 10.0 * math.log10(1.0 / (mse + eps))

    print("Saved to:", out_dir)
    print("PSNR global:")
    print("  input        :", f"{psnr_torch(y, gt):.2f} dB")
    print("  shepard-only :", f"{psnr_torch(rec_sh, gt):.2f} dB")
    print("  DPIR+DRUNet  :", f"{psnr_torch(rec_d, gt):.2f} dB")
    print("  DPIR+IRCNN   :", f"{psnr_torch(rec_i, gt):.2f} dB")
    print("PSNR missing:")
    print("  input        :", f"{psnr_on_missing(y,      gt, miss):.2f} dB")
    print("  shepard-only :", f"{psnr_on_missing(rec_sh, gt, miss):.2f} dB")
    print("  DPIR+DRUNet  :", f"{psnr_on_missing(rec_d,  gt, miss):.2f} dB")
    print("  DPIR+IRCNN   :", f"{psnr_on_missing(rec_i,  gt, miss):.2f} dB")
    print("Convergence plots saved:")
    print(" ", os.path.join(out_dir, "drunet_psnr_xk.png"))
    print(" ", os.path.join(out_dir, "drunet_rel_step_log.png"))
    print(" ", os.path.join(out_dir, "drunet_cumsum_rel_step.png"))

if __name__ == "__main__":
    run_compare(
        clean_path=r"./BSDS300/images/test/102061.jpg",
        out_dir="out_dpir_like_inpaint",
        drunet_ckpt=r"./weights_drunet_sigmap/drunet_sigmap_final.pth",
        ircnn_ckpt=r"./weights_ircnn_sigmap/ircnn_sigmap_final.pth",
        missing_ratio=0.4,
        seed=0,
        iter_num=20,
        sigma_obs_pix=5.0,
        modelSigma2_pix=2.55,   
        shepard_window=9,
        shepard_p=2.0,
    )
