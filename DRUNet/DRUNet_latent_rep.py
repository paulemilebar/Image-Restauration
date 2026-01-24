# analyze_drunet_latent_37073_diverse.py
import os, math, random
from glob import glob

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

def list_images(root):
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                paths.append(os.path.join(r, f))
    return sorted(paths)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def to_tensor01(pil):
    return TF.to_tensor(pil)  # (3,H,W) in [0,1]

def save_img01(t01_chw, path):
    t01_chw = t01_chw.detach().cpu().clamp(0, 1)
    TF.to_pil_image(t01_chw).save(path)

def save_gray01(gray_hw, path):
    arr = (gray_hw.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)

def norm01(x, eps=1e-12):
    x = x - x.min()
    return x / (x.max() + eps)

def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Noise + input builder
def add_awgn(clean_01, sigma_pixels, generator=None):
    """
    clean_01: (1,3,H,W) in [0,1]
    sigma_pixels: float
    generator: torch.Generator optionnel (CPU uniquement)
    """
    sigma = float(sigma_pixels) / 255.0

    if clean_01.device.type == "cuda":
        noise = torch.randn_like(clean_01) * sigma
    else:
        if generator is None:
            noise = torch.randn(clean_01.shape, device=clean_01.device, dtype=clean_01.dtype) * sigma
        else:
            noise = torch.randn(
                clean_01.shape,
                device=clean_01.device,
                dtype=clean_01.dtype,
                generator=generator
            ) * sigma
    return clean_01 + noise

def build_inp(noisy, sigma_pixels):
    """
    noisy: (1,3,H,W)
    returns inp: (1,4,H,W) = noisy_rgb + sigma_map
    """
    B, C, H, W = noisy.shape
    sigma_map = torch.full((B, 1, H, W), float(sigma_pixels) / 255.0, device=noisy.device, dtype=noisy.dtype)
    return torch.cat([noisy, sigma_map], dim=1)


# Forward with latents
@torch.no_grad()
def forward_collect(model, inp):
    """
    Retourne out + dict d'activations
    Clés:
      head, x1, x2, x3, x4 (bottleneck),
      y3, y2, y1, out
    """
    acts = {}

    x, pads = _pad_to_multiple(inp, mult=8)
    acts["inp_padded"] = x

    h = model.head(x)
    acts["head"] = h

    x1 = model.e1(h); acts["x1"] = x1
    x2 = model.e2(model.d1(x1)); acts["x2"] = x2
    x3 = model.e3(model.d2(x2)); acts["x3"] = x3
    x4 = model.mid(model.d3(x3)); acts["x4"] = x4  # bottleneck

    y3 = model.u3(x4)
    y3 = model.p3(model.f3(torch.cat([y3, x3], dim=1)))
    acts["y3"] = y3

    y2 = model.u2(y3)
    y2 = model.p2(model.f2(torch.cat([y2, x2], dim=1)))
    acts["y2"] = y2

    y1 = model.u1(y2)
    y1 = model.p1(model.f1(torch.cat([y1, x1], dim=1)))
    acts["y1"] = y1

    out = model.tail(y1)
    out = _unpad(out, pads)
    acts["out"] = out

    return out, acts


# Pad helpers
def _pad_to_multiple(x: torch.Tensor, mult: int = 8):
    _, _, h, w = x.shape
    pad_h = (mult - h % mult) % mult
    pad_w = (mult - w % mult) % mult
    pt = pad_h // 2
    pb = pad_h - pt
    pl = pad_w // 2
    pr = pad_w - pl
    if pad_h or pad_w:
        x = F.pad(x, (pl, pr, pt, pb), mode="reflect")
    return x, (pl, pr, pt, pb)

def _unpad(x: torch.Tensor, pads):
    pl, pr, pt, pb = pads
    if (pl, pr, pt, pb) == (0, 0, 0, 0):
        return x
    return x[:, :, pt:x.shape[2]-pb, pl:x.shape[3]-pr]


# Filter visualizations
def plot_head_filters(model, out_path_png):
    """
    head.weight: (64,4,3,3)
    - grille RGB (canaux 0..2)
    - grille sigma (canal 3)
    """
    W = model.head.weight.detach().cpu()  # (64,4,3,3)
    oc, ic, kh, kw = W.shape
    assert ic == 4 and kh == 3 and kw == 3

    W_rgb = W[:, :3, :, :]  # (oc,3,3,3)
    wr = W_rgb.reshape(oc, -1)
    wr = (wr - wr.mean(dim=1, keepdim=True)) / (wr.std(dim=1, keepdim=True) + 1e-8)
    W_rgbn = wr.reshape(oc, 3, 3, 3)

    W_s = W[:, 3, :, :]     # (oc,3,3)
    ws = W_s.reshape(oc, -1)
    ws = (ws - ws.mean(dim=1, keepdim=True)) / (ws.std(dim=1, keepdim=True) + 1e-8)
    W_sn = ws.reshape(oc, 3, 3)

    n = oc
    grid = int(math.ceil(math.sqrt(n)))
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Head filters (RGB channels 0..2)")
    canvas = np.ones((grid*3, grid*3, 3), dtype=np.float32) * 0.5

    k = 0
    for i in range(grid):
        for j in range(grid):
            if k >= n:
                break
            ker = W_rgbn[k].permute(1, 2, 0).numpy()  # (3,3,3)
            ker = (ker - ker.min()) / (ker.max() - ker.min() + 1e-8)
            canvas[i*3:(i+1)*3, j*3:(j+1)*3, :] = ker
            k += 1

    ax1.imshow(canvas)
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Head filters (sigma_map channel 3)")
    canvas2 = np.ones((grid*3, grid*3), dtype=np.float32) * 0.5

    k = 0
    for i in range(grid):
        for j in range(grid):
            if k >= n:
                break
            ker = W_sn[k].numpy()
            ker = (ker - ker.min()) / (ker.max() - ker.min() + 1e-8)
            canvas2[i*3:(i+1)*3, j*3:(j+1)*3] = ker
            k += 1

    ax2.imshow(canvas2, cmap="gray")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(out_path_png, dpi=200)
    plt.close(fig)

def plot_weight_stats(model, out_path_png):
    convs = {
        "head": model.head,
        "d1": model.d1.conv,
        "d2": model.d2.conv,
        "d3": model.d3.conv,
        "u1": model.u1.tconv,
        "u2": model.u2.tconv,
        "u3": model.u3.tconv,
        "f1": model.f1,
        "f2": model.f2,
        "f3": model.f3,
        "tail": model.tail,
    }

    fig = plt.figure(figsize=(14, 10))
    idx = 1
    for name, m in convs.items():
        W = m.weight.detach().cpu().numpy().ravel()
        ax = fig.add_subplot(4, 3, idx)
        ax.hist(W, bins=60)
        ax.set_title(f"{name}: mean={W.mean():.2e}, std={W.std():.2e}")
        idx += 1
        if idx > 12:
            break

    plt.tight_layout()
    plt.savefig(out_path_png, dpi=200)
    plt.close(fig)


# Latent visualizations
def feature_energy_map(feat_bchw):
    """
    feat: (1,C,H,W)
    returns (H,W) in [0,1] representing sqrt(sum_c feat^2)
    """
    f = feat_bchw[0]
    e = torch.sqrt(torch.sum(f*f, dim=0) + 1e-12)  # (H,W)
    return norm01(e)

def _corrcoef_1d(a: torch.Tensor, b: torch.Tensor, eps=1e-12) -> float:
    """
    Corrélation de Pearson entre deux vecteurs 1D torch (CPU).
    Retourne float.
    """
    a = a - a.mean()
    b = b - b.mean()
    num = torch.sum(a * b)
    den = torch.sqrt(torch.sum(a*a) * torch.sum(b*b) + eps)
    return float((num / den).item())

def topk_channel_grid_diverse(
    feat_bchw,
    out_png,
    k=16,
    title="",
    corr_thr=0.90,
    candidates_mul=8,
):
    """
    Sélectionne des canaux "top variance" mais en imposant de la diversité :
    - on trie les canaux par variance décroissante
    - on prend greedy ceux dont |corr| <= corr_thr avec tous les déjà pris
    (corr calculée sur la carte aplatie)

    candidates_mul: on regarde les (k*candidates_mul) meilleurs par variance pour trouver k diversifiés.
    """
    f = feat_bchw[0].detach().cpu()  # (C,H,W)
    C, H, W = f.shape

    flat = f.view(C, -1)  # (C, HW)
    var = flat.var(dim=1)  # (C,)
    # indices triés par variance décroissante
    sorted_idx = torch.argsort(var, descending=True).tolist()

    max_cand = min(len(sorted_idx), k * candidates_mul)
    cand = sorted_idx[:max_cand]

    picked = []
    picked_flat = []

    for c in cand:
        v = flat[c]
        ok = True
        for pv in picked_flat:
            cc = _corrcoef_1d(v, pv)
            if abs(cc) > corr_thr:
                ok = False
                break
        if ok:
            picked.append(c)
            picked_flat.append(v)
        if len(picked) >= min(k, C):
            break

    # si pas assez, on complète sans contrainte (fallback)
    if len(picked) < min(k, C):
        for c in sorted_idx:
            if c not in picked:
                picked.append(c)
            if len(picked) >= min(k, C):
                break

    grid = int(math.ceil(math.sqrt(len(picked))))
    fig = plt.figure(figsize=(10, 10))
    for t, c in enumerate(picked):
        ax = fig.add_subplot(grid, grid, t + 1)
        img = f[c]
        img = norm01(img)  # visualisation min-max par canal
        ax.imshow(img.numpy(), cmap="gray")
        ax.set_title(f"ch {c}")
        ax.axis("off")

    fig.suptitle(title + f"\n(diverse top-var, corr_thr={corr_thr})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

def save_latent_pack(acts, out_dir, prefix, corr_thr=0.90):
    """
    Sauve energy maps + top channels (diversifiés) pour plusieurs niveaux
    """
    keys = ["head", "x1", "x2", "x3", "x4", "y3", "y2", "y1"]
    for k in keys:
        if k not in acts:
            continue
        e = feature_energy_map(acts[k])
        save_gray01(e, os.path.join(out_dir, f"{prefix}_{k}_energy.png"))
        topk_channel_grid_diverse(
            acts[k],
            os.path.join(out_dir, f"{prefix}_{k}_topch.png"),
            k=16,
            title=f"{prefix} | {k} top variance channels",
            corr_thr=corr_thr,
        )


# PCA embedding (bottleneck)
def bottleneck_embedding(acts):
    x4 = acts["x4"]  # (1,C,H,W)
    emb = F.adaptive_avg_pool2d(x4, 1).view(-1)  # (C,)
    return emb.detach().cpu().numpy()

def plot_pca(embeddings, sigmas, out_png, title):
    X = np.stack(embeddings, axis=0)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=sigmas, s=18)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("sigma (pixels)")
    ax.set_title(title + f" | explained var: {pca.explained_variance_ratio_.sum():.2f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def main(
    clean_dir=r"./BSDS300/images/test",
    exemplar_name="37073.jpg",
    ckpt_path=r"./weights_drunet_sigmap/drunet_sigmap_final.pth",
    out_dir=r"./results_DRUNET_latent_analysis_37073",
    sigmas=(0.0, 5.0, 15.0, 25.0, 50.0, 70.0),
    seed=1,
    device=None,
    corr_thr=0.90,
    also_run_pca=False,
    n_images_for_pca=25,
):
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "filters"))
    ensure_dir(os.path.join(out_dir, "latents"))
    ensure_dir(os.path.join(out_dir, "pca"))

    seed_all(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ---- load model
    from DRUNet import DRUNetSigmaMap
    model = DRUNetSigmaMap(in_nc=4, out_nc=3, nc=(64, 128, 256, 512), nb=4).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    # ---- filters
    plot_head_filters(model, os.path.join(out_dir, "filters", "head_filters.png"))
    plot_weight_stats(model, os.path.join(out_dir, "filters", "weight_hists.png"))
    print("[OK] saved filter figures")

    # ---- locate exemplar image
    paths = list_images(clean_dir)
    if not paths:
        raise ValueError(f"No images in {clean_dir}")

    exemplar_path = None
    for p in paths:
        if os.path.basename(p) == exemplar_name:
            exemplar_path = p
            break
    if exemplar_path is None:
        raise FileNotFoundError(f"Could not find {exemplar_name} inside {clean_dir}")

    # ---- load exemplar
    pil = Image.open(exemplar_path).convert("RGB")
    clean = to_tensor01(pil).unsqueeze(0).to(device)  # (1,3,H,W)

    # ---- CPU generator for reproducibility (CPU only)
    g = torch.Generator().manual_seed(seed)

    # ---- run for each sigma and save full latent pack
    save_img01(clean[0].detach().cpu(), os.path.join(out_dir, "latents", f"{exemplar_name}_clean.png"))

    for s in sigmas:
        noisy = add_awgn(clean, s, generator=g)
        inp = build_inp(noisy, s)
        out, acts = forward_collect(model, inp)

        # save I/O
        save_img01(noisy[0].detach().cpu().clamp(0, 1), os.path.join(out_dir, "latents", f"{exemplar_name}_noisy_sigma{int(s)}.png"))
        save_img01(out[0].detach().cpu().clamp(0, 1), os.path.join(out_dir, "latents", f"{exemplar_name}_out_sigma{int(s)}.png"))

        # save latents
        prefix = f"{os.path.splitext(exemplar_name)[0]}_sigma{int(s)}"
        save_latent_pack(acts, os.path.join(out_dir, "latents"), prefix=prefix, corr_thr=corr_thr)

        print(f"[OK] saved latent pack for sigma={s}")

    # ---- optional PCA on multiple images x sigmas
    if also_run_pca:
        all_paths = paths.copy()
        random.shuffle(all_paths)
        chosen = all_paths[:min(n_images_for_pca, len(all_paths))]
        if exemplar_path not in chosen:
            chosen[0] = exemplar_path

        embeddings, sig_list = [], []
        for p in chosen:
            pil = Image.open(p).convert("RGB")
            clean_i = to_tensor01(pil).unsqueeze(0).to(device)
            for s in sigmas:
                noisy = add_awgn(clean_i, s, generator=g)
                inp = build_inp(noisy, s)
                _, acts = forward_collect(model, inp)
                embeddings.append(bottleneck_embedding(acts))
                sig_list.append(float(s))

        plot_pca(
                embeddings, sig_list,
                os.path.join(out_dir, "pca", "pca_bottleneck_by_sigma.png"),
                title=f"DRUNet bottleneck PCA2D | {len(chosen)} images x {len(sigmas)} sigmas"
        )
        print("[OK] PCA saved.")

    print("[DONE] outputs in:", out_dir)


if __name__ == "__main__":
    main(
        clean_dir=r"./BSDS300/images/test",
        exemplar_name="37073.jpg",
        ckpt_path=r"./weights_drunet_sigmap/drunet_sigmap_final.pth",
        out_dir=r"results_DRUNET/results_DRUNET_latent_analysis_37073",
        sigmas=(0.0, 5.0, 15.0, 25.0, 50.0, 70.0),
        corr_thr=0.90,
        also_run_pca=False,
        n_images_for_pca=25,
    )
