# analyze_drunet_latent.py
import os, math, random
from glob import glob

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

# sklearn (PCA) optionnel
try:
    from sklearn.decomposition import PCA
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# -------------------------
# Utils I/O
# -------------------------
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
    t01_chw = t01_chw.detach().cpu().clamp(0,1)
    TF.to_pil_image(t01_chw).save(path)

def save_gray01(gray_hw, path):
    # gray_hw in [0,1]
    arr = (gray_hw.detach().cpu().clamp(0,1).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)

def norm01(x, eps=1e-12):
    x = x - x.min()
    return x / (x.max() + eps)

def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Noise + input builder
# -------------------------
def add_awgn(clean_01, sigma_pixels, generator=None):
    """
    clean_01: (1,3,H,W) in [0,1]
    sigma_pixels: float
    generator: torch.Generator optionnel (utilisé uniquement sur CPU)
    """
    sigma = float(sigma_pixels) / 255.0

    if clean_01.device.type == "cuda":
        # GPU: pas de generator (évite mismatch)
        noise = torch.randn_like(clean_01) * sigma
    else:
        # CPU: generator OK et reproductible
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
    returns inp: (1,4,H,W) with sigma_map
    """
    B, C, H, W = noisy.shape
    sigma_map = torch.full((B,1,H,W), float(sigma_pixels)/255.0, device=noisy.device, dtype=noisy.dtype)
    return torch.cat([noisy, sigma_map], dim=1)


# -------------------------
# Forward with latents (reprend ta logique)
# -------------------------
@torch.no_grad()
def forward_collect(model, inp):
    """
    Retourne out + dict d'activations
    Clés:
      head, x1, x2, x3, x4 (bottleneck),
      y3, y2, y1 (après fusion+blocks), out
    """
    acts = {}

    # pad comme ton forward
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


# -------------------------
# Copy of your pad helpers
# -------------------------
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
    if (pl, pr, pt, pb) == (0,0,0,0):
        return x
    return x[:, :, pt:x.shape[2]-pb, pl:x.shape[3]-pr]


# -------------------------
# Filter visualizations
# -------------------------
def plot_head_filters(model, out_path_png):
    """
    head.weight: (64,4,3,3)
    - grille RGB (canaux 0..2)
    - grille sigma (canal 3)
    """
    W = model.head.weight.detach().cpu()  # (64,4,3,3)
    oc, ic, kh, kw = W.shape
    assert ic == 4 and kh == 3 and kw == 3

    # RGB composite: (oc,3,3,3)
    W_rgb = W[:, :3, :, :]
    # normalisation par filtre
    wr = W_rgb.reshape(oc, -1)
    wr = (wr - wr.mean(dim=1, keepdim=True)) / (wr.std(dim=1, keepdim=True) + 1e-8)
    W_rgbn = wr.reshape(oc, 3, 3, 3)

    # sigma channel: (oc,3,3)
    W_s = W[:, 3, :, :]
    ws = W_s.reshape(oc, -1)
    ws = (ws - ws.mean(dim=1, keepdim=True)) / (ws.std(dim=1, keepdim=True) + 1e-8)
    W_sn = ws.reshape(oc, 3, 3)

    # grid size
    n = oc
    grid = int(math.ceil(math.sqrt(n)))
    fig = plt.figure(figsize=(12, 6))

    # --- RGB grid
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Head filters (RGB channels 0..2)")
    canvas = np.ones((grid*3, grid*3, 3), dtype=np.float32) * 0.5

    k = 0
    for i in range(grid):
        for j in range(grid):
            if k >= n:
                break
            ker = W_rgbn[k].permute(1,2,0).numpy()  # (3,3,3)
            ker = (ker - ker.min()) / (ker.max() - ker.min() + 1e-8)
            canvas[i*3:(i+1)*3, j*3:(j+1)*3, :] = ker
            k += 1

    ax1.imshow(canvas)
    ax1.axis("off")

    # --- Sigma grid
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
    """
    Histos + norms pour quelques convs importantes
    """
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
        ax.set_title(f"{name}: weight hist (mean={W.mean():.2e}, std={W.std():.2e})")
        idx += 1
        if idx > 12:
            break

    plt.tight_layout()
    plt.savefig(out_path_png, dpi=200)
    plt.close(fig)


# -------------------------
# Latent visualizations
# -------------------------
def feature_energy_map(feat_bchw):
    """
    feat: (1,C,H,W)
    returns (H,W) in [0,1] representing sqrt(sum_c feat^2)
    """
    f = feat_bchw[0]
    e = torch.sqrt(torch.sum(f*f, dim=0) + 1e-12)  # (H,W)
    return norm01(e)

def topk_channel_grid(feat_bchw, out_png, k=16, title=""):
    """
    Affiche les k canaux à plus forte variance spatiale
    """
    f = feat_bchw[0].detach().cpu()  # (C,H,W)
    C, H, W = f.shape
    var = f.view(C, -1).var(dim=1)
    idx = torch.topk(var, k=min(k, C)).indices.tolist()

    grid = int(math.ceil(math.sqrt(len(idx))))
    fig = plt.figure(figsize=(10, 10))
    for t, c in enumerate(idx):
        ax = fig.add_subplot(grid, grid, t+1)
        img = f[c]
        img = norm01(img)
        ax.imshow(img.numpy(), cmap="gray")
        ax.set_title(f"ch {c}")
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

def save_latent_pack(acts, out_dir, prefix):
    """
    Sauve energy maps + top channels pour plusieurs niveaux
    """
    keys = ["head","x1","x2","x3","x4","y3","y2","y1"]
    for k in keys:
        if k not in acts:
            continue
        e = feature_energy_map(acts[k])
        save_gray01(e, os.path.join(out_dir, f"{prefix}_{k}_energy.png"))
        topk_channel_grid(
            acts[k],
            os.path.join(out_dir, f"{prefix}_{k}_topch.png"),
            k=16,
            title=f"{prefix} | {k} top variance channels"
        )


# -------------------------
# PCA embedding (bottleneck)
# -------------------------
def bottleneck_embedding(acts):
    """
    GAP sur x4 -> (C,)
    """
    x4 = acts["x4"]  # (1, C, H, W)
    emb = F.adaptive_avg_pool2d(x4, 1).view(-1)  # (C,)
    return emb.detach().cpu().numpy()

def plot_pca(embeddings, sigmas, out_png, title):
    if not SKLEARN_OK:
        print("[WARN] sklearn not available -> skip PCA plot")
        return
    X = np.stack(embeddings, axis=0)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)
    sc = ax.scatter(Z[:,0], Z[:,1], c=sigmas, s=18)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("sigma (pixels)")
    ax.set_title(title + f" | explained var: {pca.explained_variance_ratio_.sum():.2f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main(
    clean_dir=r"./BSDS300/images/test",
    ckpt_path=r"./weights_drunet_sigmap/drunet_sigmap_final.pth",
    out_dir=r"./latent_analysis_drunet",
    n_images=25,
    sigmas=(0.0, 5.0, 15.0, 25.0, 50.0, 70.0),
    seed=1,
    device=None
):
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "filters"))
    ensure_dir(os.path.join(out_dir, "latents"))
    ensure_dir(os.path.join(out_dir, "pca"))

    seed_all(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ---- load model (tu importes ta classe comme tu fais déjà)
    # from DRUNet import DRUNetSigmaMap
    from DRUNet import DRUNetSigmaMap

    model = DRUNetSigmaMap(in_nc=4, out_nc=3, nc=(64,128,256,512), nb=4).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    # ---- filters
    plot_head_filters(model, os.path.join(out_dir, "filters", "head_filters.png"))
    plot_weight_stats(model, os.path.join(out_dir, "filters", "weight_hists.png"))
    print("[OK] saved filter figures")

    # ---- choose images
    paths = list_images(clean_dir)
    if not paths:
        raise ValueError(f"No images in {clean_dir}")
    if n_images > len(paths):
        n_images = len(paths)
    chosen = random.sample(paths, n_images)

    # ---- run latent extraction + PCA dataset
    g = torch.Generator().manual_seed(seed)

    embeddings = []
    sig_list = []
    name_list = []

    # on va aussi sauver un “pack” complet pour la 1ère image à un sigma donné
    exemplar_path = chosen[0]
    exemplar_sigma = float(sigmas[len(sigmas)//2])

    for ip, p in enumerate(chosen):
        pil = Image.open(p).convert("RGB")
        clean = to_tensor01(pil).unsqueeze(0)  # (1,3,H,W)
        clean = clean.to(device)

        for s in sigmas:
            noisy = add_awgn(clean, s, generator=g)  # (1,3,H,W)
            inp = build_inp(noisy, s)

            out, acts = forward_collect(model, inp)
            emb = bottleneck_embedding(acts)

            embeddings.append(emb)
            sig_list.append(float(s))
            name_list.append(os.path.basename(p))

        # exemplar latent pack
        if p == exemplar_path:
            noisy = add_awgn(clean, exemplar_sigma, generator=g)
            inp = build_inp(noisy, exemplar_sigma)
            out, acts = forward_collect(model, inp)

            # sauver entrée/sortie (clamp seulement pour visualisation)
            save_img01(clean[0].detach().cpu(), os.path.join(out_dir, "latents", "exemplar_clean.png"))
            save_img01(noisy[0].detach().cpu().clamp(0,1), os.path.join(out_dir, "latents", f"exemplar_noisy_sigma{int(exemplar_sigma)}.png"))
            save_img01(out[0].detach().cpu().clamp(0,1), os.path.join(out_dir, "latents", f"exemplar_out_sigma{int(exemplar_sigma)}.png"))

            save_latent_pack(acts, os.path.join(out_dir, "latents"), prefix=f"exemplar_sigma{int(exemplar_sigma)}")

    # ---- PCA plot (bottleneck embeddings)
    plot_pca(
        embeddings, sig_list,
        os.path.join(out_dir, "pca", "pca_bottleneck_by_sigma.png"),
        title=f"DRUNet bottleneck (x4 GAP) PCA2D | {n_images} images x {len(sigmas)} sigmas"
    )

    # ---- simple sigma sensitivity metric (same-image embeddings across sigma)
    # on calcule la similarité cos moyenne entre sigma=0 et autres pour chaque image
    # (juste un indicateur)
    if len(sigmas) >= 2:
        sigmas = list(sigmas)
        s0 = sigmas[0]
        # reshape: (n_images, n_sigmas, dim)
        dim = embeddings[0].shape[0]
        X = np.stack(embeddings, axis=0).reshape(n_images, len(sigmas), dim)
        # normalize
        Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)
        i0 = 0
        cos = np.sum(Xn[:, i0:i0+1, :] * Xn, axis=-1)  # (n_images, n_sigmas)

        txt = []
        txt.append(f"Cosine similarity to sigma={s0} (mean over images):")
        for j, sj in enumerate(sigmas):
            txt.append(f"  sigma {sj:>5}: {cos[:,j].mean():.4f} +/- {cos[:,j].std():.4f}")
        with open(os.path.join(out_dir, "pca", "sigma_sensitivity.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(txt))

    print("[DONE] outputs in:", out_dir)


if __name__ == "__main__":
    main(
        clean_dir=r"./BSDS300/images/test",
        ckpt_path=r"./weights_drunet_sigmap/drunet_sigmap_final.pth",
        out_dir=r"./results_DRUNET_latent_analysis",
        n_images=25,
        sigmas=(0.0, 5.0, 15.0, 25.0, 50.0, 70.0)
        )
