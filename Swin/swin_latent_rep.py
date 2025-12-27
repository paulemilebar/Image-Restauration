# analyze_swinir_latent.py
# Latent analysis for SwinIR (sigma-map conditioned, 4ch input), in the same spirit as analyze_drunet_latent.py
#
# What it saves:
#  - filters/conv_first_filters.png  : first conv filters (RGB + sigma channel)
#  - filters/weight_hists.png        : weight histograms for a few key conv/linear layers
#  - latents/exemplar_*              : exemplar clean/noisy/out + latent packs (energy + top channels)
#  - pca/pca_token_mean_by_sigma.png : PCA on token embeddings (mean over tokens)
#  - pca/pca_token_mean_labeled.png  : PCA with point indices + legend mapping point->(img,sigma)
#  - pca/point_legend.csv            : mapping of point_id to (image_name, sigma)
#  - pca/sigma_sensitivity.txt       : cosine similarity vs sigma=first sigma
#
# Requirements:
#  - Your Swin model class is SwinIRSigmaMap (from swinir_model.py or SWIN.py)
#  - Checkpoint contains {"model": state_dict} or raw state_dict
#
# Notes:
#  - To be robust to your earlier key mismatch, this script remaps:
#      groups.i.blocks.*  -> groups.i.layer.blocks.*   (if needed)
#  - It also infers embed_dim and depths from the checkpoint, so you don't accidentally mismatch config.

import os, math, random, re, csv
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

# sklearn (PCA) optional
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
    returns inp: (1,4,H,W) with sigma_map
    """
    B, C, H, W = noisy.shape
    sigma_map = torch.full((B,1,H,W), float(sigma_pixels)/255.0, device=noisy.device, dtype=noisy.dtype)
    return torch.cat([noisy, sigma_map], dim=1)


# -------------------------
# Helpers: remap checkpoint keys 
# -------------------------
def remap_plain_blocks_to_layer(sd):
    """
    If checkpoint uses:
      groups.0.blocks.0....
    but model expects:
      groups.0.layer.blocks.0....
    remap automatically.
    """
    keys = list(sd.keys())
    has_plain = any(k.startswith("groups.0.blocks.") for k in keys)
    has_layer = any(k.startswith("groups.0.layer.blocks.") for k in keys)
    if has_plain and not has_layer:
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("groups."):
                parts = k.split(".")
                if len(parts) >= 3 and parts[2] == "blocks":
                    k2 = ".".join([parts[0], parts[1], "layer"] + parts[2:])
                    new_sd[k2] = v
                else:
                    new_sd[k] = v
            else:
                new_sd[k] = v
        return new_sd
    return sd


# -------------------------
# Filter visualizations (conv_first)
# -------------------------
def plot_conv_first_filters(model, out_path_png, max_out=64):
    """
    conv_first.weight: (embed_dim, 4, 3, 3)
    Show RGB composite + sigma channel like DRUNet head filters.
    """
    W = model.conv_first.weight.detach().cpu()  # (E,4,3,3)
    oc, ic, kh, kw = W.shape
    assert ic == 4 and kh == 3 and kw == 3
    n = min(oc, max_out)

    W = W[:n]

    # RGB composite
    W_rgb = W[:, :3, :, :]
    wr = W_rgb.reshape(n, -1)
    wr = (wr - wr.mean(dim=1, keepdim=True)) / (wr.std(dim=1, keepdim=True) + 1e-8)
    W_rgbn = wr.reshape(n, 3, 3, 3)

    # sigma channel
    W_s = W[:, 3, :, :]
    ws = W_s.reshape(n, -1)
    ws = (ws - ws.mean(dim=1, keepdim=True)) / (ws.std(dim=1, keepdim=True) + 1e-8)
    W_sn = ws.reshape(n, 3, 3)

    grid = int(math.ceil(math.sqrt(n)))
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("conv_first filters (RGB channels 0..2)")
    canvas = np.ones((grid*3, grid*3, 3), dtype=np.float32) * 0.5

    k = 0
    for i in range(grid):
        for j in range(grid):
            if k >= n:
                break
            ker = W_rgbn[k].permute(1,2,0).numpy()
            ker = (ker - ker.min()) / (ker.max() - ker.min() + 1e-8)
            canvas[i*3:(i+1)*3, j*3:(j+1)*3, :] = ker
            k += 1
    ax1.imshow(canvas)
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("conv_first filters (sigma_map channel 3)")
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
    Histos on a few key layers:
      conv_first, conv_after_body, conv_last + group convs
    """
    convs = {
        "conv_first": model.conv_first,
        "conv_after_body": model.conv_after_body,
        "conv_last": model.conv_last,
    }

    # add residual group convs if present
    for i, g in enumerate(getattr(model, "groups", [])):
        if hasattr(g, "conv"):
            convs[f"group{i}.conv"] = g.conv

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


# -------------------------
# Latent visualizations
# -------------------------
def feature_energy_map(feat_bchw):
    """
    feat: (1,C,H,W) -> (H,W) energy map
    """
    f = feat_bchw[0]
    e = torch.sqrt(torch.sum(f*f, dim=0) + 1e-12)
    return norm01(e)

def topk_channel_grid(feat_bchw, out_png, k=16, title=""):
    """
    Show k channels with highest spatial variance.
    """
    f = feat_bchw[0].detach().cpu()  # (C,H,W)
    C, H, W = f.shape
    var = f.view(C, -1).var(dim=1)
    idx = torch.topk(var, k=min(k, C)).indices.tolist()

    grid = int(math.ceil(math.sqrt(len(idx))))
    fig = plt.figure(figsize=(10, 10))
    for t, c in enumerate(idx):
        ax = fig.add_subplot(grid, grid, t+1)
        img = norm01(f[c])
        ax.imshow(img.numpy(), cmap="gray")
        ax.set_title(f"ch {c}")
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

def save_latent_pack(acts, out_dir, prefix):
    """
    Save energy maps + top channels for a few interesting tensors.
    For SwinIR we look at:
      fea0 (after conv_first)
      fea_body (after transformer body, before conv_after_body residual)
      fea2 (after conv_after_body + skip)
      res  (predicted residual)
    """
    keys = ["fea0", "fea_body", "fea2", "res"]
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
# Token embeddings for PCA
# -------------------------
def token_embedding_mean(tokens_blnc):
    """
    tokens: (B, L, C)
    -> global mean over tokens : (C,)
    """
    emb = tokens_blnc.mean(dim=1).view(-1)
    return emb.detach().cpu().numpy()

def plot_pca(embeddings, sigmas, out_png, title):
    if not SKLEARN_OK:
        print("[WARN] sklearn not available -> skip PCA plot")
        return None
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
    return Z

def plot_pca_labeled(Z, sigmas, labels, out_png, title):
    """
    Z: (N,2)
    labels: list of strings (short), one per point
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1,1,1)
    sc = ax.scatter(Z[:,0], Z[:,1], c=sigmas, s=18)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("sigma (pixels)")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # annotate points lightly (small text). For many points, this can clutter.
    # We'll annotate with point index, and save a CSV mapping idx->(img,sigma).
    for i in range(Z.shape[0]):
        ax.text(Z[i,0], Z[i,1], labels[i], fontsize=6)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)


# -------------------------
# Forward with latents (SwinIR)
# -------------------------
@torch.no_grad()
def forward_collect_swin(model, inp):
    """
    Returns out + dict of activations:
      inp_padded
      noisy_rgb
      fea0      : conv_first output (B,C,H,W)
      tokens0   : initial tokens (B,L,C)
      tokensT   : final tokens after groups (B,L,C)
      fea_body  : tokensT -> (B,C,H,W) AFTER norm (before conv_after_body)
      fea2      : after conv_after_body + skip (B,C,H,W)
      res       : predicted residual (B,3,H,W)
      out       : final output (B,3,H,W)
    """
    acts = {}

    # pad to multiple of window_size for window partitioning
    x, pads = _pad_to_multiple(inp, mult=model.window_size)
    acts["inp_padded"] = x

    noisy_rgb = x[:, :3, :, :]
    acts["noisy_rgb"] = noisy_rgb

    fea0 = model.conv_first(x)  # (B,C,H,W)
    acts["fea0"] = fea0
    B, C, H, W = fea0.shape

    tokens0 = fea0.flatten(2).transpose(1,2).contiguous()  # (B,L,C)
    acts["tokens0"] = tokens0

    tokens = tokens0
    for g in model.groups:
        tokens = g(tokens, H, W)
    acts["tokensT"] = tokens

    tokensN = model.norm(tokens)
    fea_body = tokensN.transpose(1,2).contiguous().view(B, C, H, W)
    acts["fea_body"] = fea_body

    fea2 = model.conv_after_body(fea_body) + fea0
    acts["fea2"] = fea2

    res = model.conv_last(fea2)        # (B,3,H,W)
    acts["res"] = res

    out = noisy_rgb + res
    out = _unpad(out, pads)
    acts["out"] = out
    return out, acts


def _pad_to_multiple(x: torch.Tensor, mult: int):
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
# Main
# -------------------------
def main(
    clean_dir=r"./BSDS300/images/test",
    ckpt_path=r"./weights_swinir_sigmap/swinir_sigmap_final.pth",
    out_dir=r"./latent_analysis_swinir",
    n_images=25,
    sigmas=(0.0, 5.0, 15.0, 25.0, 50.0, 70.0),
    num_heads = (4,4,4,4),
    depths = (2,2,2,2),
    embed_dim = 64,
    seed=1,
    device=None,
    import_from="SWIN"  # "swinir_model" or "SWIN"
):
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "filters"))
    ensure_dir(os.path.join(out_dir, "latents"))
    ensure_dir(os.path.join(out_dir, "pca"))

    seed_all(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # ---- import model class
    if import_from == "SWIN":
        from SWIN import SwinIRSigmaMap

    # ---- load checkpoint
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = state["model"] if (isinstance(state, dict) and "model" in state) else state
    sd = remap_plain_blocks_to_layer(sd)
    
    print("Inferred config from ckpt:")
    print("  embed_dim :", embed_dim)
    print("  depths    :", depths)
    print("  num_heads :", num_heads)

    # ---- build model
    model = SwinIRSigmaMap(
        in_chans=4,
        out_chans=3,
        embed_dim=embed_dim,
        window_size=8,
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=2.0
    ).to(device).eval()

    model.load_state_dict(sd, strict=True)

    # ---- filters
    plot_conv_first_filters(model, os.path.join(out_dir, "filters", "conv_first_filters.png"))
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
    point_ids = []

    exemplar_path = chosen[0]
    exemplar_sigma = float(sigmas[len(sigmas)//2])

    pid = 0
    for ip, p in enumerate(chosen):
        pil = Image.open(p).convert("RGB")
        clean = to_tensor01(pil).unsqueeze(0).to(device)

        for s in sigmas:
            noisy = add_awgn(clean, s, generator=g)
            inp = build_inp(noisy, s)

            out, acts = forward_collect_swin(model, inp)

            # embedding choice for PCA:
            #   mean of final tokens (B,L,C) -> (C,)
            emb = token_embedding_mean(acts["tokensT"])

            embeddings.append(emb)
            sig_list.append(float(s))
            name_list.append(os.path.basename(p))
            point_ids.append(pid)
            pid += 1

        # exemplar latent pack
        if p == exemplar_path:
            noisy = add_awgn(clean, exemplar_sigma, generator=g)
            inp = build_inp(noisy, exemplar_sigma)
            out, acts = forward_collect_swin(model, inp)

            # save images (clamp only for viz)
            save_img01(clean[0].detach().cpu(), os.path.join(out_dir, "latents", "exemplar_clean.png"))
            save_img01(noisy[0].detach().cpu().clamp(0,1), os.path.join(out_dir, "latents", f"exemplar_noisy_sigma{int(exemplar_sigma)}.png"))
            save_img01(out[0].detach().cpu().clamp(0,1), os.path.join(out_dir, "latents", f"exemplar_out_sigma{int(exemplar_sigma)}.png"))

            save_latent_pack(acts, os.path.join(out_dir, "latents"), prefix=f"exemplar_sigma{int(exemplar_sigma)}")

    # ---- PCA plots
    if SKLEARN_OK:
        Z = plot_pca(
            embeddings, sig_list,
            os.path.join(out_dir, "pca", "pca_token_mean_by_sigma.png"),
            title=f"SwinIR tokens mean PCA2D | {n_images} images x {len(sigmas)} sigmas"
        )

        # Save mapping point_id -> (img, sigma)
        legend_csv = os.path.join(out_dir, "pca", "point_legend.csv")
        with open(legend_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["point_id", "image_name", "sigma_pixels"])
            for i in range(len(point_ids)):
                w.writerow([point_ids[i], name_list[i], sig_list[i]])

        # labeled plot (index labels on plot)
        labels = [str(i) for i in point_ids]
        plot_pca_labeled(
            Z, sig_list, labels,
            os.path.join(out_dir, "pca", "pca_token_mean_labeled.png"),
            title="PCA labeled by point_id (see point_legend.csv for mapping)"
        )
    else:
        print("[WARN] sklearn not available -> skip PCA plots")

    # ---- sigma sensitivity (cosine similarity to sigma = first sigma)
    if len(sigmas) >= 2:
        sigmas_list = list(sigmas)
        s0 = sigmas_list[0]

        dim = embeddings[0].shape[0]
        X = np.stack(embeddings, axis=0).reshape(n_images, len(sigmas_list), dim)

        Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)
        cos = np.sum(Xn[:, 0:1, :] * Xn, axis=-1)

        txt = []
        txt.append(f"Cosine similarity to sigma={s0} (mean over images):")
        for j, sj in enumerate(sigmas_list):
            txt.append(f"  sigma {sj:>5}: {cos[:,j].mean():.4f} +/- {cos[:,j].std():.4f}")

        with open(os.path.join(out_dir, "pca", "sigma_sensitivity.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(txt))

    print("[DONE] outputs in:", out_dir)


if __name__ == "__main__":
    main(
        clean_dir=r"./BSDS300/images/test",
        ckpt_path=r"./weights_swinir_sigmap/swinir_sigmap_final.pth",
        out_dir=r"./results_latent_analysis_swinir",
        n_images=25,
        sigmas=(0.0, 5.0, 15.0, 25.0, 50.0, 70.0),
        seed=1,
        device=None,
        import_from="SWIN" 
    )
