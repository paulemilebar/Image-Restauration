# train_swinir_colab.py
# ------------------------------------------------------------
# Colab-ready training script for SwinIR (sigma-map conditioned)
# Uses Google Colab GPU automatically (cuda).
# Stable defaults: AMP on, light model, safe dataset, prints VRAM usage.
# ------------------------------------------------------------

import os, time, random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from PIL import Image, ImageFile
import torchvision.transforms.functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


# =========================
# SwinIR (sigma-map) model
# =========================
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


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def window_partition(x, window_size: int):
    # x: (B,H,W,C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)

def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size: int, num_heads: int, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1)*(2*window_size-1), num_heads)
        )

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing="ij"
        ))
        coords_flat = coords.flatten(1)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords = rel_coords.permute(1,2,0).contiguous()
        rel_coords[:,:,0] += window_size - 1
        rel_coords[:,:,1] += window_size - 1
        rel_coords[:,:,0] *= 2*window_size - 1
        relative_position_index = rel_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        # x: (B_, N, C), N=Ws*Ws
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2,0,1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1,2).reshape(B_, N, C)
        out = self.proj_drop(self.proj(out))
        return out


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=2.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim*mlp_ratio))

        self.register_buffer("attn_mask", None, persistent=False)

    def build_mask(self, H, W, device):
        if self.shift_size == 0:
            self.attn_mask = None
            return
        ws = self.window_size
        ss = self.shift_size

        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws).view(-1, ws*ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        self.attn_mask = attn_mask

    def forward(self, x, H, W):
        # x: (B, H*W, C)
        B, L, C = x.shape
        if (self.attn_mask is None) and (self.shift_size != 0):
            self.build_mask(H, W, x.device)

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))

        ws = self.window_size
        xw = window_partition(x, ws).view(-1, ws*ws, C)
        aw = self.attn(xw, self.attn_mask)
        aw = aw.view(-1, ws, ws, C)
        x = window_reverse(aw, ws, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1,2))

        x = x.view(B, H*W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualSwinBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=2.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(SwinTransformerBlock(dim, num_heads, window_size, shift, mlp_ratio))
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, H, W):
        shortcut = x
        for b in self.blocks:
            x = b(x, H, W)
        B, L, C = x.shape
        x_img = x.transpose(1,2).contiguous().view(B, C, H, W)
        x_img = self.conv(x_img)
        x = x_img.flatten(2).transpose(1,2).contiguous()
        return x + shortcut


class SwinIRSigmaMap(nn.Module):
    """
    Input : (B,4,H,W) noisy RGB + sigma_map
    Output: (B,3,H,W) denoised
    Residual formulation: out = noisy_rgb + res
    """
    def __init__(self, in_chans=4, out_chans=3, embed_dim=64, window_size=8,
                 depths=(2,2,2,2), num_heads=(4,4,4,4), mlp_ratio=2.0):
        super().__init__()
        self.window_size = window_size
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        self.groups = nn.ModuleList([
            ResidualSwinBlock(embed_dim, depth=depths[i], num_heads=num_heads[i],
                              window_size=window_size, mlp_ratio=mlp_ratio)
            for i in range(len(depths))
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, out_chans, 3, 1, 1)

    def forward(self, inp):
        x, pads = _pad_to_multiple(inp, mult=self.window_size)
        noisy_rgb = x[:, :3, :, :]

        fea = self.conv_first(x)
        B, C, H, W = fea.shape
        tokens = fea.flatten(2).transpose(1,2).contiguous()

        for g in self.groups:
            tokens = g(tokens, H, W)

        tokens = self.norm(tokens)
        fea2 = tokens.transpose(1,2).contiguous().view(B, C, H, W)
        fea2 = self.conv_after_body(fea2) + fea

        res = self.conv_last(fea2)
        out = noisy_rgb + res
        out = _unpad(out, pads)
        return out


# =========================
# Dataset (robust)
# =========================
def list_images(root: str, recursive: bool = True):
    paths = []
    if recursive:
        for r, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(IMG_EXT):
                    paths.append(os.path.join(r, f))
    else:
        for f in os.listdir(root):
            if f.lower().endswith(IMG_EXT):
                paths.append(os.path.join(root, f))
    return sorted(paths)


class RandomPatchSigmaMapDataset(Dataset):
    def __init__(self, clean_dir: str, patch: int = 128,
                 sigma_min: float = 0.0, sigma_max: float = 50.0,
                 recursive: bool = True, max_side: Optional[int] = 2048):
        self.paths = list_images(clean_dir, recursive=recursive)
        if not self.paths:
            raise ValueError(f"No images found in {clean_dir}")
        self.patch = int(patch)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.max_side = max_side
        print(f"[Dataset] {len(self.paths)} images found in: {clean_dir}")

    def __len__(self):
        return 1_000_000

    def _maybe_downscale(self, img: Image.Image) -> Image.Image:
        if self.max_side is None:
            return img
        w, h = img.size
        m = max(w, h)
        if m <= self.max_side:
            return img
        scale = self.max_side / float(m)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        return img.resize((nw, nh), Image.BICUBIC)

    def _random_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        p = self.patch
        if w < p or h < p:
            scale = max(p / w, p / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            img = img.resize((nw, nh), Image.BICUBIC)
            w, h = img.size
        x0 = random.randint(0, w - p)
        y0 = random.randint(0, h - p)
        return img.crop((x0, y0, x0 + p, y0 + p))

    def _augment(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = TF.hflip(img)
        if random.random() < 0.5:
            img = TF.vflip(img)
        k = random.randint(0, 3)
        if k:
            img = img.rotate(90 * k)
        return img

    def __getitem__(self, idx):
        path = random.choice(self.paths)
        with Image.open(path) as im:
            img = im.convert("RGB")
            img = self._maybe_downscale(img)
            img = self._random_crop(img)
            img = self._augment(img)

        clean = TF.to_tensor(img)  # (3,H,W)
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        noise = torch.randn_like(clean) * (sigma / 255.0)
        noisy = clean + noise
        sigma_map = torch.full((1, clean.shape[1], clean.shape[2]), sigma / 255.0)
        inp = torch.cat([noisy, sigma_map], dim=0)
        return inp, clean


# =========================
# Training
# =========================
def train_swinir_colab(
    clean_dir=".",                 # your images are in "."
    out_dir="weights_swinir_sigmap",
    recursive=True,
    patch=128,
    sigma_min=0.0,
    sigma_max=50.0,
    batch_size=2,
    steps_per_epoch=1000,
    max_epochs=10,
    lr0=2e-4,
    lr1=1e-4,
    plateau_epochs=5,
    log_every=50,
    num_workers=2,                 # Colab can handle 2-4
    pin_memory=True,
    use_amp=True,
    grad_clip=1.0,
    max_side=2048,
    seed=0,
    # SwinIR (safe config)
    embed_dim=64,
    window_size=8,
    depths=(2,2,2,2),
    num_heads=(4,4,4,4),
    mlp_ratio=2.0,
):
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM total (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
        torch.backends.cudnn.benchmark = True

    ds = RandomPatchSigmaMapDataset(
        clean_dir=clean_dir,
        patch=patch,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        recursive=recursive,
        max_side=max_side,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda") and pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2
    )

    model = SwinIRSigmaMap(
        in_chans=4, out_chans=3,
        embed_dim=embed_dim,
        window_size=window_size,
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio
    ).to(device).train()

    opt = Adam(model.parameters(), lr=lr0)
    loss_fn = nn.L1Loss()

    if device == "cuda":
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler(enabled=use_amp)
        autocast_ctx = lambda: autocast(enabled=use_amp)
    else:
        scaler = None
        autocast_ctx = None

    best = float("inf")
    stagnant = 0
    using_lr1 = False

    for epoch in range(1, max_epochs + 1):
        running = 0.0
        start_t = time.time()

        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}/{max_epochs}", leave=True)
        it = iter(dl)

        for step in range(steps_per_epoch):
            inp, clean = next(it)
            inp = inp.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            try:
                if device == "cuda":
                    with autocast_ctx():
                        pred = model(inp)
                        loss = loss_fn(pred, clean)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    pred = model(inp)
                    loss = loss_fn(pred, clean)
                    loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[WARN] CUDA OOM -> skip batch, empty_cache")
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise

            running += float(loss.item())

            if (step + 1) % log_every == 0:
                avg_so_far = running / (step + 1)
                elapsed = time.time() - start_t
                it_s = (step + 1) / max(elapsed, 1e-9)

                if device == "cuda":
                    alloc = torch.cuda.memory_allocated() / 1e9
                    reserv = torch.cuda.memory_reserved() / 1e9
                    pbar.set_postfix({
                        "loss": f"{avg_so_far:.5f}",
                        "lr": f"{opt.param_groups[0]['lr']:.1e}",
                        "it/s": f"{it_s:.2f}",
                        "allocGB": f"{alloc:.2f}",
                        "resGB": f"{reserv:.2f}",
                    })
                else:
                    pbar.set_postfix({
                        "loss": f"{avg_so_far:.5f}",
                        "lr": f"{opt.param_groups[0]['lr']:.1e}",
                        "it/s": f"{it_s:.2f}"
                    })

            pbar.update(1)

        pbar.close()
        avg = running / steps_per_epoch

        ckpt = os.path.join(out_dir, f"swinir_sigmap_epoch{epoch:02d}.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
        print(f"Epoch {epoch:02d} done | avg_loss={avg:.6f} | lr={opt.param_groups[0]['lr']:.1e}")

        if avg < best - 1e-7:
            best = avg
            stagnant = 0
        else:
            stagnant += 1

        if (not using_lr1) and stagnant >= plateau_epochs:
            for g in opt.param_groups:
                g["lr"] = lr1
            using_lr1 = True
            stagnant = 0
            print(f"Switch LR to {lr1}")

        if using_lr1 and stagnant >= plateau_epochs:
            print("Early stop: loss plateaued.")
            break

    final_path = os.path.join(out_dir, "swinir_sigmap_final.pth")
    torch.save({"model": model.state_dict()}, final_path)
    print(f"Saved: {final_path}")
    return final_path


if __name__ == "__main__":
    train_swinir_colab(
        clean_dir=".",        # images in current directory (or mount drive path)
        out_dir="weights_swinir_sigmap",
        recursive=True,
        patch=128,
        sigma_min=0.0,
        sigma_max=50.0,
        batch_size=2,
        steps_per_epoch=1000,
        max_epochs=10,
        lr0=2e-4,
        lr1=1e-4,
        plateau_epochs=5,
        log_every=50,
        num_workers=2,
        pin_memory=True,
        use_amp=True,
        grad_clip=1.0,
        max_side=2048,
        # model safe config
        embed_dim=64,
        window_size=8,
        depths=(2,2,2,2),
        num_heads=(4,4,4,4),
        mlp_ratio=2.0,
        seed=0
    )
