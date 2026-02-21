import os, random
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

class RandomPatchSigmaMapDataset(Dataset):
    """
    Random patches from clean images + AWGN with random sigma.
    Returns:
      inp   (4,H,W) = noisy_rgb (3) + sigma_map (1)
      clean (3,H,W)
    """
    def __init__(self, clean_dir: str, patch: int = 128, sigma_min: float = 0.0, sigma_max: float = 50.0):
        self.paths = []
        for root, _, files in os.walk(clean_dir):
            for f in files:
                if f.lower().endswith(IMG_EXT):
                    self.paths.append(os.path.join(root, f))
        if not self.paths:
            raise ValueError(f"No images found in {clean_dir}")

        self.patch = patch
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def __len__(self):
        return 1_000_000

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
        img = Image.open(path).convert("RGB")
        img = self._random_crop(img)
        img = self._augment(img)

        clean = TF.to_tensor(img)  # (3,H,W) in [0,1]

        # sigma in "pixel space" 0..50 (paper-like)
        sigma = random.uniform(self.sigma_min, self.sigma_max)

        # IMPORTANT: NO CLIP OF NOISY (paper-like)
        noise = torch.randn_like(clean) * (sigma / 255.0)
        noisy = clean + noise  # peut sortir de [0,1]

        sigma_map = torch.full((1, clean.shape[1], clean.shape[2]), sigma / 255.0)
        inp = torch.cat([noisy, sigma_map], dim=0)  # (4,H,W)
        return inp, clean

# DRUNet: bias-free, 4 scales, SConv 2x2, TConv 2x2, nb=4
class ResBlockOneReLU(nn.Module):
    """
    Residual block: Conv -> ReLU -> Conv, bias-free, un seul ReLU.
    """
    def __init__(self, nc: int):
        super().__init__()
        self.c1 = nn.Conv2d(nc, nc, 3, 1, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.c2 = nn.Conv2d(nc, nc, 3, 1, 1, bias=False)

    def forward(self, x):
        y = self.c1(x)
        y = self.act(y)
        y = self.c2(y)
        return x + y


class SConv2x2(nn.Module):
    """2×2 strided conv downscale (pas d'activation après, paper-like)"""
    def __init__(self, in_nc: int, out_nc: int):
        super().__init__()
        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x):
        return self.conv(x)


class TConv2x2(nn.Module):
    """2×2 transposed conv upscale (pas d'activation après, paper-like)"""
    def __init__(self, in_nc: int, out_nc: int):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x):
        return self.tconv(x)


def _pad_to_multiple(x: torch.Tensor, mult: int = 8) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
    ## Mirror padding
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

def _unpad(x: torch.Tensor, pads: Tuple[int,int,int,int]) -> torch.Tensor:
    pl, pr, pt, pb = pads
    if (pl, pr, pt, pb) == (0,0,0,0):
        return x
    return x[:, :, pt:x.shape[2]-pb, pl:x.shape[3]-pr]


class DRUNetSigmaMap(nn.Module):
    """
    Input:  (B,4,H,W) noisy RGB + sigma_map (4 channels)
    Output: (B,3,H,W) denoised
    paper-like : 4 scales, nc=[64,128,256,512], nb=4,
    bias-free, SConv 2x2, TConv 2x2, pas d'activation après head/tail/SConv/TConv.
    """
    def __init__(self, in_nc=4, out_nc=3, nc=(64,128,256,512), nb=4):
        super().__init__()
        c1, c2, c3, c4 = nc

        # head: pas d'activation après
        self.head = nn.Conv2d(in_nc, c1, 3, 1, 1, bias=False)

        # encoder
        self.e1 = nn.Sequential(*[ResBlockOneReLU(c1) for _ in range(nb)])
        self.d1 = SConv2x2(c1, c2)

        self.e2 = nn.Sequential(*[ResBlockOneReLU(c2) for _ in range(nb)])
        self.d2 = SConv2x2(c2, c3)

        self.e3 = nn.Sequential(*[ResBlockOneReLU(c3) for _ in range(nb)])
        self.d3 = SConv2x2(c3, c4)

        # bottleneck
        self.mid = nn.Sequential(*[ResBlockOneReLU(c4) for _ in range(nb)])

        # decoder
        self.u3 = TConv2x2(c4, c3)
        self.f3 = nn.Conv2d(c3 + c3, c3, 3, 1, 1, bias=False)
        self.p3 = nn.Sequential(*[ResBlockOneReLU(c3) for _ in range(nb)])

        self.u2 = TConv2x2(c3, c2)
        self.f2 = nn.Conv2d(c2 + c2, c2, 3, 1, 1, bias=False)
        self.p2 = nn.Sequential(*[ResBlockOneReLU(c2) for _ in range(nb)])

        self.u1 = TConv2x2(c2, c1)
        self.f1 = nn.Conv2d(c1 + c1, c1, 3, 1, 1, bias=False)
        self.p1 = nn.Sequential(*[ResBlockOneReLU(c1) for _ in range(nb)])

        # tail: pas d'activation après
        self.tail = nn.Conv2d(c1, out_nc, 3, 1, 1, bias=False)

    def forward(self, inp):
        # pad pour tailles quelconques (facteur 8 car 3 downsamples)
        x, pads = _pad_to_multiple(inp, mult=8)

        x1 = self.e1(self.head(x))
        x2 = self.e2(self.d1(x1))
        x3 = self.e3(self.d2(x2))
        x4 = self.mid(self.d3(x3))

        y3 = self.u3(x4)
        y3 = self.p3(self.f3(torch.cat([y3, x3], dim=1)))

        y2 = self.u2(y3)
        y2 = self.p2(self.f2(torch.cat([y2, x2], dim=1)))

        y1 = self.u1(y2)
        y1 = self.p1(self.f1(torch.cat([y1, x1], dim=1)))

        out = self.tail(y1)
        out = _unpad(out, pads)
        return out

