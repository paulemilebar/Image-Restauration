import os, random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# Preparation of the data we will use for training
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

class RandomPatchSigmaMapDataset(Dataset):
    def __init__(self, clean_dir: str, patch: int = 30, sigma_min: float = 0.0, sigma_max: float = 50.0):
        self.paths = [
            os.path.join(clean_dir, f) for f in os.listdir(clean_dir)
            if f.lower().endswith(IMG_EXT)
        ]
        if not self.paths:
            raise ValueError(f"No images found in {clean_dir}")
        self.patch = patch
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __len__(self):
        return 1000000

    def _random_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < self.patch or h < self.patch:
            img = img.resize((max(w, self.patch), max(h, self.patch)))
            w, h = img.size
        x0 = random.randint(0, w - self.patch)
        y0 = random.randint(0, h - self.patch)
        return img.crop((x0, y0, x0 + self.patch, y0 + self.patch))

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

        clean = TF.to_tensor(img)

        sigma = random.uniform(self.sigma_min, self.sigma_max)
        noise = torch.randn_like(clean) * (sigma / 255.0)
        noisy = (clean + noise)

        sigma_map = torch.full((1, clean.shape[1], clean.shape[2]), sigma / 255.0)
        inp = torch.cat([noisy, sigma_map], dim=0)

        return inp, clean

# Model: IRCNN (7-layer dilated) + sigma map
class IRCNNSigmaMap(nn.Module):
    """
    Input:  (B,4,H,W) = noisy RGB + sigma map
    Output: (B,3,H,W) clean
    Residual learning: predict noise implicitly then subtract
    """
    def __init__(self, features: int = 64):
        super().__init__()
        dilations = [1, 2, 3, 4, 3, 2, 1]
        layers = []

        d = dilations[0]
        layers += [
            nn.Conv2d(4, features, 3, padding=d, dilation=d, bias=True),
            nn.ReLU(inplace=True),
        ]

        for d in dilations[1:-1]:
            layers += [
                nn.Conv2d(features, features, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            ]

        d = dilations[-1]
        layers += [nn.Conv2d(features, 3, 3, padding=d, dilation=d, bias=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, inp):
        pred_noise = self.net(inp)
        noisy = inp[:, :3, :, :]
        clean = (noisy - pred_noise)
        return clean
