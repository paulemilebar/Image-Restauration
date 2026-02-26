import os, random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

class FixedSigmaPatchDataset(Dataset):
    """
    Sample patchs and add fixed gaussian noise specific to a model
    Outputs :
      noisy (3,H,W)
      clean (3,H,W)
    """
    def __init__(self, clean_dir: str, patch: int = 35, sigma: float = 25.0):
        self.paths = [
            os.path.join(clean_dir, f) for f in os.listdir(clean_dir)
            if f.lower().endswith(IMG_EXT)
        ]
        if not self.paths:
            raise ValueError(f"No images found in {clean_dir}")
        self.patch = patch
        self.sigma = sigma

    def __len__(self):
        return 1000000

    def _random_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < self.patch or h < self.patch:
            img = img.resize((max(w, self.patch), max(h, self.patch)), Image.BICUBIC)
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
        if k > 0:
            img = img.rotate(90 * k)
        return img

    def __getitem__(self, idx):
        path = random.choice(self.paths)
        img = Image.open(path).convert("RGB")
        img = self._random_crop(img)
        img = self._augment(img)

        clean = TF.to_tensor(img)
        noise = torch.randn_like(clean) * (self.sigma / 255.0)
        noisy = clean + noise

        return noisy, clean
    
    
class IRCNNFixed(nn.Module):
    """
    Input:  (B,3,H,W) = noisy RGB
    Output: (B,3,H,W) clean
    Residual learning: predict noise implicitly then subtract
    """
    def __init__(self, in_channels=3, n_filters=64):
        super(IRCNNFixed, self).__init__()
        dilations = [1, 2, 3, 4, 3, 2, 1]
        self.layers = nn.ModuleList()

        for i, d in enumerate(dilations):
            if i==0:
                self.layers.append(nn.Conv2d(in_channels, n_filters, 3, padding=d, dilation=d, bias=True))
                self.layers.append(nn.ReLU(inplace=True))
            elif i < 6:
                self.layers.append(nn.Conv2d(n_filters, n_filters, 3, padding=d, dilation=d, bias=False))
                self.layers.append(nn.BatchNorm2d(n_filters))
                self.layers.append(nn.ReLU(inplace=True))
            else:
                self.layers.append(nn.Conv2d(n_filters, in_channels, 3, padding=d, dilation=d, bias=True))


    def forward(self, x):
        noise_pred = x
        for layer in self.layers:
            noise_pred = layer(noise_pred)
        return noise_pred

    def denoise(self, x):
      return x - self.forward(x)


class IRCNNModelManager:
    """ Manage dynamic loading of our 25 experts. """
    def __init__(self, model_dir, device="cpu"):
        self.model_dir = model_dir
        self.device = device
        self.available_sigmas = [2*i for i in range(1,26)] 
        self.model = IRCNNFixed(n_filters=64).to(device)
        self.current_sigma = None

    def get_expert(self, target_sigma):
        closest = min(self.available_sigmas, key=lambda x: abs(x - target_sigma))
        if closest != self.current_sigma:
            path = os.path.join(self.model_dir, f"ircnn_sigma_{closest}_final.pth")
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state["model"])
            self.model.eval()
            self.current_sigma = closest
        return self.model

