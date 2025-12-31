import os, math
import torch
try:
    # Tentative pour quand on lance depuis le Benchmark
    from IRCNN.IRCNN import IRCNNSigmaMap
except ModuleNotFoundError:
    # Repli pour quand on lance le fichier en direct
    from IRCNN import IRCNNSigmaMap
from PIL import Image
import torchvision.transforms.functional as TF

def psnr_torch(x, y, eps=1e-8):
    # x,y in [0,1], shape (1,3,H,W)
    mse = torch.mean((x - y) ** 2).item()
    return 10.0 * math.log10(1.0 / (mse + eps))


@torch.no_grad()
def denoise_image_pil(model, img_pil, sigma, device):
    """
    img_pil: PIL RGB
    sigma: noise level in [0..50], expressed in pixel space (0-255 scale)
    """
    y = TF.to_tensor(img_pil)  # (3,H,W) in [0,1]
    sigma_map = torch.full((1, y.shape[1], y.shape[2]), sigma / 255.0)
    inp = torch.cat([y, sigma_map], dim=0).unsqueeze(0).to(device)  # (1,4,H,W)
    out = model(inp).squeeze(0).cpu()  # (3,H,W)
    return TF.to_pil_image(out)


@torch.no_grad()
def test_mode_A_clean_to_noisy(
    clean_path,
    ckpt_path,
    out_dir="test_outputs",
    sigma=25.0,
    seed=0
):
    """
    Mode A: start from clean image -> add noise -> denoise -> compute PSNR vs clean.
    Saves: clean.png, noisy.png, denoised.png
    """
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = IRCNNSigmaMap().to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    clean_pil = Image.open(clean_path).convert("RGB")
    clean = TF.to_tensor(clean_pil).unsqueeze(0)  # (1,3,H,W)

    # create noisy
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(clean.shape, generator=g) * (sigma / 255.0)
    noisy = (clean + noise)

    # denoise
    sigma_map = torch.full((1, 1, clean.shape[2], clean.shape[3]), sigma / 255.0)
    inp = torch.cat([noisy, sigma_map], dim=1).to(device)  # (1,4,H,W)
    den = model(inp).cpu()
    
    noisy_vis = noisy.clamp(0.0, 1.0)
    den_vis   = den.clamp(0.0, 1.0) 

    # PSNR
    psnr_noisy = psnr_torch(noisy_vis, clean)
    psnr_den = psnr_torch(den_vis, clean)

    # save
    TF.to_pil_image(clean.squeeze(0)).save(os.path.join(out_dir, "clean.png"))
    TF.to_pil_image(noisy_vis.squeeze(0)).save(os.path.join(out_dir, f"noisy_sigma{int(sigma)}.png"))
    TF.to_pil_image(den_vis.squeeze(0)).save(os.path.join(out_dir, f"denoised_sigma{int(sigma)}.png"))

    print("Saved to:", out_dir)
    print(f"PSNR noisy  : {psnr_noisy:.2f} dB")
    print(f"PSNR denoised: {psnr_den:.2f} dB")


@torch.no_grad()
def test_mode_B_real_noisy(
    noisy_path,
    ckpt_path,
    out_dir="test_outputs",
    sigma=25.0
):
    """
    Mode B: you have a noisy image (no GT) -> denoise and save.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = IRCNNSigmaMap().to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    noisy_pil = Image.open(noisy_path).convert("RGB")
    den_pil = denoise_image_pil(model, noisy_pil, sigma, device)

    base = os.path.splitext(os.path.basename(noisy_path))[0]
    den_pil.save(os.path.join(out_dir, f"{base}_denoised_sigma{int(sigma)}.png"))
    print("Saved to:", out_dir)

'''
test_mode_A_clean_to_noisy(
    clean_path=r"./BSDS300/images/test/102061.jpg",
    ckpt_path=r"./weights_ircnn_sigmap/ircnn_sigmap_final.pth",
    out_dir="results_IRCNN_denoise",
    sigma=20
)
'''

@torch.no_grad()
def denoise_ircnn(
    clean_path,
    ckpt_path=r"./weights_ircnn_sigmap/ircnn_sigmap_final.pth",
    out_dir=r"./benchmark/denoise",
    sigma=40,
    seed=0
):
    """
    Mode A: start from clean image -> add noise -> denoise -> compute PSNR vs clean.
    Saves: clean.png, noisy.png, denoised.png
    """
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = IRCNNSigmaMap().to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    clean_pil = Image.open(clean_path).convert("RGB")
    clean = TF.to_tensor(clean_pil).unsqueeze(0)  # (1,3,H,W)

    # create noisy
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(clean.shape, generator=g) * (sigma / 255.0)
    noisy = (clean + noise)

    # denoise
    sigma_map = torch.full((1, 1, clean.shape[2], clean.shape[3]), sigma / 255.0)
    inp = torch.cat([noisy, sigma_map], dim=1).to(device)  # (1,4,H,W)
    den = model(inp).cpu()
    
    noisy_vis = noisy.clamp(0.0, 1.0)
    den_vis   = den.clamp(0.0, 1.0) 

    # PSNR
    psnr_noisy = psnr_torch(noisy_vis, clean)
    psnr_den = psnr_torch(den_vis, clean)

    # save
    img_id = os.path.splitext(os.path.basename(clean_path))[0]
    img_dir = os.path.join(out_dir, img_id)
    os.makedirs(img_dir, exist_ok=True)
    TF.to_pil_image(den_vis.squeeze(0)).save(os.path.join(img_dir, f"denoised_ircnn.png"))

    return psnr_noisy, psnr_den


'''print(denoise_ircnn(
    clean_path=r"./BSDS300/images/test/102061.jpg",
    sigma=20
))'''