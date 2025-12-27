# SWIN_denoise.py
import os, math
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from SWIN import SwinIRSigmaMap


def psnr_torch(x, y, eps=1e-8):
    mse = torch.mean((x - y) ** 2).item()
    return 10.0 * math.log10(1.0 / (mse + eps))


def save_img01(t_bchw, path):
    t = t_bchw.detach().cpu().clamp(0, 1).squeeze(0)
    TF.to_pil_image(t).save(path)


def add_awgn(clean_bchw, sigma_pixels, seed=0):
    # deterministic noise on CPU
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(clean_bchw.shape, generator=g) * (float(sigma_pixels) / 255.0)
    return clean_bchw + noise


def build_inp(noisy_bchw, sigma_pixels):
    B, C, H, W = noisy_bchw.shape
    sigma_map = torch.full((B, 1, H, W), float(sigma_pixels) / 255.0,
                           device=noisy_bchw.device, dtype=noisy_bchw.dtype)
    return torch.cat([noisy_bchw, sigma_map], dim=1)


def remap_plain_blocks_to_layer(sd):
    """
    Checkpoint keys: groups.0.blocks.0....
    Model expects   : groups.0.layer.blocks.0....
    """
    keys = list(sd.keys())
    has_plain = any(k.startswith("groups.0.blocks.") for k in keys)
    has_layer = any(k.startswith("groups.0.layer.blocks.") for k in keys)

    if has_plain and not has_layer:
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("groups."):
                parts = k.split(".")
                # groups, i, blocks, ...
                if len(parts) >= 3 and parts[2] == "blocks":
                    k2 = ".".join([parts[0], parts[1], "layer"] + parts[2:])
                    new_sd[k2] = v
                else:
                    new_sd[k] = v
            else:
                new_sd[k] = v
        return new_sd

    return sd


@torch.no_grad()
def denoise_clean_to_noisy(
    clean_path,
    ckpt_path,
    out_dir="test_outputs_denoise_swinir",
    sigma=25.0,
    seed=0,
    embed_dim=64,
    window_size=8,
    depths=(2, 2, 2, 2),
    num_heads=(4, 4, 4, 4),
    mlp_ratio=2.0,
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # build model
    model = SwinIRSigmaMap(
        in_chans=4,
        out_chans=3,
        embed_dim=embed_dim,
        window_size=window_size,
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    ).to(device).eval()

    # load checkpoint
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = state["model"] if (isinstance(state, dict) and "model" in state) else state

    # REMAP KEYS HERE
    sd = remap_plain_blocks_to_layer(sd)

    # now strict works
    model.load_state_dict(sd, strict=True)

    # load clean
    clean_pil = Image.open(clean_path).convert("RGB")
    clean = TF.to_tensor(clean_pil).unsqueeze(0)  # (1,3,H,W)

    # add noise (no clamp)
    noisy = add_awgn(clean, sigma_pixels=sigma, seed=seed)
    inp = build_inp(noisy.to(device), sigma_pixels=sigma)

    # denoise
    den = model(inp).clamp(0, 1).cpu()

    # psnr
    noisy_clamped = noisy.clamp(0, 1)
    psnr_noisy = psnr_torch(noisy_clamped, clean)
    psnr_den = psnr_torch(den, clean)

    # save
    save_img01(clean, os.path.join(out_dir, "clean.png"))
    save_img01(noisy_clamped, os.path.join(out_dir, f"noisy_sigma{int(sigma)}.png"))
    save_img01(den, os.path.join(out_dir, f"denoised_sigma{int(sigma)}.png"))

    print("Saved to:", out_dir)
    print(f"PSNR noisy   : {psnr_noisy:.2f} dB")
    print(f"PSNR denoised: {psnr_den:.2f} dB")


if __name__ == "__main__":
    denoise_clean_to_noisy(
        clean_path=r"./BSDS300/images/test/24077.jpg",
        ckpt_path=r"./weights_swinir_sigmap/swinir_sigmap_final.pth",
        out_dir="results_SWIN_denoise",
        sigma=70.0,
    )
