import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_to_multiple(x: torch.Tensor, mult: int) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
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


# MLP
class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Window helpers
def window_partition(x, window_size: int):
    """
    x: (B, H, W, C)
    return windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    windows: (num_windows*B, window_size, window_size, C)
    return x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x


# Window Attention (with relative position bias)
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size: int, num_heads: int, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # relative position bias table
        # (2*Ws-1)*(2*Ws-1), nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1)*(2*window_size-1), num_heads)
        )

        # relative position index for each token inside window
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing="ij"
        ))  # (2, Ws, Ws)
        coords_flat = coords.flatten(1)  # (2, Ws*Ws)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel_coords = rel_coords.permute(1,2,0).contiguous()  # (N,N,2)
        rel_coords[:,:,0] += window_size - 1
        rel_coords[:,:,1] += window_size - 1
        rel_coords[:,:,0] *= 2*window_size - 1
        relative_position_index = rel_coords.sum(-1)  # (N,N)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        x: (num_windows*B, N, C) where N=Ws*Ws
        mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2,0,3,1,4)  # (3, B_, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, heads, N, N)

        # add relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2,0,1).contiguous()  # (heads, N, N)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            # mask: (nW, N, N)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)  # (B_, heads, N, head_dim)
        out = out.transpose(1,2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# Swin Transformer Block
class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int,int],
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_dim, dropout=drop)

        self.register_buffer("attn_mask", None, persistent=False)

    def build_mask(self, H: int, W: int, device):
        if self.shift_size == 0:
            self.attn_mask = None
            return

        ws = self.window_size
        ss = self.shift_size

        img_mask = torch.zeros((1, H, W, 1), device=device)  # (1,H,W,1)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws)  # (nW, ws, ws, 1)
        mask_windows = mask_windows.view(-1, ws*ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, N, N)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        self.attn_mask = attn_mask

    def forward(self, x, H: int, W: int):
        """
        x: (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W

        if (self.attn_mask is None) and (self.shift_size != 0):
            self.build_mask(H, W, x.device)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        else:
            shifted = x

        # windows
        ws = self.window_size
        x_windows = window_partition(shifted, ws)  # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, ws*ws, C)

        # attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, ws, ws, C)
        shifted = window_reverse(attn_windows, ws, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        else:
            x = shifted

        x = x.view(B, H*W, C)
        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


# Basic Layer (sequence of Swin blocks)
class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=2.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=(0,0),
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop
                )
            )

    def forward(self, x, H: int, W: int):
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


# Residual Swin Transformer Group (RSTB-like)
class ResidualSwinBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=2.0):
        super().__init__()
        self.layer = BasicLayer(dim, depth, num_heads, window_size=window_size, mlp_ratio=mlp_ratio)
        # a small conv in image space inside residual group
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, H: int, W: int):
        """
        x: (B, H*W, C)
        """
        shortcut = x
        x = self.layer(x, H, W)
        B, L, C = x.shape
        x_img = x.transpose(1,2).contiguous().view(B, C, H, W)
        x_img = self.conv(x_img)
        x = x_img.view(B, C, H*W).transpose(1,2).contiguous()
        return x + shortcut


# SwinIR for denoising (scale=1) with sigma-map conditioning
class SwinIRSigmaMap(nn.Module):
    """
    Input : (B,4,H,W) = noisy RGB + sigma_map
    Output: (B,3,H,W) denoised
    Strategy: predict residual, output = noisy_rgb + residual
    """
    def __init__(
        self,
        in_chans=4,
        out_chans=3,
        embed_dim=96,
        window_size=8,
        depths=(6,6,6,6),
        num_heads=(6,6,6,6),
        mlp_ratio=2.0,
        img_range=1.0
    ):
        super().__init__()
        assert len(depths) == len(num_heads)

        self.img_range = img_range
        self.window_size = window_size

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        self.groups = nn.ModuleList([
            ResidualSwinBlock(embed_dim, depth=depths[i], num_heads=num_heads[i],
                              window_size=window_size, mlp_ratio=mlp_ratio)
            for i in range(len(depths))
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # tail predicts residual on RGB
        self.conv_last = nn.Conv2d(embed_dim, out_chans, 3, 1, 1)

    def forward(self, inp):
        # pad to multiple of window_size (needed by window partition)
        x, pads = _pad_to_multiple(inp, mult=self.window_size)

        # keep noisy rgb for residual add
        noisy_rgb = x[:, :3, :, :]

        # shallow features
        fea = self.conv_first(x)  # (B, C, H, W)
        B, C, H, W = fea.shape

        # tokens
        tokens = fea.flatten(2).transpose(1,2).contiguous()  # (B, H*W, C)

        # Swin groups
        for g in self.groups:
            tokens = g(tokens, H, W)

        # norm + back to image
        tokens = self.norm(tokens)
        fea2 = tokens.transpose(1,2).contiguous().view(B, C, H, W)

        # conv after body + long residual
        fea2 = self.conv_after_body(fea2) + fea

        # predict residual and add to noisy
        res = self.conv_last(fea2)
        out = noisy_rgb + res

        out = _unpad(out, pads)
        return out
