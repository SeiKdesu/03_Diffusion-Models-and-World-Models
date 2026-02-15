from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def psnr(x: Tensor, y: Tensor, data_range: float = 2.0, eps: float = 1e-8) -> Tensor:
    """Peak Signal-to-Noise Ratio per image (N,). Inputs in [-1, 1] by default."""
    assert x.shape == y.shape and x.ndim == 4
    mse = F.mse_loss(x, y, reduction="none")
    mse = mse.flatten(1).mean(1)
    return 20 * torch.log10(torch.tensor(data_range, device=x.device)) - 10 * torch.log10(mse + eps)


def ssim(
    x: Tensor,
    y: Tensor,
    data_range: float = 2.0,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-8,
) -> Tensor:
    """Structural Similarity (SSIM) per image (N,). Inputs in [-1, 1] by default."""
    assert x.shape == y.shape and x.ndim == 4
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    pad = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x2
    sigma_y2 = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y2
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / (den + eps)
    return ssim_map.mean(dim=(1, 2, 3))
