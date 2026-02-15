from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def get_lpips_model(device: torch.device) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    try:
        import lpips  # type: ignore
    except Exception as exc:
        return None, f"LPIPS unavailable ({exc})"

    model = lpips.LPIPS(net="alex").to(device).eval()
    return model, None


@torch.no_grad()
def lpips_distance(model: torch.nn.Module, x: Tensor, y: Tensor) -> Tensor:
    """LPIPS per image (N,). Inputs should be in [-1, 1]."""
    return model(x, y).view(-1)
