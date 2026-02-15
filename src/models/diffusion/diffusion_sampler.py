from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .denoiser import Denoiser


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1
    deterministic: bool = False
    seed: Optional[int] = None
    use_consistency: bool = False
    sigma_min_consistency: float = 2e-3


class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: DiffusionSamplerConfig) -> None:
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser.device)

    def _get_generator(self, device: torch.device, generator: Optional[torch.Generator]) -> Optional[torch.Generator]:
        if generator is not None:
            return generator
        if not self.cfg.deterministic:
            return None
        seed = 0 if self.cfg.seed is None else int(self.cfg.seed)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return gen

    def _sample(
        self,
        prev_obs: Tensor,
        prev_act: Tensor,
        generator: Optional[torch.Generator],
        use_grad: bool,
    ) -> Tuple[Tensor, List[Tensor]]:
        device = prev_obs.device
        b, t, c, h, w = prev_obs.size()
        prev_obs = prev_obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        gen = self._get_generator(device, generator)
        x = torch.randn(b, c, h, w, device=device, generator=gen)
        trajectory = [x]
        if self.cfg.use_consistency:
            denoise_fn = self.denoiser.denoise_consistency_with_grad if use_grad else self.denoiser.denoise_consistency
        else:
            denoise_fn = self.denoiser.denoise_with_grad if use_grad else self.denoiser.denoise
        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(x, generator=gen) * self.cfg.s_noise
                x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
            if self.cfg.use_consistency:
                denoised = denoise_fn(x, sigma, prev_obs, prev_act, sigma_min=self.cfg.sigma_min_consistency)
            else:
                denoised = denoise_fn(x, sigma, prev_obs, prev_act)
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if self.cfg.order == 1 or next_sigma == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                if self.cfg.use_consistency:
                    denoised_2 = denoise_fn(
                        x_2, next_sigma * s_in, prev_obs, prev_act, sigma_min=self.cfg.sigma_min_consistency
                    )
                else:
                    denoised_2 = denoise_fn(x_2, next_sigma * s_in, prev_obs, prev_act)
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
            trajectory.append(x)
        return x, trajectory

    @torch.no_grad()
    def sample(self, prev_obs: Tensor, prev_act: Tensor, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, List[Tensor]]:
        return self._sample(prev_obs, prev_act, generator, use_grad=False)

    def sample_with_grad(
        self, prev_obs: Tensor, prev_act: Tensor, generator: Optional[torch.Generator] = None
    ) -> Tuple[Tensor, List[Tensor]]:
        return self._sample(prev_obs, prev_act, generator, use_grad=True)


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))
