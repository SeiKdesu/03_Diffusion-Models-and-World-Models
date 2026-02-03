import copy
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import trange

from agent import Agent
from data import Batch, BatchSampler, Dataset, collate_segments_to_batch, CSGOHdf5Dataset
from models.diffusion.denoiser import Denoiser, DenoiserConfig
from utils import configure_opt, save_with_backup, set_seed


OmegaConf.register_new_resolver("eval", eval)


def _append_dims(x: Tensor, target_ndim: int) -> Tensor:
    return x.reshape(x.shape + (1,) * (target_ndim - x.ndim))


def _karras_sigmas_pair(
    batch_size: int,
    *,
    num_scales: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """consistency_models公式実装の t, t2 のサンプリングと同等。"""
    assert num_scales >= 2
    indices = torch.randint(0, num_scales - 1, (batch_size,), device=device)

    t = sigma_max ** (1 / rho) + indices / (num_scales - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    )
    t = t**rho

    t2 = sigma_max ** (1 / rho) + (indices + 1) / (num_scales - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    )
    t2 = t2**rho

    return t, t2


def _get_weightings(weight_schedule: str, snrs: Tensor, sigma_data: float) -> Tensor:
    if weight_schedule == "snr":
        return snrs
    if weight_schedule == "snr+1":
        return snrs + 1
    if weight_schedule == "karras":
        return snrs + 1.0 / (sigma_data**2)
    if weight_schedule == "truncated-snr":
        return torch.clamp(snrs, min=1.0)
    if weight_schedule == "uniform":
        return torch.ones_like(snrs)
    raise ValueError(f"Unknown weight_schedule: {weight_schedule}")


@dataclass
class ConsistencyConditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor
    c_noise_cond: Tensor


def _compute_consistency_conditioners(
    denoiser: Denoiser,
    *,
    sigma: Tensor,
    sigma_min: float,
    sigma_cond: Optional[Tensor],
) -> ConsistencyConditioners:
    """Consistency distillation用の boundary condition 付き preconditioning。

    consistency_models公式実装の get_scalings_for_boundary_condition を、
    このリポジトリの Denoiser(EDM系 + sigma_offset_noise) に合わせて移植したもの。
    """

    sigma_eff = (sigma**2 + denoiser.cfg.sigma_offset_noise**2).sqrt()
    sigma_min_eff = (torch.tensor(sigma_min, device=sigma.device) ** 2 + denoiser.cfg.sigma_offset_noise**2).sqrt()

    c_in = 1 / (sigma_eff**2 + denoiser.cfg.sigma_data**2).sqrt()
    c_skip = denoiser.cfg.sigma_data**2 / ((sigma_eff - sigma_min_eff) ** 2 + denoiser.cfg.sigma_data**2)
    c_out = (sigma_eff - sigma_min_eff) * denoiser.cfg.sigma_data / (sigma_eff**2 + denoiser.cfg.sigma_data**2).sqrt()

    c_noise = sigma_eff.log() / 4
    c_noise_cond = sigma_cond.log() / 4 if sigma_cond is not None else torch.zeros_like(c_noise)

    return ConsistencyConditioners(
        c_in=_append_dims(c_in, 4),
        c_out=_append_dims(c_out, 4),
        c_skip=_append_dims(c_skip, 4),
        c_noise=_append_dims(c_noise, 1),
        c_noise_cond=_append_dims(c_noise_cond, 1),
    )


def _predict_denoised_consistency(
    denoiser: Denoiser,
    *,
    noisy_next_obs: Tensor,
    sigma: Tensor,
    sigma_min: float,
    sigma_cond: Optional[Tensor],
    obs: Tensor,
    act: Optional[Tensor],
) -> Tensor:
    cs = _compute_consistency_conditioners(denoiser, sigma=sigma, sigma_min=sigma_min, sigma_cond=sigma_cond)
    rescaled_obs = obs / denoiser.cfg.sigma_data
    rescaled_noise = noisy_next_obs * cs.c_in
    model_output = denoiser.inner_model(rescaled_noise, cs.c_noise, cs.c_noise_cond, rescaled_obs, act)
    denoised = cs.c_out * model_output + cs.c_skip * noisy_next_obs
    return denoised


@torch.no_grad()
def _heun_step_teacher(
    teacher: Denoiser,
    *,
    x_t: Tensor,
    t: Tensor,
    t2: Tensor,
    sigma_cond: Optional[Tensor],
    cond_obs: Tensor,
    cond_act: Optional[Tensor],
) -> Tensor:
    """Teacher denoiser を使って (t -> t2) を1ステップ進める。"""
    dims = x_t.ndim
    den = teacher.denoise(x_t, t, sigma_cond, cond_obs, cond_act)
    d = (x_t - den) / _append_dims(t, dims)
    x_euler = x_t + d * _append_dims(t2 - t, dims)

    den2 = teacher.denoise(x_euler, t2, sigma_cond, cond_obs, cond_act)
    d2 = (x_euler - den2) / _append_dims(t2, dims)

    x_t2 = x_t + (d + d2) * _append_dims((t2 - t) / 2, dims)
    return x_t2


@torch.no_grad()
def _euler_step_teacher(
    teacher: Denoiser,
    *,
    x_t: Tensor,
    t: Tensor,
    t2: Tensor,
    sigma_cond: Optional[Tensor],
    cond_obs: Tensor,
    cond_act: Optional[Tensor],
) -> Tensor:
    dims = x_t.ndim
    den = teacher.denoise(x_t, t, sigma_cond, cond_obs, cond_act)
    d = (x_t - den) / _append_dims(t, dims)
    return x_t + d * _append_dims(t2 - t, dims)


def _prepare_single_step_conditioning(
    denoiser: Denoiser,
    batch: Batch,
    *,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor]:
    """Denoiser.forward の1ステップ版。

    Returns:
      x0 (target frame), cond_obs (prev frames flattened), cond_act, mask
    """
    batch = batch.to(device)
    b, t, c, h, w = batch.obs.size()
    n = denoiser.cfg.inner_model.num_steps_conditioning

    assert t >= n + 1

    if denoiser.is_upsampler:
        H = denoiser.cfg.upsampling_factor * h
        W = denoiser.cfg.upsampling_factor * w
        all_obs = torch.stack([x["full_res"] for x in batch.info]).to(device)
        low_res = (
            F.interpolate(
                batch.obs.reshape(b * t, c, h, w),
                scale_factor=denoiser.cfg.upsampling_factor,
                mode="bicubic",
            )
            .reshape(b, t, c, H, W)
            .to(device)
        )
        assert all_obs.shape == low_res.shape
        prev_obs = all_obs[:, 0:n].reshape(b, n * c, H, W)
        cond_act = None
        x0 = all_obs[:, n]
        mask = batch.mask_padding[:, n]
        prev_obs = torch.cat((prev_obs, low_res[:, n]), dim=1)
    else:
        all_obs = batch.obs
        prev_obs = all_obs[:, 0:n].reshape(b, n * c, h, w)
        cond_act = batch.act[:, 0:n]
        x0 = all_obs[:, n]
        mask = batch.mask_padding[:, n]

    return x0, prev_obs, cond_act, mask


def _save_component_checkpoint(path: Path, component_name: str, model: Denoiser) -> None:
    sd = OrderedDict({f"{component_name}.{k}": v.cpu() for k, v in model.state_dict().items()})
    save_with_backup(sd, path)


@hydra.main(config_path="../config", config_name="distill_consistency", version_base="1.3")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    set_seed(int(cfg.training.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_full_res = None
    num_actions = cfg.env.num_actions
    if cfg.env.train.id == "csgo":
        assert cfg.env.path_data_low_res is not None
        assert cfg.env.path_data_full_res is not None
        dataset_full_res = CSGOHdf5Dataset(Path(cfg.env.path_data_full_res))

    num_workers = int(cfg.training.num_workers_data_loaders)
    use_manager = bool(cfg.training.cache_in_ram) and (num_workers > 0)
    p = Path(cfg.static_dataset.path) if cfg.static_dataset.path is not None else Path("dataset")
    train_dataset = Dataset(p / "train", dataset_full_res, "train_dataset", bool(cfg.training.cache_in_ram), use_manager)
    train_dataset.load_from_default_path()

    component: str = cfg.component
    if component not in ("denoiser", "upsampler"):
        raise ValueError("component must be 'denoiser' or 'upsampler'")


    agent_cfg = instantiate(cfg.agent, num_actions=num_actions)


    student_cfg: DenoiserConfig = copy.deepcopy(getattr(agent_cfg, component))
    student = Denoiser(student_cfg).to(device)
    student.train()

    teacher: Optional[Denoiser]
    if cfg.teacher.path_to_ckpt is not None:
        teacher_agent = Agent(copy.deepcopy(agent_cfg)).to(device)
        teacher_agent.load(
            path_to_ckpt=Path(cfg.teacher.path_to_ckpt),
            load_denoiser=(component == "denoiser"),
            load_upsampler=(component == "upsampler"),
            load_rew_end_model=False,
            load_actor_critic=False,
        )
        teacher = getattr(teacher_agent, component)
        teacher.eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)

        # distillationのときは student を teacher で初期化(収束が速い)
        student.load_state_dict(teacher.state_dict(), strict=True)
        print(f"[distill] teacher loaded from: {cfg.teacher.path_to_ckpt}")
    else:
        teacher = None
        print("[sanity] teacher.path_to_ckpt is null -> running teacher-less consistency training (for smoke test)")

    target = copy.deepcopy(student).to(device)
    target.eval()
    for p_ in target.parameters():
        p_.requires_grad_(False)

    # Data loader
    n = student.cfg.inner_model.num_steps_conditioning
    seq_length = n + 1 + int(cfg.sequence.num_autoregressive_steps)
    bs = BatchSampler(train_dataset, 0, 1, cfg.training.batch_size, seq_length, sample_weights=None)
    dl = DataLoader(
        dataset=train_dataset,
        batch_sampler=bs,
        collate_fn=collate_segments_to_batch,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=(device.type == "cuda"),
    )
    it = iter(dl)

    opt = configure_opt(
        student,
        lr=float(cfg.optimizer.lr),
        weight_decay=float(cfg.optimizer.weight_decay),
        eps=float(cfg.optimizer.eps),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp.enabled) and device.type == "cuda")

    out_dir = Path(cfg.training.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma_min = float(cfg.consistency.sigma_min)
    sigma_max = float(cfg.consistency.sigma_max)
    rho = float(cfg.consistency.rho)

    num_scales = int(cfg.consistency.num_scales)
    ema_rate = float(cfg.ema.rate)
    s_cond = float(cfg.consistency.s_cond)

    for step in trange(int(cfg.training.total_steps), desc="Consistency distillation"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        x0, cond_obs, cond_act, mask = _prepare_single_step_conditioning(student, batch, device=device)
        b = x0.shape[0]

        # prev_obsに軽いノイズを入れる(rolloutの s_cond を模倣)
        if s_cond > 0:
            sigma_cond = torch.full((b,), fill_value=s_cond, device=device)
            cond_obs = student.apply_noise(cond_obs, sigma_cond, sigma_offset_noise=0.0)
        else:
            sigma_cond = None

        t, t2 = _karras_sigmas_pair(
            b,
            num_scales=num_scales,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            device=device,
        )

        noise = torch.randn_like(x0)
        x_t = x0 + noise * _append_dims(t, x0.ndim)

        with torch.no_grad():
            if teacher is None:
                x_t2 = x0 + noise * _append_dims(t2, x0.ndim)
            else:
                if str(cfg.consistency.solver).lower() == "heun":
                    x_t2 = _heun_step_teacher(
                        teacher,
                        x_t=x_t,
                        t=t,
                        t2=t2,
                        sigma_cond=sigma_cond,
                        cond_obs=cond_obs,
                        cond_act=cond_act,
                    )
                elif str(cfg.consistency.solver).lower() == "euler":
                    x_t2 = _euler_step_teacher(
                        teacher,
                        x_t=x_t,
                        t=t,
                        t2=t2,
                        sigma_cond=sigma_cond,
                        cond_obs=cond_obs,
                        cond_act=cond_act,
                    )
                else:
                    raise ValueError("cfg.consistency.solver must be euler|heun")

        with torch.no_grad():
            distiller_target = _predict_denoised_consistency(
                target,
                noisy_next_obs=x_t2,
                sigma=t2,
                sigma_min=sigma_min,
                sigma_cond=sigma_cond,
                obs=cond_obs,
                act=cond_act,
            )

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            distiller = _predict_denoised_consistency(
                student,
                noisy_next_obs=x_t,
                sigma=t,
                sigma_min=sigma_min,
                sigma_cond=sigma_cond,
                obs=cond_obs,
                act=cond_act,
            )

            snrs = t.pow(-2)
            weights = _get_weightings(str(cfg.consistency.weight_schedule), snrs, student.cfg.sigma_data)

            if str(cfg.consistency.loss_norm).lower() == "l1":
                diffs = (distiller - distiller_target).abs()
                per_ex = diffs.flatten(1).mean(dim=1)
            elif str(cfg.consistency.loss_norm).lower() == "l2":
                diffs = (distiller - distiller_target).pow(2)
                per_ex = diffs.flatten(1).mean(dim=1)
            else:
                raise ValueError("cfg.consistency.loss_norm must be l1|l2")


            valid = mask.float()
            denom = valid.sum().clamp(min=1.0)
            loss = ((per_ex * weights) * valid).sum() / denom

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg.optimizer.max_grad_norm))
        scaler.step(opt)
        scaler.update()


        with torch.no_grad():
            for p_t, p_s in zip(target.parameters(), student.parameters()):
                p_t.mul_(ema_rate).add_(p_s, alpha=1 - ema_rate)

        if step % int(cfg.training.log_every) == 0:
            print({"step": step, "loss": float(loss.detach().cpu())})

        if step > 0 and step % int(cfg.training.save_every) == 0:
            _save_component_checkpoint(out_dir / f"{component}_student_step_{step:07d}.pt", component, student)
            _save_component_checkpoint(out_dir / f"{component}_ema_step_{step:07d}.pt", component, target)

    _save_component_checkpoint(out_dir / f"{component}_student_final.pt", component, student)
    _save_component_checkpoint(out_dir / f"{component}_ema_final.pt", component, target)


if __name__ == "__main__":
    main()
