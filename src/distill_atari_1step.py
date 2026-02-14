#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from agent import Agent
from coroutines.collector import make_collector, NumToCollect
from data import BatchSampler, Dataset, collate_segments_to_batch
from envs import make_atari_env
from metrics import get_lpips_model, lpips_distance
from models.diffusion import DiffusionSampler, DiffusionSamplerConfig, Denoiser
from utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill Atari world model to 1-step (few-step) student.")
    parser.add_argument("--game", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--dataset-dir", type=str, default="dataset/atari3/{game}")
    parser.add_argument("--collect-steps", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--teacher-steps", type=int, default=8)
    parser.add_argument("--student-steps", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--out-dir", type=str, default="outputs/distill")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--consistency-noise-std", type=float, default=0.05)
    parser.add_argument("--lpips-weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def normalize_game_name(game: str) -> str:
    if game.endswith("NoFrameskip-v4"):
        return game.replace("NoFrameskip-v4", "")
    return game


def download_atari_teacher(game_base: str) -> Path:
    return Path(hf_hub_download(repo_id="eloialonso/diamond", filename=f"atari_100k/models/{game_base}.pt"))


def download_atari_config() -> tuple[Path, Path]:
    path_agent = Path(hf_hub_download(repo_id="eloialonso/diamond", filename="atari_100k/config/agent/default.yaml"))
    path_env = Path(hf_hub_download(repo_id="eloialonso/diamond", filename="atari_100k/config/env/atari.yaml"))
    return path_agent, path_env


def resolve_agent_cfg(cfg_agent):
    container = OmegaConf.create({"agent": cfg_agent})
    OmegaConf.resolve(container)
    return container.agent


def resolve_dataset_path(dataset_dir: str, game_base: str) -> Path:
    p = Path(dataset_dir.format(game=game_base))
    if (p / "info.pt").is_file():
        return p
    if (p / "train" / "info.pt").is_file():
        return p / "train"
    return p


def maybe_collect_dataset(
    dataset: Dataset,
    env_cfg,
    agent: Agent,
    num_steps: int,
    device: torch.device,
) -> None:
    if num_steps <= 0:
        return
    if len(dataset) > 0:
        return
    dataset._directory.mkdir(parents=True, exist_ok=True)
    train_env = make_atari_env(num_envs=1, device=device, **env_cfg.train)
    collector = make_collector(train_env, agent.actor_critic, dataset, epsilon=0.0, reset_every_collect=False, verbose=True)
    collector.send(NumToCollect(steps=num_steps))
    dataset.save_to_default_path()


def build_sampler(denoiser: Denoiser, steps: int) -> DiffusionSampler:
    cfg = DiffusionSamplerConfig(
        num_steps_denoising=steps,
        sigma_min=2e-3,
        sigma_max=5.0,
        rho=7,
        order=1,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
    )
    return DiffusionSampler(denoiser, cfg)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    game_base = normalize_game_name(args.game)
    OmegaConf.register_new_resolver("eval", eval)
    path_agent_cfg, path_env_cfg = download_atari_config()
    cfg_agent = resolve_agent_cfg(OmegaConf.load(path_agent_cfg))
    cfg_env = OmegaConf.load(path_env_cfg)
    cfg_env.train.id = cfg_env.test.id = f"{game_base}NoFrameskip-v4"

    path_teacher = download_atari_teacher(game_base)

    # Build env for action space
    tmp_env = make_atari_env(num_envs=1, device=device, **cfg_env.train)
    num_actions = int(tmp_env.num_actions)

    # Teacher agent
    teacher = Agent(instantiate(cfg_agent, num_actions=num_actions)).to(device).eval()
    teacher.load(path_teacher)

    # Student denoiser initialized from teacher
    student = Denoiser(teacher.denoiser.cfg).to(device)
    student.load_state_dict(teacher.denoiser.state_dict())
    student.train()

    # Dataset
    dataset_path = resolve_dataset_path(args.dataset_dir, game_base)
    dataset = Dataset(dataset_path)
    dataset.load_from_default_path()
    maybe_collect_dataset(dataset, cfg_env, teacher, args.collect_steps, device)
    dataset.load_from_default_path()
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset at {dataset_path} is empty. Use --collect-steps to gather data.")

    n = teacher.denoiser.cfg.inner_model.num_steps_conditioning
    bs = BatchSampler(dataset, 0, 1, args.batch_size, n + 1, sample_weights=None, can_sample_beyond_end=False)
    dl = DataLoader(
        dataset,
        batch_sampler=bs,
        collate_fn=collate_segments_to_batch,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    data_iter = iter(dl)

    teacher_sampler = build_sampler(teacher.denoiser, args.teacher_steps)
    student_sampler = build_sampler(student, args.student_steps)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)

    lpips_model, lpips_err = get_lpips_model(device)
    if lpips_model is None and args.lpips_weight > 0:
        print(f"[distill] {lpips_err}. LPIPS loss will be skipped.")
        args.lpips_weight = 0.0

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / game_base / run_id / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.iters + 1):
        batch = next(data_iter).to(device)
        prev_obs = batch.obs[:, :n]
        prev_act = batch.act[:, :n]

        seed = args.seed + step
        gen_teacher = torch.Generator(device=device).manual_seed(seed)
        gen_student = torch.Generator(device=device).manual_seed(seed)

        with torch.no_grad():
            teacher_next, _ = teacher_sampler.sample(prev_obs, prev_act, generator=gen_teacher)

        student_next, _ = student_sampler.sample_with_grad(prev_obs, prev_act, generator=gen_student)

        loss = F.mse_loss(student_next, teacher_next)

        if args.consistency_weight > 0:
            noisy_prev_obs = prev_obs + torch.randn_like(prev_obs) * args.consistency_noise_std
            gen_student_2 = torch.Generator(device=device).manual_seed(seed + 1)
            student_next_noisy, _ = student_sampler.sample_with_grad(noisy_prev_obs, prev_act, generator=gen_student_2)
            loss = loss + args.consistency_weight * F.mse_loss(student_next_noisy, teacher_next)

        if args.lpips_weight > 0 and lpips_model is not None:
            with torch.no_grad():
                lp = lpips_distance(lpips_model, student_next, teacher_next).mean()
            loss = loss + args.lpips_weight * lp

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % 50 == 0:
            print(f"[distill] step {step}/{args.iters} loss={loss.item():.4f}")

        if step % args.save_every == 0 or step == args.iters:
            ckpt_path = out_dir / "student_1step.pt"
            sd = {f"denoiser.{k}": v.cpu() for k, v in student.state_dict().items()}
            torch.save(sd, ckpt_path)
            print(f"[distill] saved {ckpt_path}")


if __name__ == "__main__":
    main()
