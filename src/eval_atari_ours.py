#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader

from agent import Agent
from coroutines.collector import make_collector, NumToCollect
from data import BatchSampler, Dataset, collate_segments_to_batch
from envs import WorldModelEnv, WorldModelEnvConfig, make_atari_env
from metrics import get_lpips_model, lpips_distance, psnr, ssim
from models.actor_critic import ActorCritic, ActorCriticConfig, ActorCriticLossConfig
from models.diffusion import DiffusionSampler, DiffusionSamplerConfig, Denoiser
from utils import set_seed


DEFAULT_GAMES = "BreakoutNoFrameskip-v4,PongNoFrameskip-v4,SeaquestNoFrameskip-v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 1-step / few-step Atari world models.")
    parser.add_argument("--games", type=str, default=DEFAULT_GAMES)
    parser.add_argument("--steps", type=str, default="1,2,4,8,teacher")
    parser.add_argument("--teacher-steps", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--dataset-dir", type=str, default="dataset/atari3/{game}")
    parser.add_argument("--collect-steps", type=int, default=0)
    parser.add_argument("--pred-samples", type=int, default=256)
    parser.add_argument("--pred-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--policy", type=str, choices=["pretrained", "random"], default="pretrained")
    parser.add_argument("--student-ckpt-root", type=str, default="outputs/distill")
    parser.add_argument("--student-run-id", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="results/atari3")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rl-eval", action="store_true")
    parser.add_argument("--rl-updates", type=int, default=500)
    parser.add_argument("--rl-batch-size", type=int, default=8)
    parser.add_argument("--rl-episodes", type=int, default=20)
    parser.add_argument("--rl-lr", type=float, default=3e-4)
    parser.add_argument("--rl-seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
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


def build_sampler(denoiser: Denoiser, steps: int, deterministic: bool, seed: int) -> DiffusionSampler:
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
        deterministic=deterministic,
        seed=seed,
    )
    return DiffusionSampler(denoiser, cfg)


def load_student_denoiser(path: Path, teacher: Denoiser) -> Denoiser:
    sd = torch.load(path, map_location="cpu")
    student = Denoiser(teacher.cfg)
    student.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith("denoiser.")})
    return student


def rollout_wm(
    env: WorldModelEnv,
    policy: Optional[ActorCritic],
    horizon: int,
    episodes: int,
    num_actions: int,
) -> Tuple[float, float, float, float]:
    total_frames = episodes * horizon
    drift_vals: List[float] = []
    collapsed = 0
    start = time.perf_counter()
    for _ in range(episodes):
        obs, _ = env.reset()
        hx = torch.zeros(env.num_envs, policy.lstm_dim, device=env.device) if policy is not None else None
        cx = torch.zeros(env.num_envs, policy.lstm_dim, device=env.device) if policy is not None else None
        collapsed_frames = 0
        prev = obs
        for _ in range(horizon):
            if policy is None:
                act = torch.randint(low=0, high=num_actions, size=(env.num_envs,), device=env.device)
            else:
                logits, _, (hx, cx) = policy.predict_act_value(obs, (hx, cx))
                act = Categorical(logits=logits).sample()
            next_obs, _, _, _, _ = env.step(act)
            if torch.isnan(next_obs).any() or torch.isinf(next_obs).any() or next_obs.std() < 0.01:
                collapsed_frames += 1
            drift_vals.append(F.mse_loss(next_obs, prev).item())
            prev = next_obs
            obs = next_obs
        if collapsed_frames / max(1, horizon) > 0.5:
            collapsed += 1
    duration = time.perf_counter() - start
    hz = total_frames / duration if duration > 0 else 0.0
    ms_per_frame = 1000.0 / hz if hz > 0 else 0.0
    collapse_rate = collapsed / max(1, episodes)
    drift = float(np.mean(drift_vals)) if drift_vals else 0.0
    return hz, ms_per_frame, collapse_rate, drift


def prediction_metrics(
    denoiser: Denoiser,
    steps: int,
    dataset: Dataset,
    device: torch.device,
    pred_samples: int,
    pred_batch_size: int,
    deterministic: bool,
    seed: int,
    lpips_model,
) -> Tuple[float, float, float]:
    if len(dataset) == 0:
        return float("nan"), float("nan"), float("nan")
    n = denoiser.cfg.inner_model.num_steps_conditioning
    bs = BatchSampler(dataset, 0, 1, pred_batch_size, n + 1, sample_weights=None, can_sample_beyond_end=False)
    dl = DataLoader(
        dataset,
        batch_sampler=bs,
        collate_fn=collate_segments_to_batch,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    sampler = build_sampler(denoiser, steps, deterministic, seed)
    num_batches = max(1, pred_samples // pred_batch_size)
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []

    it = iter(dl)
    for i in range(num_batches):
        batch = next(it).to(device)
        prev_obs = batch.obs[:, :n]
        prev_act = batch.act[:, :n]
        gt = batch.obs[:, n]
        gen = torch.Generator(device=device).manual_seed(seed + i)
        pred, _ = sampler.sample(prev_obs, prev_act, generator=gen)
        psnr_vals.append(psnr(pred, gt).mean().item())
        ssim_vals.append(ssim(pred, gt).mean().item())
        if lpips_model is not None:
            lpips_vals.append(lpips_distance(lpips_model, pred, gt).mean().item())

    psnr_mean = float(np.mean(psnr_vals)) if psnr_vals else float("nan")
    ssim_mean = float(np.mean(ssim_vals)) if ssim_vals else float("nan")
    lpips_mean = float(np.mean(lpips_vals)) if lpips_vals else float("nan")
    return psnr_mean, ssim_mean, lpips_mean


def rl_train_and_eval(
    denoiser: Denoiser,
    rew_end_model,
    actor_cfg: ActorCriticConfig,
    env_cfg,
    dataset: Dataset,
    device: torch.device,
    updates: int,
    batch_size: int,
    lr: float,
    episodes: int,
    seed: int,
    deterministic: bool,
    steps: int,
) -> Tuple[float, float]:
    torch.manual_seed(seed)
    n = denoiser.cfg.inner_model.num_steps_conditioning
    bs = BatchSampler(dataset, 0, 1, batch_size, n, sample_weights=None, can_sample_beyond_end=False)
    dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch, num_workers=0)

    wm_cfg = WorldModelEnvConfig(
        horizon=50,
        num_batches_to_preload=8,
        diffusion_sampler=DiffusionSamplerConfig(
            num_steps_denoising=steps,
            sigma_min=2e-3,
            sigma_max=5.0,
            rho=7,
            order=1,
            s_churn=0.0,
            s_tmin=0.0,
            s_tmax=float("inf"),
            s_noise=1.0,
            deterministic=deterministic,
            seed=seed,
        ),
    )
    wm_env = WorldModelEnv(denoiser, rew_end_model, dl, wm_cfg)

    policy = ActorCritic(actor_cfg).to(device)
    loss_cfg = ActorCriticLossConfig(
        backup_every=15,
        gamma=0.985,
        lambda_=0.95,
        weight_value_loss=1.0,
        weight_entropy_loss=0.001,
    )
    policy.setup_training(wm_env, loss_cfg)

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    for _ in range(updates):
        loss, _ = policy()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Evaluate in real env
    test_env = make_atari_env(num_envs=1, device=device, **env_cfg.test)
    policy.eval()
    returns = []
    for ep in range(episodes):
        obs, _ = test_env.reset(seed=[seed + ep])
        hx = torch.zeros(1, policy.lstm_dim, device=device)
        cx = torch.zeros(1, policy.lstm_dim, device=device)
        done = False
        ret = 0.0
        while not done:
            logits, _, (hx, cx) = policy.predict_act_value(obs, (hx, cx))
            act = Categorical(logits=logits).sample()
            obs, rew, end, trunc, _ = test_env.step(act)
            ret += rew.item()
            done = bool((end + trunc).item())
        returns.append(ret)

    return float(np.mean(returns)), float(np.std(returns))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) / run_id
    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    steps_list = [s.strip() for s in args.steps.split(",") if s.strip()]
    steps_parsed: List[int] = []
    for s in steps_list:
        steps_parsed.append(args.teacher_steps if s == "teacher" else int(s))

    lpips_model, lpips_err = get_lpips_model(device)
    if lpips_model is None:
        print(f"[eval] {lpips_err}. LPIPS metrics will be skipped.")

    world_model_rows: List[Dict[str, object]] = []
    rl_rows: List[Dict[str, object]] = []

    for game in games:
        game_base = normalize_game_name(game)
        path_agent_cfg, path_env_cfg = download_atari_config()
        cfg_agent = OmegaConf.load(path_agent_cfg)
        cfg_env = OmegaConf.load(path_env_cfg)
        cfg_env.train.id = cfg_env.test.id = f"{game_base}NoFrameskip-v4"

        path_teacher = download_atari_teacher(game_base)

        tmp_env = make_atari_env(num_envs=1, device=device, **cfg_env.train)
        num_actions = int(tmp_env.num_actions)

        teacher = Agent(instantiate(cfg_agent, num_actions=num_actions)).to(device).eval()
        teacher.load(path_teacher)

        student_path = None
        if args.student_run_id is not None:
            candidate = Path(args.student_ckpt_root) / game_base / args.student_run_id / "checkpoints" / "student_1step.pt"
            if candidate.is_file():
                student_path = candidate
        else:
            candidates = sorted((Path(args.student_ckpt_root) / game_base).glob("*/checkpoints/student_1step.pt"))
            if candidates:
                student_path = candidates[-1]

        student = None
        if student_path is not None:
            student = load_student_denoiser(student_path, teacher.denoiser).to(device).eval()
            print(f"[eval] using student ckpt: {student_path}")
        else:
            print(f"[eval] no student ckpt found for {game_base}, skipping student metrics")

        dataset_path = resolve_dataset_path(args.dataset_dir, game_base)
        dataset = Dataset(dataset_path)
        dataset.load_from_default_path()
        maybe_collect_dataset(dataset, cfg_env, teacher, args.collect_steps, device)
        dataset.load_from_default_path()

        for model_name, denoiser in [("teacher", teacher.denoiser), ("student", student)]:
            if denoiser is None:
                continue
            for steps in steps_parsed:
                sampler_cfg = DiffusionSamplerConfig(
                    num_steps_denoising=steps,
                    sigma_min=2e-3,
                    sigma_max=5.0,
                    rho=7,
                    order=1,
                    s_churn=0.0,
                    s_tmin=0.0,
                    s_tmax=float("inf"),
                    s_noise=1.0,
                    deterministic=args.deterministic,
                    seed=args.seed,
                )
                wm_cfg = WorldModelEnvConfig(
                    horizon=args.horizon,
                    num_batches_to_preload=8,
                    diffusion_sampler=sampler_cfg,
                )
                n = denoiser.cfg.inner_model.num_steps_conditioning
                bs = BatchSampler(dataset, 0, 1, 1, n, sample_weights=None, can_sample_beyond_end=False)
                dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch, num_workers=0)
                wm_env = WorldModelEnv(denoiser, teacher.rew_end_model, dl, wm_cfg)

                policy = teacher.actor_critic if args.policy == "pretrained" else None
                hz, ms_per_frame, collapse_rate, drift = rollout_wm(
                    wm_env,
                    policy,
                    args.horizon,
                    args.episodes,
                    num_actions,
                )

                psnr_val, ssim_val, lpips_val = prediction_metrics(
                    denoiser,
                    steps,
                    dataset,
                    device,
                    args.pred_samples,
                    args.pred_batch_size,
                    args.deterministic,
                    args.seed,
                    lpips_model,
                )

                world_model_rows.append(
                    {
                        "game": game_base,
                        "steps": steps,
                        "model": model_name,
                        "horizon": args.horizon,
                        "hz": hz,
                        "ms_per_frame": ms_per_frame,
                        "psnr": psnr_val,
                        "ssim": ssim_val,
                        "lpips": lpips_val,
                        "collapse_rate": collapse_rate,
                        "drift": drift,
                    }
                )

        if args.rl_eval:
            for model_name, denoiser in [("teacher", teacher.denoiser), ("student", student)]:
                if denoiser is None:
                    continue
                mean_ret, std_ret = rl_train_and_eval(
                    denoiser=denoiser,
                    rew_end_model=teacher.rew_end_model,
                    actor_cfg=instantiate(cfg_agent.actor_critic, num_actions=num_actions),
                    env_cfg=cfg_env,
                    dataset=dataset,
                    device=device,
                    updates=args.rl_updates,
                    batch_size=args.rl_batch_size,
                    lr=args.rl_lr,
                    episodes=args.rl_episodes,
                    seed=args.rl_seed,
                    deterministic=args.deterministic,
                    steps=1 if model_name == "student" else args.teacher_steps,
                )
                rl_rows.append(
                    {
                        "game": game_base,
                        "model": model_name,
                        "steps": 1 if model_name == "student" else args.teacher_steps,
                        "seed": args.rl_seed,
                        "episodes": args.rl_episodes,
                        "return_mean": mean_ret,
                        "return_std": std_ret,
                        "hns": float("nan"),
                    }
                )

    # Write CSVs
    out_root.mkdir(parents=True, exist_ok=True)
    wm_csv = out_root / "world_model_metrics.csv"
    with wm_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "game",
                "steps",
                "model",
                "horizon",
                "hz",
                "ms_per_frame",
                "psnr",
                "ssim",
                "lpips",
                "collapse_rate",
                "drift",
            ],
        )
        writer.writeheader()
        writer.writerows(world_model_rows)

    if args.rl_eval:
        rl_csv = out_root / "rl_metrics.csv"
        with rl_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["game", "model", "steps", "seed", "episodes", "return_mean", "return_std", "hns"],
            )
            writer.writeheader()
            writer.writerows(rl_rows)

    # Plots
    try:
        import matplotlib.pyplot as plt

        for game in {row["game"] for row in world_model_rows}:
            rows = [r for r in world_model_rows if r["game"] == game]
            for metric, fname in [("psnr", "quality_vs_steps"), ("lpips", "quality_vs_steps_lpips")]:
                plt.figure()
                for model in sorted({r["model"] for r in rows}):
                    xs = [r["steps"] for r in rows if r["model"] == model]
                    ys = [r[metric] for r in rows if r["model"] == model]
                    plt.plot(xs, ys, marker="o", label=model)
                plt.xlabel("Steps")
                plt.ylabel(metric.upper())
                plt.title(f"{game} {metric.upper()} vs steps")
                plt.legend()
                plt.savefig(plots_dir / f"{fname}_{game}.png", dpi=150)
                plt.close()

            plt.figure()
            for model in sorted({r["model"] for r in rows}):
                xs = [r["hz"] for r in rows if r["model"] == model]
                ys = [r["lpips"] for r in rows if r["model"] == model]
                plt.scatter(xs, ys, label=model)
            plt.xlabel("Hz")
            plt.ylabel("LPIPS")
            plt.title(f"{game} LPIPS vs speed")
            plt.legend()
            plt.savefig(plots_dir / f"quality_vs_speed_{game}.png", dpi=150)
            plt.close()

            plt.figure()
            for metric in ["collapse_rate", "drift"]:
                xs = [r["steps"] for r in rows if r["model"] == "teacher"]
                ys = [r[metric] for r in rows if r["model"] == "teacher"]
                plt.plot(xs, ys, marker="o", label=f"teacher_{metric}")
                if any(r["model"] == "student" for r in rows):
                    xs = [r["steps"] for r in rows if r["model"] == "student"]
                    ys = [r[metric] for r in rows if r["model"] == "student"]
                    plt.plot(xs, ys, marker="o", label=f"student_{metric}")
            plt.xlabel("Steps")
            plt.ylabel("Value")
            plt.title(f"{game} stability vs steps")
            plt.legend()
            plt.savefig(plots_dir / f"stability_vs_steps_{game}.png", dpi=150)
            plt.close()
    except Exception as exc:
        print(f"[eval] Plotting skipped: {exc}")

    print(f"[eval] wrote {wm_csv}")
    if args.rl_eval:
        print(f"[eval] wrote {out_root / 'rl_metrics.csv'}")


if __name__ == "__main__":
    main()
