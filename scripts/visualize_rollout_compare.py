#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.distributions.categorical import Categorical

from agent import Agent
from data import Batch, Dataset
from data.segment import SegmentId
from data.utils import make_segment
from metrics import get_lpips_model, lpips_distance
from models.diffusion import DiffusionSampler, DiffusionSamplerConfig, Denoiser
from utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualitative rollout comparison for Atari world models.")
    parser.add_argument("--game", type=str, default="Breakout")
    parser.add_argument("--rollout-mode", type=str, choices=["aligned", "closed_loop_free"], default="aligned")
    parser.add_argument("--sequence-id", type=int, default=0)
    parser.add_argument("--sequence-ids", type=str, default=None)
    parser.add_argument("--num-sequences", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=str, default="1,5,10,15")
    parser.add_argument("--teacher-steps", type=int, default=3)
    parser.add_argument("--student-steps", type=int, default=1)
    parser.add_argument("--student-ckpt-root", type=str, default="outputs/distill")
    parser.add_argument("--student-run-id", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default="dataset/atari3/{game}")
    parser.add_argument("--output-dir", type=str, default="results/qualitative")
    parser.add_argument("--use-teacher-reference-for-diff", action="store_true")
    parser.add_argument("--diff-mode", type=str, choices=["abs", "heatmap"], default="abs")
    parser.add_argument("--policy", type=str, choices=["pretrained", "random"], default="pretrained")
    parser.add_argument("--cell-size", type=int, default=96)
    parser.add_argument("--font-scale", type=float, default=0.6)
    parser.add_argument("--gif-fps", type=int, default=6)
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


def resolve_agent_cfg(cfg_agent, cfg_env):
    container = OmegaConf.create({"agent": cfg_agent, "env": cfg_env})
    OmegaConf.resolve(container)
    return container.agent


def resolve_dataset_path(dataset_dir: str, game_base: str) -> Path:
    p = Path(dataset_dir.format(game=game_base))
    if (p / "info.pt").is_file():
        return p
    if (p / "train" / "info.pt").is_file():
        return p / "train"
    return p


def build_sampler(denoiser: Denoiser, steps: int, seed: int) -> DiffusionSampler:
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
        deterministic=True,
        seed=seed,
    )
    return DiffusionSampler(denoiser, cfg)


def load_student_denoiser(path: Path, teacher: Denoiser) -> Denoiser:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    student = Denoiser(teacher.cfg)
    student.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith("denoiser.")})
    return student


def to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().clamp(-1, 1)
    x = x.add(1).div(2).mul(255).byte()
    return x.permute(1, 2, 0).cpu().numpy()


def diff_to_uint8(pred: torch.Tensor, ref: torch.Tensor, mode: str) -> np.ndarray:
    diff = (pred - ref).abs().clamp(0, 2) / 2.0
    diff = diff.mul(255).byte()
    img = diff.permute(1, 2, 0).cpu().numpy()
    if mode == "heatmap":
        try:
            import matplotlib.cm as cm
        except Exception:
            return img
        gray = img.mean(axis=2) / 255.0
        colored = cm.get_cmap("magma")(gray)[:, :, :3]
        return (colored * 255).astype(np.uint8)
    return img


def parse_frames(arg: str) -> List[int]:
    return [int(x) for x in arg.split(",") if x.strip()]


def pick_sequence_ids(dataset: Dataset, args: argparse.Namespace) -> List[int]:
    if args.sequence_ids:
        return [int(x) for x in args.sequence_ids.split(",") if x.strip()]
    if args.num_sequences is not None:
        return list(range(min(args.num_sequences, dataset.num_episodes)))
    return [args.sequence_id]


def build_segment(dataset: Dataset, episode_id: int, seq_len: int, seed: int) -> Batch:
    length = int(dataset.lengths[episode_id])
    if length < seq_len:
        raise ValueError(f"Episode {episode_id} too short: {length} < {seq_len}")
    start = 0
    max_start = length - seq_len
    if max_start > 0:
        rng = np.random.RandomState(seed)
        start = int(rng.randint(0, max_start + 1))
    seg = make_segment(dataset.load_episode(episode_id), SegmentId(episode_id, start, start + seq_len), should_pad=True)
    batch = Batch(
        obs=seg.obs.unsqueeze(0),
        act=seg.act.unsqueeze(0),
        rew=seg.rew.unsqueeze(0),
        end=seg.end.unsqueeze(0),
        trunc=seg.trunc.unsqueeze(0),
        mask_padding=seg.mask_padding.unsqueeze(0),
        info=[seg.info],
        segment_ids=[seg.id],
    )
    return batch


def rollout_with_actions(
    sampler: DiffusionSampler,
    prev_obs: torch.Tensor,
    prev_act: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    obs_hist = prev_obs.clone()
    act_hist = prev_act.clone()
    preds: List[torch.Tensor] = []
    for t in range(actions.size(1)):
        act = actions[:, t]
        act_hist[:, -1] = act
        next_obs, _ = sampler.sample(obs_hist, act_hist)
        preds.append(next_obs)
        obs_hist = obs_hist.roll(-1, dims=1)
        obs_hist[:, -1] = next_obs
        act_hist = act_hist.roll(-1, dims=1)
        act_hist[:, -1] = act
    return torch.stack(preds, dim=1)


def rollout_closed_loop_actions(
    sampler: DiffusionSampler,
    prev_obs: torch.Tensor,
    prev_act: torch.Tensor,
    policy,
    num_actions: int,
    horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    obs_hist = prev_obs.clone()
    act_hist = prev_act.clone()
    preds: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    hx = torch.zeros(prev_obs.size(0), policy.lstm_dim, device=prev_obs.device) if policy is not None else None
    cx = torch.zeros(prev_obs.size(0), policy.lstm_dim, device=prev_obs.device) if policy is not None else None
    for _ in range(horizon):
        if policy is None:
            act = torch.randint(low=0, high=num_actions, size=(prev_obs.size(0),), device=prev_obs.device)
        else:
            logits, _, (hx, cx) = policy.predict_act_value(obs_hist[:, -1], (hx, cx))
            act = Categorical(logits=logits).sample()
        actions.append(act)
        act_hist[:, -1] = act
        next_obs, _ = sampler.sample(obs_hist, act_hist)
        preds.append(next_obs)
        obs_hist = obs_hist.roll(-1, dims=1)
        obs_hist[:, -1] = next_obs
        act_hist = act_hist.roll(-1, dims=1)
        act_hist[:, -1] = act
    return torch.stack(preds, dim=1), torch.stack(actions, dim=1)


def make_grid(
    rows: List[List[np.ndarray]],
    col_labels: List[str],
    row_labels: List[str],
    title: str,
    cell_size: int,
    font_scale: float,
) -> Image.Image:
    font = ImageFont.load_default()
    cell_h = cell_size
    cell_w = cell_size
    label_h = int(14 * font_scale)
    title_h = int(16 * font_scale)
    grid_h = title_h + label_h + len(rows) * (cell_h + label_h)
    grid_w = label_h + len(col_labels) * (cell_w + label_h)

    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((label_h, 2), title, fill=(0, 0, 0), font=font)

    for c, label in enumerate(col_labels):
        x = label_h + c * (cell_w + label_h)
        y = title_h
        draw.text((x + 2, y + 2), label, fill=(0, 0, 0), font=font)

    for r, row in enumerate(rows):
        y = title_h + label_h + r * (cell_h + label_h)
        draw.text((2, y + 2), row_labels[r], fill=(0, 0, 0), font=font)
        for c, img in enumerate(row):
            x = label_h + c * (cell_w + label_h)
            im = Image.fromarray(img).resize((cell_w, cell_h), resample=Image.NEAREST)
            canvas.paste(im, (x, y + label_h))
    return canvas


def save_gif(frames: List[Image.Image], out_path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except Exception:
        print("[warn] imageio not available; GIF will be skipped.")
        return
    imgs = [np.array(f) for f in frames]
    imageio.mimsave(out_path, imgs, fps=fps)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OmegaConf.register_new_resolver("eval", eval)

    game_base = normalize_game_name(args.game)
    path_agent_cfg, path_env_cfg = download_atari_config()
    cfg_env = OmegaConf.load(path_env_cfg)
    cfg_env.train.id = cfg_env.test.id = f"{game_base}NoFrameskip-v4"
    cfg_agent = resolve_agent_cfg(OmegaConf.load(path_agent_cfg), cfg_env)

    # Teacher agent + policy
    tmp_env = __import__("envs").make_atari_env(num_envs=1, device=device, **cfg_env.train)
    num_actions = int(tmp_env.num_actions)
    teacher = Agent(instantiate(cfg_agent, num_actions=num_actions)).to(device).eval()
    teacher.load(download_atari_teacher(game_base))

    # Student
    student_path = None
    if args.student_run_id is not None:
        candidate = Path(args.student_ckpt_root) / game_base / args.student_run_id / "checkpoints" / "student_1step.pt"
        if candidate.is_file():
            student_path = candidate
    else:
        candidates = sorted((Path(args.student_ckpt_root) / game_base).glob("*/checkpoints/student_1step.pt"))
        if candidates:
            student_path = candidates[-1]
    if student_path is None:
        raise FileNotFoundError(f"No student checkpoint found for {game_base} under {args.student_ckpt_root}")

    student = load_student_denoiser(student_path, teacher.denoiser).to(device).eval()

    # Dataset
    dataset_path = resolve_dataset_path(args.dataset_dir, game_base)
    dataset = Dataset(dataset_path)
    dataset.load_from_default_path()
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset at {dataset_path} is empty.")

    frame_indices = parse_frames(args.frames)
    horizon = max(frame_indices)

    seq_ids = pick_sequence_ids(dataset, args)
    lpips_model, _ = get_lpips_model(device)

    for seq_id in seq_ids:
        out_dir = Path(args.output_dir) / f"seq_{seq_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        n = teacher.denoiser.cfg.inner_model.num_steps_conditioning
        seq_len = n + horizon if args.rollout_mode == "aligned" else n
        batch = build_segment(dataset, seq_id, seq_len, seed=args.seed)
        batch = batch.to(device)

        prev_obs = batch.obs[:, :n]
        prev_act = batch.act[:, :n]

        teacher3_sampler = build_sampler(teacher.denoiser, args.teacher_steps, seed=args.seed)
        teacher1_sampler = build_sampler(teacher.denoiser, 1, seed=args.seed)
        student_sampler = build_sampler(student, args.student_steps, seed=args.seed)

        compared_models = []
        diff_reference = None
        source_mode = "on_the_fly_inference"
        action_source = None

        if args.rollout_mode == "aligned":
            actions = batch.act[:, n - 1 : n - 1 + horizon]
            gt = batch.obs[:, n : n + horizon]
            teacher3 = rollout_with_actions(teacher3_sampler, prev_obs, prev_act, actions)
            teacher1 = rollout_with_actions(teacher1_sampler, prev_obs, prev_act, actions)
            student1 = rollout_with_actions(student_sampler, prev_obs, prev_act, actions)
            diff_reference = "gt"
            compared_models = ["gt", "teacher3", "teacher1", "student1"]
            action_source = "dataset_actions_aligned"
        else:
            policy = teacher.actor_critic if args.policy == "pretrained" else None
            teacher3, actions = rollout_closed_loop_actions(
                teacher3_sampler, prev_obs, prev_act, policy, num_actions, horizon
            )
            teacher1 = rollout_with_actions(teacher1_sampler, prev_obs, prev_act, actions)
            student1 = rollout_with_actions(student_sampler, prev_obs, prev_act, actions)
            gt = None
            diff_reference = "teacher3" if args.use_teacher_reference_for_diff else None
            compared_models = ["teacher3", "teacher1", "student1"]
            action_source = "policy_on_teacher3" if args.policy == "pretrained" else "random_policy_on_teacher3"

        # Save individual frames
        for t in range(horizon):
            if gt is not None:
                Image.fromarray(to_uint8(gt[0, t])).save(frames_dir / f"gt_t{t+1}.png")
            Image.fromarray(to_uint8(teacher3[0, t])).save(frames_dir / f"teacher3_t{t+1}.png")
            Image.fromarray(to_uint8(teacher1[0, t])).save(frames_dir / f"teacher1_t{t+1}.png")
            Image.fromarray(to_uint8(student1[0, t])).save(frames_dir / f"student1_t{t+1}.png")
            if args.rollout_mode == "aligned":
                diff = diff_to_uint8(student1[0, t], gt[0, t], args.diff_mode)
                Image.fromarray(diff).save(frames_dir / f"diff_s1_gt_t{t+1}.png")
            elif args.use_teacher_reference_for_diff:
                diff = diff_to_uint8(student1[0, t], teacher3[0, t], args.diff_mode)
                Image.fromarray(diff).save(frames_dir / f"diff_s1_t3_t{t+1}.png")

        # Build grid for selected frames
        rows = []
        row_labels = []
        for t in frame_indices:
            idx = t - 1
            row_labels.append(f"t+{t}")
            cols = []
            col_labels = []
            if gt is not None:
                cols.append(to_uint8(gt[0, idx]))
                col_labels.append("GT")
            cols.append(to_uint8(teacher3[0, idx]))
            col_labels.append(f"T{args.teacher_steps}")
            cols.append(to_uint8(teacher1[0, idx]))
            col_labels.append("T1")
            cols.append(to_uint8(student1[0, idx]))
            col_labels.append("S1")
            if gt is not None:
                cols.append(diff_to_uint8(student1[0, idx], gt[0, idx], args.diff_mode))
                col_labels.append("Diff(S1-GT)")
            elif args.use_teacher_reference_for_diff:
                cols.append(diff_to_uint8(student1[0, idx], teacher3[0, idx], args.diff_mode))
                col_labels.append("Diff(S1-T3)")
            rows.append(cols)

        title = f"{game_base} | mode={args.rollout_mode} | seq={seq_id} | seed={args.seed}"
        grid = make_grid(rows, col_labels, row_labels, title, args.cell_size, args.font_scale)
        grid_path = out_dir / f"grid_frames_t{'_t'.join(str(x) for x in frame_indices)}.png"
        grid.save(grid_path)

        # GIF from full horizon
        gif_frames = []
        for t in range(horizon):
            row_labels = [f"t+{t+1}"]
            cols = []
            col_labels = []
            if gt is not None:
                cols.append(to_uint8(gt[0, t]))
                col_labels.append("GT")
            cols.append(to_uint8(teacher3[0, t]))
            col_labels.append(f"T{args.teacher_steps}")
            cols.append(to_uint8(teacher1[0, t]))
            col_labels.append("T1")
            cols.append(to_uint8(student1[0, t]))
            col_labels.append("S1")
            if gt is not None:
                cols.append(diff_to_uint8(student1[0, t], gt[0, t], args.diff_mode))
                col_labels.append("Diff(S1-GT)")
            elif args.use_teacher_reference_for_diff:
                cols.append(diff_to_uint8(student1[0, t], teacher3[0, t], args.diff_mode))
                col_labels.append("Diff(S1-T3)")
            grid_frame = make_grid([cols], col_labels, row_labels, title, args.cell_size, args.font_scale)
            gif_frames.append(grid_frame)
        save_gif(gif_frames, out_dir / "rollout_compare.gif", fps=args.gif_fps)

        # Metadata
        metadata = {
            "game": game_base,
            "rollout_mode": args.rollout_mode,
            "sequence_id": seq_id,
            "seed": args.seed,
            "compared_models": compared_models,
            "diff_reference": diff_reference,
            "frame_indices": frame_indices,
            "value_range_assumption": "pred/gt in [-1,1] -> uint8 0..255",
            "source_mode": source_mode,
            "aligned_definition": "fixed initial history + dataset action sequence (open-loop) for horizon steps",
            "frame_index_convention": "t+1 corresponds to index 0 in rollout tensors",
            "diff_space": "abs(pred-ref) in [-1,1] space, normalized to 0..255",
            "action_source": action_source,
            "rollout_horizon": horizon,
            "diff_mode": args.diff_mode,
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # Optional LPIPS sanity (no crash if missing)
        if lpips_model is None:
            continue
        _ = lpips_distance(lpips_model, teacher3[:, 0], student1[:, 0]).mean().item()

    print(f"[qual] wrote outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
