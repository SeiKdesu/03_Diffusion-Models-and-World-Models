#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-horizon rollout curves.")
    parser.add_argument("--input-root", type=str, default="results/eval_rollout")
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/eval_rollout/plots")
    parser.add_argument("--games", type=str, default="Breakout,Pong,Seaquest")
    parser.add_argument("--modes", type=str, default="aligned,closed_loop_free")
    parser.add_argument(
        "--metrics",
        type=str,
        default="aligned:lpips_to_gt,psnr_to_gt,ssim_to_gt closed_loop_free:temporal_lpips_pred,pixel_delta_norm_pred",
    )
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show-seeds", action="store_true")
    return parser.parse_args()


def parse_list(arg: str) -> List[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def parse_metrics_spec(spec: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    parts = [p for p in spec.split(" ") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        mode, metrics = part.split(":", 1)
        out[mode.strip()] = [m.strip() for m in metrics.split(",") if m.strip()]
    return out


def normalize_mode(mode: str) -> str:
    m = mode.strip().lower()
    if m in ("aligned", "closed_loop_free"):
        return m
    if m in ("closed_loop", "closedloop", "closed-loop"):
        return "closed_loop_free"
    return m


def parse_bool(val: Optional[str]) -> Optional[bool]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def to_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def to_int(val: Optional[str]) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(float(val))
    except Exception:
        return None


def label_for(model_name: str, steps: Optional[int]) -> str:
    if steps is None:
        return model_name
    return f"{model_name}_step{steps}"


def is_gt_metric(metric: str) -> bool:
    return metric.endswith("_to_gt")


def read_summary_csv(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            game = row.get("game")
            model_name = row.get("model_name") or row.get("model")
            steps = to_int(row.get("steps"))
            mode = normalize_mode(row.get("rollout_mode", ""))
            horizon = to_int(row.get("horizon"))
            metric = row.get("metric")
            mean = to_float(row.get("mean"))
            std = to_float(row.get("std"))
            count = to_int(row.get("count")) or 0
            if not all([game, model_name, mode, horizon, metric]):
                continue
            rows.append(
                {
                    "game": game,
                    "mode": mode,
                    "model": label_for(model_name, steps),
                    "horizon": horizon,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "n": count,
                }
            )
    return rows


def read_metrics_per_horizon(input_root: Path) -> Tuple[List[Dict[str, object]], Dict[Tuple, Dict[int, float]]]:
    rows: List[Dict[str, object]] = []
    seed_series: Dict[Tuple, Dict[int, float]] = defaultdict(dict)
    files = list(input_root.rglob("metrics_per_horizon.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics_per_horizon.csv under {input_root}")
    for path in files:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                game = row.get("game")
                model_name = row.get("model_name") or row.get("model")
                steps = to_int(row.get("steps"))
                mode = normalize_mode(row.get("rollout_mode", ""))
                horizon = to_int(row.get("horizon"))
                seed = to_int(row.get("seed"))
                has_gt = parse_bool(row.get("has_gt"))
                if not all([game, model_name, mode, horizon]) or seed is None:
                    continue
                for metric, val in row.items():
                    if metric in (
                        "game",
                        "model_name",
                        "model",
                        "steps",
                        "seed",
                        "rollout_mode",
                        "horizon",
                        "has_gt",
                        "reference_type",
                    ):
                        continue
                    v = to_float(val)
                    if v is None or math.isnan(v):
                        continue
                    if is_gt_metric(metric) and has_gt is False:
                        continue
                    rows.append(
                        {
                            "game": game,
                            "mode": mode,
                            "model": label_for(model_name, steps),
                            "horizon": horizon,
                            "metric": metric,
                            "value": v,
                            "seed": seed,
                        }
                    )
                    seed_series[(game, mode, label_for(model_name, steps), metric, seed)][horizon] = v
    return rows, seed_series


def aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, int, str], List[float]] = defaultdict(list)
    for row in rows:
        key = (row["game"], row["mode"], row["model"], row["horizon"], row["metric"])
        grouped[key].append(float(row["value"]))
    out: List[Dict[str, object]] = []
    for (game, mode, model, horizon, metric), values in grouped.items():
        arr = np.array(values, dtype=np.float64)
        out.append(
            {
                "game": game,
                "mode": mode,
                "model": model,
                "horizon": horizon,
                "metric": metric,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=0)),
                "n": int(arr.size),
            }
        )
    return out


def write_aggregated_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["game", "rollout_mode", "model_name", "horizon", "metric", "mean", "std", "n"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "game": row["game"],
                    "rollout_mode": row["mode"],
                    "model_name": row["model"],
                    "horizon": row["horizon"],
                    "metric": row["metric"],
                    "mean": row["mean"],
                    "std": row["std"],
                    "n": row["n"],
                }
            )


def plot_metric(
    rows: List[Dict[str, object]],
    seed_series: Optional[Dict[Tuple, Dict[int, float]]],
    game: str,
    mode: str,
    metric: str,
    models: Optional[List[str]],
    output_dir: Path,
    show_seeds: bool,
    dpi: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib required for plotting: {exc}")

    data = [r for r in rows if r["game"] == game and r["mode"] == mode and r["metric"] == metric]
    if models is not None:
        data = [r for r in data if r["model"] in models]
    if not data:
        return

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    for model in sorted({r["model"] for r in data}):
        series = sorted([r for r in data if r["model"] == model], key=lambda x: x["horizon"])
        horizons = [r["horizon"] for r in series]
        means = [r["mean"] for r in series]
        stds = [r["std"] for r in series]
        ax.plot(horizons, means, label=model)
        ax.fill_between(horizons, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)

        if show_seeds and seed_series is not None:
            for (g, m, mdl, met, seed), series_map in seed_series.items():
                if g != game or m != mode or mdl != model or met != metric:
                    continue
                xs = sorted(series_map.keys())
                ys = [series_map[x] for x in xs]
                ax.plot(xs, ys, alpha=0.25, linewidth=0.8)

    ax.set_title(f"{game} | {mode} | {metric}")
    ax.set_xlabel("horizon")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{game}_{metric}_vs_horizon.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    games = parse_list(args.games)
    modes = [normalize_mode(m) for m in parse_list(args.modes)]
    metrics_by_mode = parse_metrics_spec(args.metrics)
    models = parse_list(args.models) if args.models else None

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        if not summary_path.is_file():
            raise FileNotFoundError(f"summary.csv not found at {summary_path}")
        rows = read_summary_csv(summary_path)
        seed_series = None
    else:
        rows_raw, seed_series = read_metrics_per_horizon(Path(args.input_root))
        if args.max_horizon is not None:
            rows_raw = [r for r in rows_raw if int(r["horizon"]) <= args.max_horizon]
            if seed_series is not None:
                seed_series = {
                    k: {h: v for h, v in series.items() if h <= args.max_horizon}
                    for k, series in seed_series.items()
                }
        rows = aggregate_rows(rows_raw)

    output_dir = Path(args.output_dir)
    write_aggregated_csv(rows, output_dir / "aggregated_curves.csv")

    for mode in modes:
        metrics = metrics_by_mode.get(mode, [])
        for game in games:
            for metric in metrics:
                out_mode_dir = output_dir / mode
                plot_metric(rows, seed_series, game, mode, metric, models, out_mode_dir, args.show_seeds, args.dpi)

    # README
    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Rollout horizon curves",
                "",
                f"- modes: {', '.join(modes)}",
                f"- games: {', '.join(games)}",
                "- aggregated data in aggregated_curves.csv",
            ]
        )
    )
    print(f"[plot] wrote plots under {output_dir}")


if __name__ == "__main__":
    main()
