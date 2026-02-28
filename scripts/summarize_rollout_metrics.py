#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-horizon rollout metrics (mean/std across seeds).")
    parser.add_argument("--input-root", type=str, default="results/eval_rollout")
    parser.add_argument("--output", type=str, default="results/eval_rollout/summary.csv")
    return parser.parse_args()


def is_number(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    files = list(root.rglob("metrics_per_horizon.csv"))
    if not files:
        raise SystemExit(f"No metrics_per_horizon.csv found under {root}")

    metric_fields = [
        "psnr_to_gt",
        "ssim_to_gt",
        "lpips_to_gt",
        "psnr_to_teacher",
        "ssim_to_teacher",
        "lpips_to_teacher",
        "temporal_lpips_pred",
        "pixel_delta_norm_pred",
    ]

    grouped: Dict[Tuple[str, str, str, str, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for path in files:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    row.get("game", ""),
                    row.get("model_name", ""),
                    row.get("steps", ""),
                    row.get("rollout_mode", ""),
                    row.get("horizon", ""),
                    row.get("reference_type", ""),
                )
                for m in metric_fields:
                    v = row.get(m, "")
                    if v == "" or not is_number(v):
                        continue
                    fv = float(v)
                    if math.isnan(fv):
                        continue
                    grouped[key][m].append(fv)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "game",
                "model_name",
                "steps",
                "rollout_mode",
                "horizon",
                "reference_type",
                "metric",
                "mean",
                "std",
                "count",
            ],
        )
        writer.writeheader()
        for key, metrics in grouped.items():
            for metric, values in metrics.items():
                if not values:
                    continue
                arr = np.array(values, dtype=np.float64)
                writer.writerow(
                    {
                        "game": key[0],
                        "model_name": key[1],
                        "steps": key[2],
                        "rollout_mode": key[3],
                        "horizon": key[4],
                        "reference_type": key[5],
                        "metric": metric,
                        "mean": float(arr.mean()),
                        "std": float(arr.std(ddof=0)),
                        "count": int(arr.size),
                    }
                )

    print(f"[summary] wrote {out_path}")


if __name__ == "__main__":
    main()
