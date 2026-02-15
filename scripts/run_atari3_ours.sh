#!/usr/bin/env bash
set -euo pipefail

GAMES=("BreakoutNoFrameskip-v4" "PongNoFrameskip-v4" "SeaquestNoFrameskip-v4")
GAMES_CSV="BreakoutNoFrameskip-v4,PongNoFrameskip-v4,SeaquestNoFrameskip-v4"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TEACHER_STEPS="${TEACHER_STEPS:-3}"
DISTILL_ITERS="${DISTILL_ITERS:-500}"
COLLECT_STEPS="${COLLECT_STEPS:-5000}"

echo "[run] RUN_ID=${RUN_ID}"

for GAME in "${GAMES[@]}"; do
  GAME_BASE="${GAME/NoFrameskip-v4/}"
  echo "[run] distill ${GAME} -> ${GAME_BASE}"
  python src/distill_atari_1step.py \
    --game "${GAME}" \
    --teacher-steps "${TEACHER_STEPS}" \
    --student-steps 1 \
    --iters "${DISTILL_ITERS}" \
    --collect-steps "${COLLECT_STEPS}" \
    --dataset-dir "dataset/atari3/${GAME_BASE}" \
    --run-id "${RUN_ID}"
done

echo "[run] world-model evaluation"
python src/eval_atari_ours.py \
  --games "${GAMES_CSV}" \
  --steps "1,2,4,8,teacher" \
  --teacher-steps "${TEACHER_STEPS}" \
  --student-ckpt-root "outputs/distill" \
  --student-run-id "${RUN_ID}" \
  --dataset-dir "dataset/atari3/{game}" \
  --out-dir "results/atari3" \
  --run-id "${RUN_ID}" \
  --deterministic

python - <<PY
import csv
from pathlib import Path

path = Path("results/atari3") / "${RUN_ID}" / "world_model_metrics.csv"
if not path.is_file():
    print(f"[summary] missing {path}")
    raise SystemExit(0)

rows = list(csv.DictReader(path.open()))
print("\\n[summary] world_model_metrics.csv")
print("game,model,steps,hz,psnr,lpips,collapse_rate,drift")
for r in rows:
    print("{game},{model},{steps},{hz:.2f},{psnr:.2f},{lpips},{collapse_rate:.2f},{drift:.4f}".format(
        game=r["game"],
        model=r["model"],
        steps=r["steps"],
        hz=float(r["hz"]),
        psnr=float(r["psnr"]) if r["psnr"] != "nan" else float("nan"),
        lpips=r["lpips"],
        collapse_rate=float(r["collapse_rate"]),
        drift=float(r["drift"]),
    ))
PY
