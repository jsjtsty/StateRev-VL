#!/usr/bin/env bash
# One-command runner for the 4 stale-origin mechanism experiments.
# Target machine: 1x A100 80G.
#
# Setup (once) on the target machine:
#   conda create -n staterev python=3.12 -y
#   conda activate staterev
#   pip install "transformers==5.15.1" torch av numpy
#   # (match the reference env: transformers 5.15.1, torch 2.13.0, av 18.1.0)
#
# Data that must be copied to this repo first (gitignored upstream):
#   dataset/vetbench/cup/*.mp4                                   (50 videos, ~64 MB)
#   models/Qwen3-VL-8B-Instruct/                                 (~17 GB)
#   outputs/vetbench/composition_analysis_v1/transformers_behavior.csv
#   outputs/vetbench/state_retention_analysis_v1/state_retention_pairs.csv
#   outputs/vetbench/hidden_state_probe/hidden_states.npz        (145 MB)
#
# Stages (any subset via --stages, comma-separated):
#   check   preflight only (python env, GPU, data files, model dir)
#   build   rebuild CPU manifests (temporal + rescue; seconds)
#   exp1    nonlinear retention probe        (CPU, hours)
#   exp2    revision rescue inference        (GPU, ~1-2 h)
#   exp3    generation dynamics tracing      (GPU, ~1-2 h)
#   all     build + exp1 (background CPU) + exp2 + exp3 (foreground GPU)
#
# Usage:
#   bash scripts/run_stale_origin_experiments.sh                    # all
#   bash scripts/run_stale_origin_experiments.sh --stages check
#   bash scripts/run_stale_origin_experiments.sh --stages exp2,exp3 \
#       --model-dir /path/to/Qwen3-VL-8B-Instruct
#
# All outputs land in outputs/vetbench/stale_origin_analysis_v1/
# per-stage logs in outputs/vetbench/stale_origin_analysis_v1/run_logs/.

set -euo pipefail

PYTHON="${PYTHON:-python}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
STAGES="${STAGES:-all}"
# cap CPU workers: 80 forked torch workers on a shared box froze it
N_JOBS="${N_JOBS:-$(( $(nproc) < 16 ? $(nproc) : 16 ))}"
N_PERMS=100

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="outputs/vetbench/stale_origin_analysis_v1"
LOG_DIR="$OUT_DIR/run_logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/runner.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN_LOG"; }
fail() { log "FATAL: $*"; exit 1; }

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --python)    PYTHON="$2"; shift 2 ;;
    --stages)    STAGES="$2"; shift 2 ;;
    --n-jobs)    N_JOBS="$2"; shift 2 ;;
    --n-perms)   N_PERMS="$2"; shift 2 ;;
    -h|--help)   usage ;;
    *) fail "unknown argument: $1 (see --help)" ;;
  esac
done

have() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
if have all; then STAGES="build,exp1,exp2,exp3"; fi

log "=== stale-origin experiments runner ==="
log "repo=$REPO_ROOT python=$PYTHON model_dir=$MODEL_DIR stages=$STAGES n_jobs=$N_JOBS"

# ---------------------------------------------------------------- preflight
log "--- preflight: python environment ---"
"$PYTHON" - <<'EOF' | tee -a "$RUN_LOG"
import sys, numpy, torch, av, transformers
from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
print(f"python={sys.version.split()[0]} numpy={numpy.__version__} "
      f"torch={torch.__version__} av={av.__version__} "
      f"transformers={transformers.__version__}")
ref = (5, 15, 1)
cur = tuple(int(x) for x in transformers.__version__.split(".")[:3])
if cur != ref:
    print(f"WARNING: transformers {transformers.__version__} != verified "
          f"{ref[0]}.{ref[1]}.{ref[2]} (model loading may fail with "
          f"weight-shape mismatch; fix: pip install 'transformers==5.15.1')")
print(f"cuda available={torch.cuda.is_available()} "
      f"devices={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"gpu0={torch.cuda.get_device_name(0)}")
EOF

log "--- preflight: data files ---"
n_vid=$(ls dataset/vetbench/cup/*.mp4 2>/dev/null | wc -l)
[[ "$n_vid" -eq 50 ]] || fail "expected 50 videos in dataset/vetbench/cup/, found $n_vid"
log "videos: $n_vid"
for f in \
  outputs/vetbench/composition_analysis_v1/transformers_behavior.csv \
  outputs/vetbench/state_retention_analysis_v1/state_retention_pairs.csv \
  outputs/vetbench/hidden_state_probe/hidden_states.npz; do
  [[ -f "$f" ]] || fail "missing $f (copy it from the analysis machine)"
  log "ok: $f ($(du -h "$f" | cut -f1))"
done
"$PYTHON" - <<'EOF' | tee -a "$RUN_LOG"
import numpy as np
d = np.load("outputs/vetbench/hidden_state_probe/hidden_states.npz")
keys = list(d.keys())
n_t0 = sum(k.endswith("_t0") or k.endswith("_t0__before") or "_t0" in k for k in keys)
print(f"npz keys={len(keys)} (t0 present: {n_t0 > 0})")
EOF

if have exp2 || have exp3; then
  log "--- preflight: GPU + model ---"
  "$PYTHON" -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 1, 'no CUDA device visible'" \
    || fail "torch cannot see a CUDA device (check nvidia-smi / CUDA_VISIBLE_DEVICES)"
  log "gpu: $("$PYTHON" -c 'import torch; print(torch.cuda.get_device_name(0))')"
  [[ -d "$MODEL_DIR" && -f "$MODEL_DIR/config.json" ]] \
    || fail "model dir not found: $MODEL_DIR (use --model-dir)"
  [[ $(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l) -ge 1 ]] \
    || fail "no *.safetensors in $MODEL_DIR"
  log "model dir ok: $MODEL_DIR ($(du -sh --dereference "$MODEL_DIR" 2>/dev/null | cut -f1))"
fi

if [[ "$STAGES" == "check" ]]; then
  log "preflight complete (check-only mode)"
  exit 0
fi

# ------------------------------------------------------------------ stages
run_logged() {  # $1 = stage name, rest = command
  local name="$1"; shift
  log ">>> $name: $*"
  "$@" 2>&1 | tee "$LOG_DIR/$name.log" | tail -5
  log "<<< $name done"
}

EXP1_PID=""
if have exp1 && have exp2; then
  # exp1 is pure CPU: run it in the background while GPU stages proceed
  log ">>> exp1 (background): nonlinear retention, n_jobs=$N_JOBS"
  ( "$PYTHON" scripts/nonlinear_state_retention.py \
      --n-jobs "$N_JOBS" --n-perms "$N_PERMS" \
      > "$LOG_DIR/exp1.log" 2>&1 ) &
  EXP1_PID=$!
  log "exp1 pid=$EXP1_PID (log: $LOG_DIR/exp1.log)"
fi

if have build; then
  run_logged build_temporal "$PYTHON" scripts/build_temporal_intervention_manifest.py
  run_logged build_rescue "$PYTHON" scripts/run_revision_rescue.py --mode build
fi

if have exp1 && ! have exp2; then
  run_logged exp1 "$PYTHON" scripts/nonlinear_state_retention.py \
    --n-jobs "$N_JOBS" --n-perms "$N_PERMS"
fi

if have exp2; then
  run_logged exp2_rescue "$PYTHON" scripts/run_revision_rescue.py \
    --mode run --model-dir "$MODEL_DIR"
fi

if have exp3; then
  run_logged exp3_trace "$PYTHON" scripts/trace_generation_dynamics.py \
    --mode run --model-dir "$MODEL_DIR"
fi

if [[ -n "$EXP1_PID" ]]; then
  log "waiting for background exp1 (pid $EXP1_PID) ..."
  if ! wait "$EXP1_PID"; then
    tail -20 "$LOG_DIR/exp1.log"
    fail "exp1 (background) failed; see $LOG_DIR/exp1.log"
  fi
  log "exp1 done (log: $LOG_DIR/exp1.log)"
fi

log "=== all requested stages finished ==="
log "artifacts in $OUT_DIR/:"
ls -1 "$OUT_DIR" | tee -a "$RUN_LOG"
