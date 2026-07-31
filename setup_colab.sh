#!/usr/bin/env bash
# setup_colab.sh -- idempotent environment rebuild for xgap.
#
# Run this at the top of every Colab session, before importing anything else:
#   !bash /content/drive/MyDrive/xgap/setup_colab.sh
#
# IMPORTANT: this script sets MUJOCO_GL=egl for the *subprocesses it spawns*
# (pip installs, the nvidia-smi/nproc probes). It CANNOT set it inside the
# Jupyter kernel's own Python process -- Colab's `!` shell is a child process,
# env vars set there do not propagate back. Your notebook's FIRST cell, before
# any other import (torch, lerobot, mujoco, numpy-that-imports-mujoco, etc.)
# MUST be:
#
#   import os
#   os.environ["MUJOCO_GL"] = "egl"
#
# Only after that cell should you `import lerobot` / run scripts that touch LIBERO.

set -euo pipefail

export MUJOCO_GL=egl

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
# XGAP_DRIVE_ROOT: where this repo (code/configs/outputs) lives persistently.
# Override with `XGAP_DRIVE_ROOT=/some/path bash setup_colab.sh` if mounted elsewhere.
XGAP_DRIVE_ROOT="${XGAP_DRIVE_ROOT:-/content/drive/MyDrive/xgap}"

# HF_HOME: model/config downloads (checkpoints, small). Persisted to Drive so
# they are not re-downloaded every session.
export HF_HOME="${HF_HOME:-${XGAP_DRIVE_ROOT}/.hf_cache}"

# LIBERO demo dataset cache: kept LOCAL to the Colab VM disk (/content), not Drive.
# Rationale (fill in after first real run -- see README.md "Dataset cache decision"):
#   Drive is a network mount; large sequential reads of many small parquet/video
#   shards over it are frequently the actual bottleneck in Colab, worse than a
#   cold re-download to local SSD if the dataset is small enough. We do not know
#   yet whether HuggingFaceVLA/libero is small enough for Drive to be fine --
#   measure `du -sh` and first-load wall time in Colab and record the verdict in
#   README.md before changing this default.
export LEROBOT_DATASET_CACHE="${LEROBOT_DATASET_CACHE:-/content/xgap_dataset_cache}"

mkdir -p "${HF_HOME}" "${LEROBOT_DATASET_CACHE}"
mkdir -p "${XGAP_DRIVE_ROOT}/outputs" "${XGAP_DRIVE_ROOT}/logs"

echo "[xgap setup] XGAP_DRIVE_ROOT=${XGAP_DRIVE_ROOT}"
echo "[xgap setup] HF_HOME=${HF_HOME}"
echo "[xgap setup] LEROBOT_DATASET_CACHE=${LEROBOT_DATASET_CACHE}"

# ---------------------------------------------------------------------------
# 1. Detect state BEFORE touching anything (for the restart-required banner
#    and the torch-reinstall check below).
# ---------------------------------------------------------------------------
TORCH_VER_BEFORE="$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT_INSTALLED')"
LEROBOT_WAS_INSTALLED="no"
python3 -c 'import lerobot' 2>/dev/null && LEROBOT_WAS_INSTALLED="yes" || true

# ---------------------------------------------------------------------------
# 2. Install lerobot[libero], pinned (see requirements.txt for the pin + why).
#    Idempotent: pip no-ops if the pinned version is already satisfied.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -q -r "${SCRIPT_DIR}/requirements.txt"

TORCH_VER_AFTER="$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT_INSTALLED')"

if [ "${TORCH_VER_BEFORE}" != "${TORCH_VER_AFTER}" ]; then
  echo ""
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "!! WARNING: torch changed during install: ${TORCH_VER_BEFORE} -> ${TORCH_VER_AFTER}"
  echo "!! This can desync Colab's preinstalled CUDA runtime from torch's expectations."
  echo "!! If GPU code breaks after this, consider:"
  echo "!!   pip install --no-deps 'lerobot[libero]==0.6.1'"
  echo "!! and manually installing any missing libero/hf-libero deps."
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo ""
fi

# ---------------------------------------------------------------------------
# 3. Verify the LIBERO integration actually imports (fail loudly, not silently).
# ---------------------------------------------------------------------------
python3 - <<'PYEOF'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
from lerobot.envs.configs import LiberoEnv  # noqa: F401
from lerobot.envs.libero import create_libero_envs  # noqa: F401
print("[xgap setup] lerobot LIBERO integration import OK")
PYEOF

# ---------------------------------------------------------------------------
# 4. Restart banner: only needed the first time lerobot goes from absent -> present,
#    since Colab sometimes has stale compiled extensions / mujoco-py caches in the
#    running kernel process.
# ---------------------------------------------------------------------------
if [ "${LEROBOT_WAS_INSTALLED}" = "no" ]; then
  echo ""
  echo "=========================================================================="
  echo " xgap setup: lerobot was just installed for the first time this session."
  echo " RESTART THE COLAB RUNTIME NOW (Runtime > Restart session), then re-run"
  echo " this script once more (it will no-op on the install step) before"
  echo " running any evaluation script."
  echo "=========================================================================="
  echo ""
fi

# ---------------------------------------------------------------------------
# 5. Hardware / library metadata log -- Colab hardware varies session to session,
#    so every run appends a timestamped snapshot instead of overwriting.
# ---------------------------------------------------------------------------
META_LOG="${XGAP_DRIVE_ROOT}/logs/env_meta.log"
{
  echo "---- xgap setup run ----"
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  echo "nproc: $(nproc 2>/dev/null || echo unknown)"
  echo "free -g:"
  free -g 2>/dev/null || echo "  (free unavailable)"
  echo "nvidia-smi:"
  nvidia-smi 2>/dev/null || echo "  (no GPU / nvidia-smi unavailable)"
  echo "python: $(python3 --version 2>&1)"
  echo "torch: ${TORCH_VER_AFTER}"
  python3 -c 'import lerobot; print("lerobot:", lerobot.__version__)' 2>/dev/null || echo "lerobot: unknown"
  python3 -c 'import libero; print("libero pkg present")' 2>/dev/null || echo "libero: import failed"
  echo "-------------------------"
} | tee -a "${META_LOG}"

echo "[xgap setup] done. Env meta appended to ${META_LOG}"
