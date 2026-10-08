#!/usr/bin/env bash
# Edit these defaults once. No conda activation or shell export is required.
# GPU/model/worker Python choices remain in DIRMO_CONFIG.
DIRMO_CPU_PYTHON="${DIRMO_CPU_PYTHON:-/home/uslee/miniconda3/envs/dirmo/bin/python}"
DIRMO_CONFIG="${DIRMO_CONFIG:-configs/local.json}"
# Prefer the user's expanded GPU example when present; otherwise use the
# versioned small example. An explicit manifest always takes precedence.
if [[ -z "${DIRMO_MANIFEST:-}" ]]; then
    DIRMO_MANIFEST="examples/local_case.json"
    if [[ -f "$DIRMO_PROJECT_ROOT/examples/gpu_case.json" ]]; then
        DIRMO_MANIFEST="examples/gpu_case.json"
    fi
fi
DIRMO_RUN_DIR="${DIRMO_RUN_DIR:-runs/pipeline_01}"
DIRMO_DATASET="${DIRMO_DATASET:-$DIRMO_RUN_DIR/branches.jsonl}"
DIRMO_TRAIN_DATASET="${DIRMO_TRAIN_DATASET:-$DIRMO_RUN_DIR/training.jsonl}"
DIRMO_ROUTER_MODEL="${DIRMO_ROUTER_MODEL:-$DIRMO_RUN_DIR/router.json}"
DIRMO_KERNEL_ID="${DIRMO_KERNEL_ID:-}"
DIRMO_EPOCHS="${DIRMO_EPOCHS:-100}"
DIRMO_START_TIMEOUT="${DIRMO_START_TIMEOUT:-600}"
