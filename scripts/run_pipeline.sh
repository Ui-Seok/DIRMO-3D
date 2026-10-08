#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash scripts/run_pipeline.sh [RUN_DIR]

Start workers, collect measured branches, prepare and train the router on the
FM GPU, restart workers, then optimize with the learned router.
RUN_DIR defaults to scripts/settings.sh. An existing run directory is refused.
Uses the config and manifest in scripts/settings.sh; no export is required.
Existing managed workers are stopped. Workers are stopped again on exit.
HELP
    exit 0
fi
if (( $# > 1 )) || [[ "${1:-}" == -* ]]; then
    printf 'Usage: bash scripts/run_pipeline.sh [RUN_DIR]\n' >&2
    exit 2
fi
if (( $# == 1 )); then
    export DIRMO_RUN_DIR="$1"
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# Child launchers must use the same resolved paths, even without shell exports.
export DIRMO_CPU_PYTHON DIRMO_CONFIG DIRMO_MANIFEST DIRMO_RUN_DIR DIRMO_DATASET
export DIRMO_TRAIN_DATASET DIRMO_ROUTER_MODEL DIRMO_KERNEL_ID DIRMO_EPOCHS DIRMO_START_TIMEOUT

if [[ -e "$DIRMO_RUN_DIR" || -e "$DIRMO_DATASET" || -e "$DIRMO_TRAIN_DATASET" \
      || -e "$DIRMO_ROUTER_MODEL" ]]; then
    printf 'Output already exists. Choose a new run directory: %s\n' "$DIRMO_RUN_DIR" >&2
    exit 2
fi
if [[ ! -f "$DIRMO_MANIFEST" ]]; then
    printf 'Manifest not found: %s\n' "$DIRMO_MANIFEST" >&2
    exit 2
fi
printf 'Config: %s\nManifest: %s\nOutput: %s\n' \
    "$DIRMO_CONFIG" "$DIRMO_MANIFEST" "$DIRMO_RUN_DIR"

dirmo_cleanup() {
    local dirmo_exit_status=$?
    trap - EXIT
    dirmo_cli stop-workers || true
    exit "$dirmo_exit_status"
}

dirmo_cli doctor
# Starting both workers cannot append an FM worker to an existing LLM worker.
dirmo_cli stop-workers
trap dirmo_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf '\n[1/5] Starting LLM and FLOWR workers\n'
bash "$DIRMO_PROJECT_ROOT/scripts/start_workers.sh"
printf '\n[2/5] Collecting measured branch data\n'
bash "$DIRMO_PROJECT_ROOT/scripts/collect.sh"
printf '\n[3/5] Preparing data and training the router on the FM GPU\n'
bash "$DIRMO_PROJECT_ROOT/scripts/train_router.sh"
printf '\n[4/5] Restarting workers after training\n'
bash "$DIRMO_PROJECT_ROOT/scripts/stop_workers.sh"
bash "$DIRMO_PROJECT_ROOT/scripts/start_workers.sh"
printf '\n[5/5] Optimizing with the learned router\n'
bash "$DIRMO_PROJECT_ROOT/scripts/optimize_learned.sh"
printf '\nFinished. Results: %s/learned; logs: %s/logs\n' \
    "$DIRMO_RUN_DIR" "$DIRMO_RUN_DIR"
