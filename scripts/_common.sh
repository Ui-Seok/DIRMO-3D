#!/usr/bin/env bash
# Shared functions for the purpose-specific Bash launchers; source this file.
set -euo pipefail
DIRMO_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DIRMO_PROJECT_ROOT/scripts/settings.sh"
cd -- "$DIRMO_PROJECT_ROOT"

if ! command -v "$DIRMO_CPU_PYTHON" >/dev/null 2>&1; then
    printf 'Python not found: %s\nEdit scripts/settings.sh (DIRMO_CPU_PYTHON).\n' \
        "$DIRMO_CPU_PYTHON" >&2
    exit 2
fi
if [[ ! -f "$DIRMO_CONFIG" ]]; then
    printf 'Config not found: %s\nEdit scripts/settings.sh (DIRMO_CONFIG).\n' \
        "$DIRMO_CONFIG" >&2
    exit 2
fi
export PYTHONPATH="$DIRMO_PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""

dirmo_help_requested() {
    local dirmo_argument
    for dirmo_argument in "$@"; do
        if [[ "$dirmo_argument" == "--help" || "$dirmo_argument" == "-h" ]]; then
            return 0
        fi
    done
    return 1
}

dirmo_cli() {
    "$DIRMO_CPU_PYTHON" -m dirmo --config "$DIRMO_CONFIG" "$@"
}

dirmo_logged() {
    local dirmo_stage="$1"
    shift
    if dirmo_help_requested "$@"; then
        dirmo_cli "$@"
        return
    fi
    mkdir -p -- "$DIRMO_RUN_DIR/logs"
    dirmo_cli "$@" 2>&1 | tee "$DIRMO_RUN_DIR/logs/$dirmo_stage.log"
}
