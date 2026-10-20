#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
dirmo_cli start-workers --timeout "$DIRMO_START_TIMEOUT" "$@"

# Start only LLM
# DIRMO_LLM_GPU=4 DIRMO_FM_GPU=3 bash ~/code/DIRMO-3D/scripts/start_workers.sh --only llm

# Start only FM
# DIRMO_LLM_GPU=3 DIRMO_FM_GPU=4 bash ~/code/DIRMO-3D/scripts/start_workers.sh --only fm