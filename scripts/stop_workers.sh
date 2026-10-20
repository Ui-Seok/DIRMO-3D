#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
dirmo_cli stop-workers "$@"


# Stop only LLM
# DIRMO_LLM_GPU=4 DIRMO_FM_GPU=3 bash ~/code/DIRMO-3D/scripts/stop_workers.sh --only llm

# Stop only FM
# DIRMO_LLM_GPU=3 DIRMO_FM_GPU=4 bash ~/code/DIRMO-3D/scripts/stop_workers.sh --only fm