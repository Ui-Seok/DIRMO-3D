#!/usr/bin/env bash
set -euo pipefail
bash "$(dirname -- "${BASH_SOURCE[0]}")/start_workers.sh" --only llm "$@"


# Start on GPU 4
# DIRMO_LLM_GPU=4 DIRMO_FM_GPU=3 bash ~/code/DIRMO-3D/scripts/start_llm.sh