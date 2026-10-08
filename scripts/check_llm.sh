#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
dirmo_runtime_dir="$("$DIRMO_CPU_PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("runtime_dir", "worker-runtime"))' \
    "$DIRMO_CONFIG")"
"$DIRMO_CPU_PYTHON" "$DIRMO_PROJECT_ROOT/scripts/check_llm.py" \
    --runtime-dir "$dirmo_runtime_dir" "$@"
