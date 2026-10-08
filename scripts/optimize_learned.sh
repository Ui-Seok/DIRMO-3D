#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
dirmo_logged learned optimize --manifest "$DIRMO_MANIFEST" \
    --output "$DIRMO_RUN_DIR/learned" --router learned \
    --router-model "$DIRMO_ROUTER_MODEL" "$@"
