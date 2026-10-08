#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
dirmo_logged compare compare --manifest "$DIRMO_MANIFEST" \
    --output "$DIRMO_RUN_DIR/compare" --router-model "$DIRMO_ROUTER_MODEL" \
    --policies rule learned "$@"
