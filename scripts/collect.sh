#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
dirmo_logged collect collect --manifest "$DIRMO_MANIFEST" \
    --output "$DIRMO_RUN_DIR/branches" --dataset "$DIRMO_DATASET" "$@"
