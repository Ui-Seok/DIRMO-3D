#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
dirmo_logged rule optimize --manifest "$DIRMO_MANIFEST" \
    --output "$DIRMO_RUN_DIR/rule" --router rule "$@"
