#!/usr/bin/env bash
# Multi-pass branch collection over a case manifest. Intended to run inside tmux:
# long jobs must survive a dropped session. Restarting is safe -- finished cases are
# skipped, so a crashed sweep resumes where it stopped.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

dirmo_passes="${1:-3}"
dirmo_oracle="${DIRMO_ORACLE_EDITS:-3}"
dirmo_regions="${DIRMO_REGIONS:-0}"
if ! [[ "$dirmo_passes" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Usage: bash scripts/run_sweep.sh [PASSES]\n' >&2
    exit 2
fi
if [[ ! -f "$DIRMO_MANIFEST" ]]; then
    printf 'Manifest not found: %s\n' "$DIRMO_MANIFEST" >&2
    exit 2
fi

mkdir -p -- "$DIRMO_RUN_DIR/logs"
printf 'config   %s\nmanifest %s (%s cases)\noutput   %s\ndataset  %s\noracle   %s edits/case\nregions  %s/case\npasses   %s\n' \
    "$DIRMO_CONFIG" "$DIRMO_MANIFEST" "$(wc -l < "$DIRMO_MANIFEST")" \
    "$DIRMO_RUN_DIR" "$DIRMO_DATASET" "$dirmo_oracle" "$dirmo_regions" "$dirmo_passes"

dirmo_cli stop-workers >/dev/null 2>&1 || true
dirmo_cli start-workers --timeout "$DIRMO_START_TIMEOUT" >"$DIRMO_RUN_DIR/logs/workers.json"
trap 'dirmo_cli stop-workers >/dev/null 2>&1 || true' EXIT

for (( dirmo_offset = 0; dirmo_offset < dirmo_passes; dirmo_offset++ )); do
    printf '\n===== pass %d of %d (seed-offset %d) started %s =====\n' \
        "$(( dirmo_offset + 1 ))" "$dirmo_passes" "$dirmo_offset" "$(date -Is)"
    dirmo_cli collect --manifest "$DIRMO_MANIFEST" \
        --output "$DIRMO_RUN_DIR/branches_s$dirmo_offset" \
        --dataset "$DIRMO_DATASET" \
        --seed-offset "$dirmo_offset" \
        --oracle-edits "$dirmo_oracle" \
        --regions "$dirmo_regions" \
        > "$DIRMO_RUN_DIR/logs/pass_$dirmo_offset.json" \
        2> "$DIRMO_RUN_DIR/logs/pass_$dirmo_offset.err" \
        || printf 'pass %d exited nonzero; see logs/pass_%d.err\n' "$dirmo_offset" "$dirmo_offset"
    printf 'pass %d finished %s; dataset rows: %s\n' "$dirmo_offset" "$(date -Is)" \
        "$(wc -l < "$DIRMO_DATASET" 2>/dev/null || echo 0)"
done
printf '\nsweep finished %s; dataset %s\n' "$(date -Is)" "$DIRMO_DATASET"
