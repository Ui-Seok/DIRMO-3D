#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# Help must not prepare data or stop the FM worker.
if dirmo_help_requested "$@"; then
    dirmo_cli train-router --help
    exit 0
fi

# --dataset names the raw collection data for this wrapper. Other training
# arguments are forwarded to the existing CLI.
dirmo_input_dataset="$DIRMO_DATASET"
dirmo_selected_kernel="$DIRMO_KERNEL_ID"
dirmo_training_args=()
while (( $# > 0 )); do
    case "$1" in
        --dataset|--kernel-id)
            if (( $# < 2 )) || [[ "$2" == --* || -z "$2" ]]; then
                printf '%s needs a value.\n' "$1" >&2
                exit 2
            fi
            if [[ "$1" == "--dataset" ]]; then
                dirmo_input_dataset="$2"
            else
                dirmo_selected_kernel="$2"
            fi
            shift 2
            ;;
        --dataset=*)
            dirmo_input_dataset="${1#*=}"
            shift
            ;;
        --kernel-id=*)
            dirmo_selected_kernel="${1#*=}"
            shift
            ;;
        *)
            dirmo_training_args+=("$1")
            shift
            ;;
    esac
done

dirmo_prepare_args=(--config "$DIRMO_CONFIG" --dataset "$dirmo_input_dataset"
                    --output "$DIRMO_TRAIN_DATASET")
if [[ -n "$dirmo_selected_kernel" ]]; then
    dirmo_prepare_args+=(--kernel-id "$dirmo_selected_kernel")
fi
mkdir -p -- "$DIRMO_RUN_DIR/logs"
"$DIRMO_CPU_PYTHON" "$DIRMO_PROJECT_ROOT/scripts/prepare_router_data.py" \
    "${dirmo_prepare_args[@]}" 2>&1 | tee "$DIRMO_RUN_DIR/logs/prepare_training.log"

# Data preparation must succeed before the CLI frees the FM GPU.
dirmo_logged train train-router --dataset "$DIRMO_TRAIN_DATASET" \
    --output "$DIRMO_ROUTER_MODEL" --device fm --epochs "$DIRMO_EPOCHS" \
    "${dirmo_training_args[@]}"
