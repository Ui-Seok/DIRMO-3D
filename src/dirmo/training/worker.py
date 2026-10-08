"""Optional gate fitting on the FM GPU after its resident model has stopped."""

import argparse
import json
import os

from ..gpu import assert_worker_mask


def main():
    assert_worker_mask()
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Gate training worker must see exactly one FM GPU")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kernel-id")
    parser.add_argument("--group-unit", default="target")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from .router import train_router
    from ..io import write_json
    torch.cuda.reset_peak_memory_stats()
    report = train_router(args.dataset, args.output, args.kernel_id, args.group_unit, args.epochs, args.seed, "cuda:0")
    report.update(peak_memory_bytes=torch.cuda.max_memory_allocated(), actual_gpu_uuid=os.environ["DIRMO_GPU_UUID"],
                  physical_gpu=os.environ["DIRMO_PHYSICAL_GPU"], pid=os.getpid())
    write_json(args.output + ".report.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
