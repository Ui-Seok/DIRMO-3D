#!/usr/bin/env python3
"""Full fixed-checkpoint evaluation, including the last five-epoch adapter."""
import argparse
import json
import os
import time
from pathlib import Path

from dirmo.io import write_json
from dirmo.training.smdd_full import evaluate
from dirmo.training.smdd_sft import encode_pair


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/LocalData1/uslee/checkpoints/Qwen3.5-9B"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("runs/smdd_lora_02/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--generation-batch", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--memory-fraction", type=float, default=.90)
    parser.add_argument("--min-free-gib", type=float, default=36)
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one GPU with CUDA_VISIBLE_DEVICES")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    free, _ = torch.cuda.mem_get_info()
    if free < args.min_free_gib * 1024**3:
        raise RuntimeError("Insufficient free GPU memory")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False, local_files_only=True)
    model.eval()
    args.output.mkdir(parents=True)

    def event(kind, **values):
        item = {"event": kind, "utc_timestamp": time.time(), **values}
        with (args.output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(item) + "\n")
        print(json.dumps(item), flush=True)

    report = {"status": "running", "adapter": str(args.adapter),
              "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "used_for_selection": False, "splits": {}}
    for split in ["test", "validation", "train"]:
        rows = [json.loads(line) for line in (args.data / f"{split}.jsonl").read_text().splitlines()]
        samples = [encode_pair(tokenizer, row, args.max_length) for row in rows]
        report["splits"][split] = evaluate(model, tokenizer, samples, args.output / split, args, event, f"last_{split}")
        write_json(args.output / "report.json", report)
    report["status"] = "complete"
    write_json(args.output / "report.json", report)
