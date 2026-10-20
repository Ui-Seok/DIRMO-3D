#!/usr/bin/env python3
"""Evaluate saved SMDD adapters on validation or an explicitly selected test split."""
import argparse
import json
import os
import random
from pathlib import Path

from dirmo.data.smdd import json_text
from dirmo.training.smdd_sft import average_loss, encode_pair, predict, summarize_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/LocalData1/uslee/checkpoints/Qwen3.5-9B"))
    parser.add_argument("--adapter", type=Path, help="Omit to evaluate the base model")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows; otherwise a deterministic random subset")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--memory-fraction", type=float, default=0.60)
    parser.add_argument("--min-free-gib", type=float, default=28)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU")
    if not 0 < args.memory_fraction <= 1 or args.limit < 0:
        raise ValueError("Invalid memory fraction or sample limit")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    free, _ = torch.cuda.mem_get_info()
    if free < args.min_free_gib * 1024**3:
        raise RuntimeError("Insufficient free GPU memory")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = [json.loads(line) for line in (args.data / f"{args.split}.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty split")
    if args.limit:
        rows = random.Random(args.seed).sample(rows, min(args.limit, len(rows)))
    samples = [encode_pair(tokenizer, row, args.max_length) for row in rows]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False, local_files_only=True)
    model.eval()
    args.output.mkdir(parents=True)
    loss = average_loss(model, samples, "cuda:0")
    predictions = []
    for sample in samples:
        result = predict(model, tokenizer, sample, "cuda:0", args.max_new_tokens)
        predictions.append(result)
        with (args.output / "predictions.jsonl").open("a") as handle:
            handle.write(json_text(result) + "\n")
        print(json_text({"task_id": result["task_id"], "assessment": result["assessment"]}), flush=True)
    report = {"model": str(args.model), "adapter": str(args.adapter) if args.adapter else None,
              "split": args.split, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "completion_loss": loss, "predictions": summarize_predictions(predictions),
              "limitations": ["ADMET-AI and Boltz optimization objectives/holds have not been measured.",
                              "Local graph checks are not the complete SMDD benchmark evaluation."]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
