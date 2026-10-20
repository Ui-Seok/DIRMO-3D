"""Completion-only LoRA SFT for the SMDD pilot, independent of router training."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from pathlib import Path

from dirmo.data.smdd import assess_product, json_text
from dirmo.io import revision


def encode_pair(tokenizer, row: dict, max_length: int) -> dict:
    prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # Encode continuation separately: its prefix is exactly the prompt used for
    # inference. This also avoids BPE boundary changes to the assistant prefix.
    target_ids = tokenizer.encode(row["answer"] + tokenizer.eos_token, add_special_tokens=False)
    ids = prompt_ids + target_ids
    if len(ids) > max_length:
        raise ValueError(f"{row['task_id']}: {len(ids)} tokens > {max_length}; refusing truncation")
    if not prompt_ids or not target_ids:
        raise ValueError("Empty prompt/completion")
    return {"input_ids": ids, "prompt_length": len(prompt_ids),
            "target_ids": target_ids, "row": row}


def completion_loss(model, sample: dict, device):
    import torch
    import torch.nn.functional as F
    ids = torch.tensor([sample["input_ids"]], device=device)
    target = torch.tensor(sample["target_ids"], device=device)
    # Only completion prediction positions go through the large vocabulary head.
    # The preceding prompt token predicts the first completion token. Prompt
    # tokens are context, not labels; no input-token loss is optimized.
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=len(sample["target_ids"]) + 1).logits[0, :-1]
    return F.cross_entropy(logits.float(), target)


def average_loss(model, samples: list, device) -> float:
    import torch
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for sample in samples:
            n = len(sample["target_ids"])
            total += float(completion_loss(model, sample, device)) * n
            tokens += n
    return total / tokens


def predict(model, tokenizer, sample: dict, device, max_new_tokens: int) -> dict:
    import torch
    ids = torch.tensor([sample["input_ids"][:sample["prompt_length"]]], device=device)
    start = time.monotonic()
    with torch.inference_mode():
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
    row = sample["row"]
    return {"task_id": row["task_id"], "protein_key": row["protein_key"],
            "raw": raw, "seconds": time.monotonic() - start,
            "output_tokens": output.shape[1] - ids.shape[1],
            "assessment": assess_product(raw, row["reference_smiles"], row["witness_smiles"])}


def summarize_predictions(predictions: list) -> dict:
    fields = ["json_valid", "smiles_valid", "local_hard_constraints_passed", "changed", "witness_exact_match"]
    return {"rows": len(predictions),
            "counts": {field: sum(bool(p["assessment"].get(field)) for p in predictions) for field in fields},
            "rates": {field: sum(bool(p["assessment"].get(field)) for p in predictions) / len(predictions)
                      for field in fields} if predictions else {},
            "oracle_goal_success": None, "oracle_hold_success": None,
            "boltz_binding_success": None}


def run(args) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if args.epochs < 1 or args.accumulation < 1 or args.learning_rate <= 0 or args.rank < 1:
        raise ValueError("epochs, accumulation, learning rate and rank must be positive")
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("memory_fraction must be in (0, 1]")
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one physical GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    free, total = torch.cuda.mem_get_info(device)
    if free < args.min_free_gib * 1024**3:
        raise RuntimeError(f"Only {free / 1024**3:.1f} GiB free; need {args.min_free_gib} GiB")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    splits = {name: [json.loads(line) for line in (args.data / f"{name}.jsonl").read_text().splitlines() if line.strip()]
              for name in ["train", "validation"]}
    encoded = {name: [encode_pair(tokenizer, row, args.max_length) for row in rows] for name, rows in splits.items()}
    if not all(encoded.values()):
        raise ValueError("Empty train/validation split")
    # Deterministic sample chosen before looking at model outputs; never use test
    # rows for checkpoint selection or pilot tuning.
    prediction_samples = random.Random(args.seed).sample(encoded["validation"],
                        min(args.eval_generations, len(encoded["validation"])))
    args.output.mkdir(parents=True)
    log_path = args.output / "events.jsonl"

    def event(kind, **values):
        item = {"event": kind, "time": time.time(), **values}
        with log_path.open("a") as handle:
            handle.write(json_text(item) + "\n")
        print(json_text(item), flush=True)

    params = vars(args).copy()
    params = {key: str(value) if isinstance(value, Path) else value for key, value in params.items()}
    identity = {"args": params, "pid": os.getpid(), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": torch.cuda.get_device_name(device),
                "uuid": str(torch.cuda.get_device_properties(device).uuid),
                "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft", "rdkit"]},
                "data_sha256": {name: hashlib.sha256((args.data / f"{name}.jsonl").read_bytes()).hexdigest() for name in splits},
                "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
                "model_revision": revision(str(args.model)),
                "dataset_revision": revision(str(Path(splits["train"][0]["source"]).parent.parent)),
                "training_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "token_lengths": {name: {"min": min(len(x["input_ids"]) for x in xs),
                                         "max": max(len(x["input_ids"]) for x in xs)} for name, xs in encoded.items()},
                "validation_prediction_ids": [s["row"]["task_id"] for s in prediction_samples]}
    (args.output / "settings.json").write_text(json.dumps(identity, indent=2) + "\n")
    start = time.monotonic()
    event("loading", free_gib=free / 1024**3, rows={k: len(v) for k, v in encoded.items()})
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
                          dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    # Anchor to the text backbone: vision tower, embeddings and LM head remain
    # frozen. Cover both full-attention and Gated DeltaNet projection layers.
    target_modules = (r"model\.language_model\.layers\.\d+\."
                      r"(?:self_attn|linear_attn)\."
                      r"(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)")
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2,
                              lora_dropout=0.05, bias="none", target_modules=target_modules,
                              task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    event("loaded", trainable_parameters=trainable, allocated_gib=torch.cuda.memory_allocated() / 1024**3)
    # Initial LoRA B matrices are zero, so this is the unchanged base model.
    baseline_loss = average_loss(model, encoded["validation"], device)
    baseline = []
    for sample in prediction_samples:
        item = predict(model, tokenizer, sample, device, args.max_new_tokens)
        baseline.append(item)
        with (args.output / "base_predictions.jsonl").open("a") as handle:
            handle.write(json_text(item) + "\n")
        event("base_prediction", task_id=item["task_id"], assessment=item["assessment"])
    event("baseline", validation_loss=baseline_loss, predictions=summarize_predictions(baseline))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    updates_per_epoch = math.ceil(len(encoded["train"]) / args.accumulation)
    total_updates = updates_per_epoch * args.epochs
    step, history, best_loss = 0, [], float("inf")
    for epoch in range(args.epochs):
        model.train()
        order = list(encoded["train"])
        random.Random(args.seed + epoch).shuffle(order)
        epoch_loss = 0.0
        for offset in range(0, len(order), args.accumulation):
            batch = order[offset:offset + args.accumulation]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for sample in batch:
                loss = completion_loss(model, sample, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                batch_loss += float(loss.detach())
                (loss / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            step += 1
            warmup = max(1, round(total_updates * 0.05))
            factor = min(step / warmup, 1.0) * (0.5 + 0.5 * math.cos(math.pi * max(0, step - warmup) / max(1, total_updates - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * factor
            optimizer.step()
            epoch_loss += batch_loss
            event("update", epoch=epoch + 1, step=step, total_steps=total_updates,
                  loss=batch_loss / len(batch), lr=optimizer.param_groups[0]["lr"],
                  peak_gib=torch.cuda.max_memory_allocated() / 1024**3)
        loss = average_loss(model, encoded["validation"], device)
        item = {"epoch": epoch + 1, "train_loss": epoch_loss / len(order), "validation_loss": loss}
        history.append(item)
        event("epoch", **item)
        if loss < best_loss:
            best_loss = loss
            model.save_pretrained(args.output / "adapter")
            tokenizer.save_pretrained(args.output / "adapter")
    # Reload the best validation-loss adapter instead of evaluating an arbitrary
    # final epoch. Test data has still not been read by this program.
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model, load_file(str(args.output / "adapter/adapter_model.safetensors")))
    model.eval()
    tuned = []
    for sample in prediction_samples:
        item = predict(model, tokenizer, sample, device, args.max_new_tokens)
        tuned.append(item)
        with (args.output / "adapter_predictions.jsonl").open("a") as handle:
            handle.write(json_text(item) + "\n")
        event("adapter_prediction", task_id=item["task_id"], assessment=item["assessment"])
    report = {"status": "trained", "base_validation_loss": baseline_loss,
              "best_validation_loss": best_loss, "history": history,
              "base": summarize_predictions(baseline), "adapter": summarize_predictions(tuned),
              "trainable_parameters": trainable, "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "seconds": time.monotonic() - start, "settings": identity,
              "limitations": ["Small grouped validation pilot; test split remains untouched.",
                              "Witness match is imitation, not proof of molecular optimization.",
                              "ADMET-AI objectives/holds and Boltz binding probability have not been evaluated.",
                              "Adapter output is product_smiles; it is not the live DIRMO Proposal schema."]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    event("complete", report=str(args.output / "report.json"))
    return report
