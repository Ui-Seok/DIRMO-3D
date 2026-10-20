"""Full SMDD continuation: validation LR comparison, five epochs, frozen test.

The starting adapter has already seen one complete training epoch. Each LR trial
starts from exactly those same weights. Test data is opened only after selection.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import time
from pathlib import Path

from dirmo.data.smdd import assess_product, json_text
from dirmo.io import write_json
from dirmo.training.smdd_sft import encode_pair, summarize_predictions


def response_diagnostic(raw, row):
    match = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", raw, flags=re.DOTALL)
    return assess_product(match.group(1).strip() if match else raw,
                          row["reference_smiles"], row["witness_smiles"])


def batch_losses(model, samples, device, pad_id):
    """Right-padded teacher forcing, masked completion prediction positions only."""
    import torch
    import torch.nn.functional as F
    width = max(len(s["input_ids"]) for s in samples)
    ids = torch.full((len(samples), width), pad_id, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    positions = sorted({s["prompt_length"] - 1 + i for s in samples for i in range(len(s["target_ids"]))})
    position_index = {p: i for i, p in enumerate(positions)}
    labels = torch.full((len(samples), len(positions)), -100, device=device, dtype=torch.long)
    for i, sample in enumerate(samples):
        n = len(sample["input_ids"])
        ids[i, :n] = torch.tensor(sample["input_ids"], device=device)
        mask[i, :n] = 1
        indices = [position_index[sample["prompt_length"] - 1 + j] for j in range(len(sample["target_ids"]))]
        labels[i, indices] = torch.tensor(sample["target_ids"], device=device)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False,
                   logits_to_keep=torch.tensor(positions, device=device)).logits
    losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                             reduction="none", ignore_index=-100).reshape(labels.shape)
    counts = (labels != -100).sum(dim=1)
    sums = losses.sum(dim=1)
    return sums / counts, sums.sum() / counts.sum()


def generate_batch(model, tokenizer, samples, device, max_new_tokens):
    import torch
    prompts = [s["input_ids"][:s["prompt_length"]] for s in samples]
    width = max(map(len, prompts))
    # Generation must be left-padded: each prompt ends at the last input column.
    ids = torch.full((len(samples), width), tokenizer.eos_token_id, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, prompt in enumerate(prompts):
        ids[i, -len(prompt):] = torch.tensor(prompt, device=device)
        mask[i, -len(prompt):] = 1
    start = time.monotonic()
    with torch.inference_mode():
        outputs = model.generate(input_ids=ids, attention_mask=mask, use_cache=True,
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    elapsed = time.monotonic() - start
    results = []
    for sample, output in zip(samples, outputs):
        tokens = output[width:].tolist()
        if tokenizer.eos_token_id in tokens:
            tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        row = sample["row"]
        strict = assess_product(raw, row["reference_smiles"], row["witness_smiles"])
        diagnostic = response_diagnostic(raw, row)
        metrics = diagnostic.get("metrics")
        results.append({"task_id": row["task_id"], "protein_key": row["protein_key"],
                        "raw": raw, "assessment": strict, "fence_removed_diagnostic": diagnostic,
                        "graph_metric_deltas": {k: v - row["reference_metrics"][k] for k, v in metrics.items()}
                                               if metrics else None,
                        "output_tokens": len(tokens), "batch_seconds": elapsed,
                        "hit_token_cap": len(tokens) == max_new_tokens and tokens[-1] != tokenizer.eos_token_id})
    return results


def summarize(records):
    strict = summarize_predictions(records)
    diagnostic = summarize_predictions([{"assessment": r["fence_removed_diagnostic"]} for r in records])
    diagnostic["counts"]["changed_and_local_passed"] = sum(
        bool(r["fence_removed_diagnostic"].get("changed")) and
        bool(r["fence_removed_diagnostic"].get("local_hard_constraints_passed")) for r in records)
    diagnostic["rates"]["changed_and_local_passed"] = diagnostic["counts"]["changed_and_local_passed"] / len(records)
    loss_rows = [r for r in records if r["completion_loss"] is not None]
    loss = sum(r["completion_loss"] * r["completion_tokens"] for r in loss_rows) / sum(r["completion_tokens"] for r in loss_rows) if loss_rows else None
    return {"rows": len(records), "completion_loss": loss, "nonfinite_loss_rows": len(records) - len(loss_rows),
            "strict": strict, "fence_removed_diagnostic": diagnostic,
            "hit_token_cap": sum(r["hit_token_cap"] for r in records)}


def selection_key(report):
    """Predeclared chemical validity priority; no scalar utility or test score."""
    counts = report["fence_removed_diagnostic"]["counts"]
    loss = report["completion_loss"]
    return (counts["changed_and_local_passed"], counts["smiles_valid"],
            -loss if loss is not None and math.isfinite(loss) else -math.inf)


def evaluate(model, tokenizer, samples, folder, args, event, label):
    import torch
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    model.eval()
    losses = {}
    with torch.no_grad():
        for offset in range(0, len(samples), args.microbatch):
            batch = samples[offset:offset + args.microbatch]
            values, _ = batch_losses(model, batch, "cuda:0", tokenizer.eos_token_id)
            for sample, value in zip(batch, values.tolist()):
                losses[sample["row"]["task_id"]] = value if math.isfinite(value) else None
    records = []
    for offset in range(0, len(samples), args.generation_batch):
        batch = samples[offset:offset + args.generation_batch]
        generated = generate_batch(model, tokenizer, batch, "cuda:0", args.max_new_tokens)
        for sample, record in zip(batch, generated):
            record["completion_loss"] = losses[record["task_id"]]
            record["completion_tokens"] = len(sample["target_ids"])
            records.append(record)
            with (folder / "predictions.jsonl").open("a") as handle:
                handle.write(json_text(record) + "\n")
        event("evaluation_progress", label=label, done=len(records), rows=len(samples))
    report = summarize(records)
    # Macros expose differences between target groups instead of only row counts.
    groups = {}
    for record in records:
        groups.setdefault(record["protein_key"], []).append(record)
    report["per_protein"] = {key: summarize(values) for key, values in groups.items()}
    report["selection_key"] = [v if math.isfinite(v) else None for v in selection_key(report)]
    write_json(folder / "report.json", report)
    event("evaluation_complete", label=label, rows=report["rows"], loss=report["completion_loss"],
          strict=report["strict"]["counts"], diagnostic=report["fence_removed_diagnostic"]["counts"])
    return report


def length_batches(samples, size, seed):
    order = list(samples)
    rng = random.Random(seed)
    rng.shuffle(order)
    batches = []
    # Local length bucketing reduces padding and vocabulary-head memory while
    # reshuffling all batches each epoch. Every row is used exactly once.
    for offset in range(0, len(order), size * 8):
        block = sorted(order[offset:offset + size * 8], key=lambda s: len(s["input_ids"]))
        batches.extend(block[i:i + size] for i in range(0, len(block), size))
    rng.shuffle(batches)
    return batches


def train_epoch(model, optimizer, tokenizer, samples, args, epoch, lr, event, label):
    import torch
    model.train()
    torch.manual_seed(args.seed + epoch)
    torch.cuda.manual_seed_all(args.seed + epoch)
    batches = length_batches(samples, args.microbatch, args.seed + epoch)
    total, count = 0.0, 0
    for index, batch in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        values, loss = batch_losses(model, batch, "cuda:0", tokenizer.eos_token_id)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite TRAINING loss; optimizer update would corrupt weights")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        warmup = max(1, round(len(batches) * 0.05))
        for group in optimizer.param_groups:
            group["lr"] = lr * min((index + 1) / warmup, 1.0)
        optimizer.step()
        total += float(values.detach().sum())
        count += len(batch)
        event("training_update", label=label, total_epoch=epoch, update=index + 1, updates=len(batches),
              rows_seen=count, rows=len(samples), loss=float(loss.detach()),
              lr=optimizer.param_groups[0]["lr"], peak_gib=torch.cuda.max_memory_allocated() / 1024**3)
    if count != len(samples):
        raise AssertionError("Training coverage mismatch")
    return total / count


def run(args):
    import torch
    from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.total_epochs < 3 or args.microbatch < 1 or args.generation_batch < 1 or any(lr <= 0 for lr in args.learning_rates):
        raise ValueError("Invalid epoch/batch/LR arguments")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to one GPU")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    free, _ = torch.cuda.mem_get_info()
    if free < args.min_free_gib * 1024**3:
        raise RuntimeError(f"Need {args.min_free_gib} GiB free, found {free / 1024**3:.1f}")
    origin = json.loads((args.pilot / "training/report.json").read_text())
    if origin["settings"]["args"]["epochs"] != 1 or len(origin["history"]) != 1:
        raise ValueError("This continuation expects the completed one-epoch starting adapter")
    for name in ["train", "validation"]:
        actual = hashlib.sha256((args.data / f"{name}.jsonl").read_bytes()).hexdigest()
        if actual != origin["settings"]["data_sha256"][name]:
            raise ValueError(f"{name} data differs from starting adapter training data")
    cfg_hash = hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest()
    if cfg_hash != origin["settings"]["model_config_sha256"]:
        raise ValueError("Starting adapter uses a different base model")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    samples = {}
    for name in ["train", "validation"]:
        rows = [json.loads(line) for line in (args.data / f"{name}.jsonl").read_text().splitlines()]
        samples[name] = [encode_pair(tokenizer, row, args.max_length) for row in rows]
    args.output.mkdir(parents=True)

    def event(kind, **values):
        record = {"event": kind, "utc_timestamp": time.time(), **values}
        with (args.output / "events.jsonl").open("a") as handle:
            handle.write(json_text(record) + "\n")
        print(json_text(record), flush=True)

    start = time.monotonic()
    manifest = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "pid": os.getpid(), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "uuid": str(torch.cuda.get_device_properties(0).uuid),
                "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "rdkit"]},
                "train_rows": len(samples["train"]), "validation_rows": len(samples["validation"]),
                "starting_adapter": str(args.pilot / "training/adapter"),
                "starting_adapter_sha256": hashlib.sha256((args.pilot / "training/adapter/adapter_model.safetensors").read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "validation_selection": "changed AND local constraints passed; then valid SMILES; then lower completion loss (fence-only diagnostic)",
                "early_stopping": False, "test_used_for_selection": False,
                "warm_start": "Adapter weights from epoch 1; pilot optimizer state was not saved",
                "status": "running"}
    write_json(args.output / "manifest.json", manifest)
    event("loading", free_gib=free / 1024**3, microbatch=args.microbatch, generation_batch=args.generation_batch)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
    model = PeftModel.from_pretrained(model, args.pilot / "training/adapter", is_trainable=True, local_files_only=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    manifest["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_json(args.output / "manifest.json", manifest)
    event("loaded", trainable_parameters=manifest["trainable_parameters"],
          allocated_gib=torch.cuda.memory_allocated() / 1024**3)

    def state():
        return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}

    def restore(values):
        set_peft_model_state_dict(model, values)

    def optimizer(lr):
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    initial = state()
    initial_report = evaluate(model, tokenizer, samples["validation"], args.output / "epoch_01_validation", args, event, "epoch_01")
    best_state, best_report, best_epoch = initial, initial_report, 1
    model.save_pretrained(args.output / "best_adapter")
    history = [{"total_epoch": 1, "source": "previous completed training", "validation": initial_report}]
    trials = []
    selected = None
    for lr in args.learning_rates:
        restore(initial)
        opt = optimizer(lr)
        label = f"lr_{lr:g}"
        event("lr_trial_start", learning_rate=lr, total_epoch=2)
        training_loss = train_epoch(model, opt, tokenizer, samples["train"], args, 2, lr, event, label)
        folder = args.output / "lr_trials" / label
        report = evaluate(model, tokenizer, samples["validation"], folder / "validation", args, event, label)
        model.save_pretrained(folder / "adapter")
        item = {"learning_rate": lr, "total_epoch": 2, "train_loss": training_loss, "validation": report}
        trials.append(item)
        if selected is None or selection_key(report) > selection_key(selected["validation"]):
            selected = item
            selected_state = state()
            selected_optimizer = copy.deepcopy(opt.state_dict())
        if selection_key(report) > selection_key(best_report):
            best_report, best_epoch, best_state = report, 2, state()
            model.save_pretrained(args.output / "best_adapter")
        write_json(args.output / "lr_trials.json", trials)
    restore(selected_state)
    chosen_lr = selected["learning_rate"]
    opt = optimizer(chosen_lr)
    opt.load_state_dict(selected_optimizer)
    history.append({**selected, "label": "selected_lr_trial"})
    event("lr_selected", learning_rate=chosen_lr, selection_key=selected["validation"]["selection_key"])
    # User requested continuing even when validation looks poor: run every epoch
    # to the requested total. Keep best and last checkpoints independently.
    for epoch in range(3, args.total_epochs + 1):
        lr = chosen_lr * (0.75 ** (epoch - 2))
        training_loss = train_epoch(model, opt, tokenizer, samples["train"], args, epoch, lr, event, f"epoch_{epoch:02d}")
        report = evaluate(model, tokenizer, samples["validation"], args.output / f"epoch_{epoch:02d}_validation", args, event, f"epoch_{epoch:02d}")
        model.save_pretrained(args.output / "checkpoints" / f"epoch_{epoch:02d}")
        history.append({"total_epoch": epoch, "learning_rate": lr, "train_loss": training_loss, "validation": report})
        if selection_key(report) > selection_key(best_report):
            best_report, best_epoch, best_state = report, epoch, state()
            model.save_pretrained(args.output / "best_adapter")
        write_json(args.output / "history.json", history)
    model.save_pretrained(args.output / "last_adapter")
    restore(best_state)
    tokenizer.save_pretrained(args.output / "best_adapter")
    write_json(args.output / "selection.json", {"selected_peak_learning_rate": chosen_lr, "best_total_epoch": best_epoch,
        "completed_total_epochs": args.total_epochs, "validation_key": best_report["selection_key"],
        "test_used_for_selection": False, "early_stopping": False})
    event("checkpoint_fixed", selected_lr=chosen_lr, best_total_epoch=best_epoch, completed_total_epochs=args.total_epochs)
    # Only now open the test file. No subsequent parameter updates occur.
    test_rows = [json.loads(line) for line in (args.data / "test.jsonl").read_text().splitlines()]
    samples["test"] = [encode_pair(tokenizer, row, args.max_length) for row in test_rows]
    final = {}
    model.eval()
    for name in ["test", "validation", "train"]:
        final[name] = {}
        final[name]["adapter"] = evaluate(model, tokenizer, samples[name], args.output / "final" / "adapter" / name, args, event, f"final_adapter_{name}")
        with model.disable_adapter():
            final[name]["base"] = evaluate(model, tokenizer, samples[name], args.output / "final" / "base" / name, args, event, f"final_base_{name}")
        write_json(args.output / "full_results.json", final)
    report = {"status": "complete", "starting_total_epochs": 1, "completed_total_epochs": args.total_epochs,
              "selected_peak_learning_rate": chosen_lr, "best_total_epoch": best_epoch,
              "lr_trials": trials, "history": history, "final": final,
              "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3, "seconds": time.monotonic() - start,
              "limitations": ["LR chosen among tested candidates, not a global optimum.",
                              "Test was evaluated only after checkpoint selection; train scores are fitting diagnostics.",
                              "ADMET-AI and Boltz goals/holds/binding probability were not measured.",
                              "PLIP features were not added to this baseline input; their benefit requires a separate comparison."]}
    write_json(args.output / "report.json", report)
    manifest["status"] = "complete"
    write_json(args.output / "manifest.json", manifest)
    event("complete", report=str(args.output / "report.json"))
    return report
