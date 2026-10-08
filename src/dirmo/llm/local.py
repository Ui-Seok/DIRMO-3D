from __future__ import annotations

import json
import time
from pathlib import Path

from ..gpu import assert_worker_mask
from ..io import digest, environment, revision, sha256_file


class LocalLLM:
    def __init__(self, config):
        assert_worker_mask()
        import torch
        from transformers import AutoTokenizer
        self.torch, self.config = torch, config
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Worker must see exactly one GPU")
        model_path = Path(config.model_path or "")
        if not config.model_path or not (model_path / "config.json").is_file():
            raise FileNotFoundError("Set llm.model_path to an existing local model directory")
        local_revision = revision(str(model_path))
        if config.revision and local_revision != config.revision:
            raise ValueError("Local LLM revision does not match configured revision")
        files = sorted(model_path.glob("*.safetensors"))
        if not files or any(f.stat().st_size < 1024 * 1024 for f in files):
            raise ValueError("LLM weights are missing or are Git LFS pointers")
        architecture = json.loads((model_path / "config.json").read_text())
        dtype = getattr(torch, config.dtype)
        kwargs = {"local_files_only": True, "dtype": dtype, "attn_implementation": "sdpa"}
        if config.backend == "qwen3.5":
            from transformers import Qwen3_5ForConditionalGeneration
            if architecture["model_type"] != "qwen3_5":
                raise ValueError("Qwen3.5 adapter requires the Qwen3.5 model architecture")
            loader = Qwen3_5ForConditionalGeneration
        else:
            from transformers import AutoModelForCausalLM
            if architecture["model_type"] != "qwen2" or config.thinking:
                raise ValueError("ChemDFM-v2 is Qwen2.5-based and has no thinking switch")
            loader = AutoModelForCausalLM
        if config.quantization == "int4":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
            kwargs["device_map"] = {"": "cuda:0"}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = loader.from_pretrained(model_path, **kwargs)
        if config.quantization == "none":
            self.model.to("cuda:0")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        test = self.tokenizer.apply_chat_template([{"role": "user", "content": "Return JSON."}],
               tokenize=False, add_generation_prompt=True,
               **({"enable_thinking": config.thinking} if config.backend == "qwen3.5" else {}))
        if config.backend == "qwen3.5" and not config.thinking and not test.rstrip().endswith("</think>"):
            raise ValueError("Tokenizer template did not apply enable_thinking=False")
        self.identity = {"path": str(model_path), "revision": local_revision,
                         "config_sha256": sha256_file(model_path / "config.json"),
                         "weights": {f.name: sha256_file(f) for f in files},
                         "loader": loader.__name__, "thinking_applied": config.thinking,
                         "quantization": config.quantization, "dtype": config.dtype,
                         "environment": environment()}
        self.identity["model_id"] = digest(self.identity)

    def propose(self, messages: list[dict]) -> dict:
        start, cfg, torch = time.monotonic(), self.config, self.torch
        kwargs = {"enable_thinking": cfg.thinking} if cfg.backend == "qwen3.5" else {}
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        inputs = self.tokenizer(text, return_tensors="pt")
        input_tokens = inputs["input_ids"].shape[-1]
        if input_tokens + cfg.output_cap > cfg.context_cap:
            return {"status": "failed", "reason": "context_cap_exceeded; prompt was not truncated",
                    "raw": "", "total_tokens": 0, "input_tokens": input_tokens, "output_tokens": 0}
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        torch.manual_seed(cfg.seed)
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=cfg.output_cap, do_sample=True,
                          temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
                          repetition_penalty=1.05 if cfg.backend == "chemdfm-v2" else 1.0,
                          pad_token_id=self.tokenizer.eos_token_id)
            tokens = outputs[0, input_tokens:]
            return {"status": "ok", "raw": self.tokenizer.decode(tokens, skip_special_tokens=True),
                    "input_tokens": input_tokens, "output_tokens": len(tokens),
                    "total_tokens": input_tokens + len(tokens), "latency_seconds": time.monotonic() - start,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(), "model_identity": self.identity,
                    "generation": cfg.model_dump(mode="json")}
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {"status": "oom", "reason": "Batch=1 OOM; no hidden retry or model switching",
                    "raw": "", "total_tokens": None, "latency_seconds": time.monotonic() - start,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated()}

    def health(self):
        return {"model_identity": self.identity, "peak_memory_bytes": self.torch.cuda.max_memory_allocated(),
                "device": "cuda:0", "device_uuid": str(self.torch.cuda.get_device_properties(0).uuid)}
