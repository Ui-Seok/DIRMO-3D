from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .schemas import GenerationParameters, Schema


class WorkerConfig(Schema):
    python: str | None = None
    gpu: str
    port: int = Field(ge=1024, le=65535)
    minimum_free_mib: int = Field(default=4000, ge=0)


class LLMConfig(WorkerConfig):
    backend: Literal["qwen3.5", "chemdfm-v2"] = "qwen3.5"
    model_path: str | None = None
    revision: str | None = None
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    quantization: Literal["none", "int4"] = "none"
    thinking: bool = False
    context_cap: int = Field(default=4096, ge=512, le=16384)
    output_cap: int = Field(default=1024, ge=64, le=4096)
    temperature: float = Field(default=0.7, gt=0)
    top_p: float = Field(default=0.8, gt=0, le=1)
    top_k: int = Field(default=20, ge=1)
    parsing_retries: int = Field(default=1, ge=0, le=2)
    seed: int = Field(default=42, ge=0)


class FMConfig(WorkerConfig):
    repo_path: str | None = None
    repo_revision: str = "739a33c2aab74d074dd24605b9a2e34dc809ae9c"
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    max_atoms: int = Field(default=64, ge=1)
    max_pocket_atoms: int = Field(default=1000, ge=10)
    parameters: GenerationParameters = Field(default_factory=GenerationParameters)


class EvaluationConfig(Schema):
    posebusters: bool = True
    vina: bool = True
    interactions: bool = False
    clash_tolerance: float = Field(default=0.75, gt=0, le=1)
    pocket_cutoff: float = Field(default=6.0, gt=0)


class Config(Schema):
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig(gpu="0", port=8765))
    fm: FMConfig = Field(default_factory=lambda: FMConfig(gpu="1", port=8766))
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime_dir: str = "worker-runtime"
    run_root: str = "runs"
    seed: int = 42

    @model_validator(mode="after")
    def separate_workers(self):
        if self.llm.gpu == self.fm.gpu or self.llm.port == self.fm.port:
            raise ValueError("LLM and FM need distinct GPUs and ports")
        if self.llm.output_cap >= self.llm.context_cap:
            raise ValueError("output_cap must be below context_cap")
        return self


def load_config(path: str) -> Config:
    raw = json.loads(Path(path).read_text())
    overrides = {
        "DIRMO_LLM_GPU": ("llm", "gpu"), "DIRMO_FM_GPU": ("fm", "gpu"),
        "DIRMO_LLM_PYTHON": ("llm", "python"), "DIRMO_FM_PYTHON": ("fm", "python"),
        "DIRMO_LLM_MODEL": ("llm", "model_path"), "DIRMO_FM_CHECKPOINT": ("fm", "checkpoint_path"),
        "DIRMO_FLOWR_REPO": ("fm", "repo_path"),
    }
    for env, (section, key) in overrides.items():
        if os.environ.get(env):
            raw.setdefault(section, {})[key] = os.environ[env]
    return Config.model_validate(raw)
