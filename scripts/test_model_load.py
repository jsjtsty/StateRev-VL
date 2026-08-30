from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from _theory_of_space_utils import DEFAULT_MODEL_DIR


def resolve_model_dir() -> Path:
    return Path(os.environ.get("QWEN3_VL_MODEL_DIR", str(DEFAULT_MODEL_DIR)))


def main() -> None:
    model_dir = resolve_model_dir()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    print("Loading processor...")
    _ = AutoProcessor.from_pretrained(model_dir)

    print("Loading model...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    print("Model loaded.")
    print("Model dtype:", next(model.parameters()).dtype)
    print("Model device:", next(model.parameters()).device)
    print(f"CUDA allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"CUDA reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
