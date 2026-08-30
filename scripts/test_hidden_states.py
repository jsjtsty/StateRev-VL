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

    processor = AutoProcessor.from_pretrained(model_dir)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Answer with one word: what color is snow?"}],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states or []
    print("Number of hidden-state tensors:", len(hidden_states))
    for idx, tensor in enumerate(hidden_states):
        print(idx, tuple(tensor.shape), tensor.dtype, tensor.device)
    print(f"CUDA allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
