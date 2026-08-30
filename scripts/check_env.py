from __future__ import annotations

import platform
import sys

import torch
import transformers

from _theory_of_space_utils import DEFAULT_MODEL_DIR, candidate_theory_of_space_sources


def main() -> None:
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print(f"Logical GPU 0 device name: {torch.cuda.get_device_name(0)}")
    else:
        print("Logical GPU 0 device name: unavailable")

    sources = candidate_theory_of_space_sources()
    print(f"Theory of Space source exists: {bool(sources)}")
    if sources:
        print(f"Theory of Space source path: {sources[0]}")
    print(f"Qwen3.5 model dir: {DEFAULT_MODEL_DIR}")
    print(f"Qwen3.5 model exists: {DEFAULT_MODEL_DIR.exists()}")


if __name__ == "__main__":
    main()
