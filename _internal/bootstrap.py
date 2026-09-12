"""SoulX-Singer offline environment bootstrap.

Call this before any model or library import to lock down network access
and point all model caches to local project directories.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- 1. Offline: block runtime network downloads ---
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# Prevent NLTK from phoning home for data
os.environ.setdefault("NLTK_DATA", str(ROOT / "_data" / "nltk_data"))

# Disable HuggingFace telemetry
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

# --- 2. Local model cache ---
_hf_cache = str(ROOT / "_data" / "huggingface")
os.environ.setdefault("HF_HOME", _hf_cache)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _hf_cache)
os.environ.setdefault("TRANSFORMERS_CACHE", _hf_cache)

# --- 3. Local Gradio temp ---
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / "_data" / "gradio_tmp"))

# --- 4. Ensure _data directories exist ---
for d in [
    ROOT / "_data" / "nltk_data",
    ROOT / "_data" / "huggingface",
    ROOT / "_data" / "gradio_tmp",
]:
    d.mkdir(parents=True, exist_ok=True)

# --- 5. Add project root to Python path ---
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_models() -> bool:
    """Quick check that expected model directories exist."""
    required = [
        ROOT / "pretrained_models" / "SoulX-Singer" / "model.pt",
        ROOT / "pretrained_models" / "SoulX-Singer" / "model-svc.pt",
        ROOT / "pretrained_models" / "whisper-base" / "config.json",
        ROOT / "pretrained_models" / "SoulX-Singer-Preprocess" / "rmvpe" / "rmvpe.pt",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("[!!!] 缺少以下模型文件：")
        for p in missing:
            print(f"      {p}")
        return False
    print("[OK] 所有模型文件就绪")
    return True


if __name__ == "__main__":
    print(f"SoulX-Singer root: {ROOT}")
    check_models()
