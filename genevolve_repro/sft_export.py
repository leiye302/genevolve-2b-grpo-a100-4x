"""Check and fingerprint the caller's existing HF SFT export."""

import hashlib
from pathlib import Path

from .common import read_json


TEXT_SIGNATURE = {
    "hidden_size": 2048,
    "num_hidden_layers": 28,
    "intermediate_size": 6144,
    "vocab_size": 151936,
}
REQUIRED = ("config.json", "tokenizer_config.json", "preprocessor_config.json")
OPTIONAL = (
    "chat_template.jinja", "generation_config.json", "processor_config.json",
    "video_preprocessor_config.json", "tokenizer.json", "vocab.json", "merges.txt",
    "added_tokens.json", "special_tokens_map.json",
)


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_files(folder):
    """Validate the existing 2B HF export; never fall back to base weights."""
    folder = Path(folder)
    if not all((folder / name).is_file() for name in REQUIRED):
        raise ValueError(
            "Point --sft-checkpoint at the complete Qwen3-VL-2B SFT HF export "
            "(model, tokenizer and processor), not at native ZeRO shards."
        )
    config = read_json(folder / "config.json")
    text = config.get("text_config") or {}
    if (config.get("model_type") != "qwen3_vl"
            or any(text.get(key) != value for key, value in TEXT_SIGNATURE.items())):
        raise ValueError("Expected the matched Qwen3-VL-2B-Instruct SFT export.")
    files = set(REQUIRED)
    files.update(name for name in OPTIONAL if (folder / name).is_file())
    if "tokenizer.json" not in files and not {"vocab.json", "merges.txt"} <= files:
        raise ValueError("The SFT export is missing its tokenizer vocabulary.")
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        weight_map = read_json(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("The SFT safetensors index has no weights.")
        shards = list(weight_map.values())
        if any(not isinstance(name, str) or "/" in name or "\\" in name
               or not name.endswith(".safetensors") for name in shards):
            raise ValueError("The SFT index must point to local safetensors shards.")
        files.add(index.name)
    else:
        shards = {"model.safetensors"}
    files.update(shards)
    for name in sorted(files):
        path = folder / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"The SFT export is missing a nonempty {name}.")
    return sorted(files)


def sft_manifest(folder):
    """Read-only content identity, portable across filesystem locations."""
    folder = Path(folder)
    files = export_files(folder)
    trainer_state = folder / "trainer_state.json"
    return {
        "model_family": "Qwen3-VL-2B-Instruct",
        "sft_global_step": read_json(trainer_state).get("global_step") if trainer_state.is_file() else None,
        "sha256": {name: file_sha256(folder / name) for name in files},
    }
