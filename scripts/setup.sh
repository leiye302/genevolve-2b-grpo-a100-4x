#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")/.."

mode="${1:-}"
if [[ "$#" -gt 1 || ( -n "$mode" && "$mode" != --sources-only && "$mode" != --renderer && "$mode" != --wise-sources ) ]]; then
    echo "Usage: bash scripts/setup.sh [--sources-only|--renderer|--wise-sources]" >&2
    exit 2
fi

base_python="${GENEVOLVE_PYTHON:-python3.11}"

fetch_pin() {
    local source="$1" target="$2"
    local url revision
    read -r url revision < <("$base_python" -c '
import json, sys
with open("sources.json", encoding="utf-8") as stream:
    pin = json.load(stream)[sys.argv[1]]
print(pin["repo"], pin["revision"])
' "$source")
    revision="${revision%$'\r'}"
    if [[ ! -d "$target/.git" ]]; then
        git init -q "$target"
        git -C "$target" remote add origin "$url"
    fi
    if [[ "$(git -C "$target" remote get-url origin)" != "$url" ]]; then
        echo "Unexpected source URL at $target" >&2
        exit 1
    fi
    if ! git -C "$target" rev-parse --verify HEAD >/dev/null 2>&1; then
        git -C "$target" fetch --depth 1 origin "$revision"
        git -C "$target" checkout --detach "$revision"
    fi
    if [[ "$(git -C "$target" rev-parse HEAD)" != "$revision" ||
          -n "$(git -C "$target" status --porcelain --untracked-files=no)" ]]; then
        echo "Expected a clean checkout at $revision under $target" >&2
        exit 1
    fi
}

if [[ "$mode" == --wise-sources ]]; then
    mkdir -p .deps
    fetch_pin wise .deps/wise
    exit 0
fi
if [[ "$mode" != --renderer ]]; then
    mkdir -p .deps
    fetch_pin agent .deps/genevolve
fi
if [[ "$mode" == --sources-only ]]; then
    exit 0
fi

if [[ "$(uname -s)" != Linux ]]; then
    echo "The CUDA environments require Linux." >&2
    exit 1
fi
"$base_python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"'

if [[ "$mode" == --renderer ]]; then
    env_dir=.venv-render
else
    env_dir=.venv
    if ! command -v nvcc >/dev/null || ! nvcc --version | grep -q 'release 12.8'; then
        echo "Activate the CUDA 12.8 toolkit before building FlashAttention-2." >&2
        exit 1
    fi
fi
"$base_python" -m venv "$env_dir"
py="$env_dir/bin/python"
"$py" -m pip install --upgrade pip wheel setuptools ninja packaging psutil
"$py" -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

if [[ "$mode" == --renderer ]]; then
    "$py" -m pip install diffusers==0.38.0 transformers==4.57.1 accelerate==1.11.0 \
        Pillow==11.3.0 fastapi==0.121.1 uvicorn==0.38.0
    "$py" -c 'from diffusers import QwenImageEditPlusPipeline'
else
    "$py" -m pip install --no-build-isolation -r requirements.txt
    MAX_JOBS="${GENEVOLVE_BUILD_JOBS:-4}" "$py" -m pip install --no-build-isolation flash-attn==2.8.3
    "$py" -c 'import flash_attn, torch; from transformers import Qwen3VLForConditionalGeneration; assert torch.version.cuda == "12.8"'
fi
"$py" -m pip check
