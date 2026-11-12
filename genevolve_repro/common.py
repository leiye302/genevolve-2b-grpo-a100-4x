import json
import importlib.util
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".deps" / "genevolve"
RL = "GenEvolve-Data-RL"
BENCH = "GenEvolve-Bench"
WORLD_SIZE = 4


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        except BaseException:
            stream.close()
            temporary.unlink()
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield row


def source_lock():
    return read_json(ROOT / "sources.json")


def load_upstream():
    if not (UPSTREAM / "genevolve" / "agent.py").is_file():
        raise RuntimeError("Run bash scripts/setup.sh to fetch the pinned GenEvolve source.")
    location = str(UPSTREAM)
    if location not in sys.path:
        sys.path.insert(0, location)


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relative_image(root, relative):
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or invalid dataset image {relative!r} under {root}")
    return path


def model_ready(path):
    if not (path / "config.json").is_file() or not (path / "tokenizer_config.json").is_file():
        return False
    index = path / "model.safetensors.index.json"
    shards = set(read_json(index)["weight_map"].values()) if index.is_file() else {"model.safetensors"}
    return bool(shards) and all(
        (path / name).is_file() and (path / name).stat().st_size > 0 for name in shards
    )


def required_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} before using the live services.")
    return value
