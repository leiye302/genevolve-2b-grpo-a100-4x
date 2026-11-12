from dataclasses import dataclass
from pathlib import Path

from .common import BENCH, RL, jsonl, relative_image, source_lock, write_json


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    gt: Path | None


def examples(folder, split):
    rows = []
    ids = set()
    for row in jsonl(folder / f"{split}.jsonl"):
        sample_id = str(row["id"])
        prompt = row.get("prompt") or row.get("question")
        if not isinstance(prompt, str) or not prompt.strip() or not sample_id or sample_id in ids:
            raise ValueError(f"Invalid or repeated example {sample_id!r} in {folder}/{split}")
        gt = relative_image(folder, row["gt_image"])
        rows.append(Example(sample_id, prompt, gt))
        ids.add(sample_id)
    return rows


def verify_data(work):
    folder = work / "data"
    counts = {}
    for dataset, split, count in ((RL, "train", 2575), (RL, "eval", 600), (BENCH, "test", 594)):
        actual = len(examples(folder / dataset, split))
        if actual != count:
            raise ValueError(f"{dataset}/{split} has {actual} rows, expected {count}")
        counts[f"{dataset}/{split}"] = actual
    return counts


def prepare(work):
    from huggingface_hub import snapshot_download

    lock = source_lock()
    dataset = lock["data"]
    print(f"Preparing {dataset['repo']} at {dataset['revision'][:7]}", flush=True)
    snapshot_download(
        dataset["repo"], revision=dataset["revision"], repo_type="dataset", local_dir=work / "data",
        allow_patterns=[f"{folder}/{pattern}" for folder in (RL, BENCH)
                        for pattern in ("*.jsonl", "images/*")],
    )
    counts = verify_data(work)
    write_json(work / "prepared.json", {"sources": lock, "counts": counts})
    print(counts, flush=True)
