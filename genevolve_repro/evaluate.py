import os

import torch
import torch.distributed as dist

from .common import BENCH, RL, ROOT, jsonl, load_upstream, model_ready, read_json, required_env, write_json
from .data import examples
from .distributed import barrier, initialize, load_model
from .feedback import LiveFeedback, image_scorer, judge_settings
from .policy import make_agent, valid_program


def evaluate_case(agent, feedback, example, metadata, image_path):
    trajectory = agent.collect(example.prompt)
    result = trajectory.result
    row = {
        **{key: metadata.get(key) or "unknown" for key in ("eval_type", "category", "difficulty")},
        "id": example.id, "prompt": example.prompt, "gt_image": str(example.gt),
        "result": result.to_dict(), "success": False, "eval_success": False,
    }
    if valid_program(result):
        feedback.render(result, image_path)
        row.update(
            success=True, eval_success=True, image_path=str(image_path),
            scores=feedback.score_image(example, image_path),
        )
    return row


def summarize(folder, rows):
    scorer = image_scorer()
    path = folder / "results_eval.json"
    write_json(path, rows)
    summary = scorer.build_benchmark_summary(path, len(rows))
    write_json(folder / "summary.json", summary)
    scorer._write_csv(str(folder / "summary.csv"), summary)
    return summary


def _evaluate(work, policy, split, max_cases, rank, size, device):
    from transformers import AutoTokenizer

    load_upstream()
    required_env("SERPER_API_KEY")
    if not model_ready(policy):
        raise RuntimeError(f"Expected an HF policy export at {policy}.")
    if split == "wise":
        from .wise import WiseFeedback, summarize_wise, wise_examples

        pool, metadata = wise_examples()
        count = 1000
    else:
        data, name = (BENCH, "test") if split == "bench" else (RL, "eval")
        dataset = work / "data" / data
        pool = examples(dataset, name)
        metadata = list(jsonl(dataset / f"{name}.jsonl"))
        count = 594 if split == "bench" else 600
        if len(pool) != count:
            raise ValueError(f"Expected all {count} cases in the pinned {data} split.")
    if max_cases is not None:
        if not 1 <= max_cases <= count:
            raise ValueError(f"--max-cases must be between 1 and {count}.")
        count = max_cases
    marker = policy / "completed.json"
    suffix = "" if max_cases is None else f"-{max_cases}-cases"
    folder = work / "eval" / f"{split}-{policy.name}{suffix}"
    cfg = read_json(ROOT / "configs" / "evolve.json")
    contract = {
        "policy": str(policy), "split": split, "count": count,
        "checkpoint": read_json(marker) if marker.is_file() else None,
    }
    if split != "wise":
        contract["judge"] = judge_settings(cfg)
    lock = folder / "config.json"
    if lock.is_file() and read_json(lock) != contract:
        raise RuntimeError(f"Evaluation in {folder} uses a different policy or judge. Archive it before rerunning.")
    if rank == 0:
        write_json(lock, contract)
    barrier()
    jobs = [(i, folder / "cases" / f"{i:05d}.json") for i in range(rank, count, size)]
    jobs = [(i, path) for i, path in jobs if not path.is_file()]
    if jobs:
        if split != "rl-val":
            # Use the released run_agent.py defaults for final benchmark inference.
            cfg.update(temperature=0.6, top_p=0.9)
        os.environ.setdefault("IMAGE_DOWNLOAD_DIR", str(work / "cache" / f"eval-rank{rank}"))
        torch.manual_seed(cfg["seed"] + rank)
        model = load_model(policy, torch.bfloat16).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(policy, use_fast=True)
        agent = make_agent(model, tokenizer, device, cfg)
        feedback = WiseFeedback() if split == "wise" else LiveFeedback(cfg, size)
        for i, path in jobs:
            # Make a resumed evaluation independent of which earlier cases exist.
            torch.manual_seed(cfg["seed"] + i)
            image_name = f"{pool[i].id}.png" if split == "wise" else f"{i:05d}.png"
            image = folder / "images" / image_name
            row = (
                feedback.evaluate(agent, pool[i], metadata[i], image) if split == "wise"
                else evaluate_case(agent, feedback, pool[i], metadata[i], image)
            )
            write_json(path, row)
            print(f"{split} case {i + 1}/{count} scored={row['success']}", flush=True)
    barrier()
    if rank == 0:
        rows = [read_json(folder / "cases" / f"{i:05d}.json") for i in range(count)]
        scores = summarize_wise(folder, rows) if split == "wise" else summarize(folder, rows)["all"]
        print(scores, flush=True)
        print(f"Benchmark artifacts at {folder}", flush=True)


def worker(work, policy, split, max_cases):
    rank, size, device = initialize()
    try:
        _evaluate(work, policy, split, max_cases, rank, size, device)
    finally:
        dist.destroy_process_group()
