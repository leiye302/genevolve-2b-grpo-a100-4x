from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import math
import os
import random
from statistics import fmean

import torch
import torch.distributed as dist

from .actor_text import ACTOR_TEXT_PROTOCOL

from .common import RL, ROOT, load_upstream, model_ready, read_json, required_env, source_lock, write_json
from .data import examples
from .distributed import (
    copy_actor, export_policy, gather, initialize, latest, load_checkpoint, load_model,
    make_sharded_actor, max_count, restore_optimizer, restore_rng, save_checkpoint,
)
from .feedback import LiveFeedback, judge_settings
from .objectives import advantages, grpo_loss, kl_reward
from .policy import ActorScorer, make_agent, score_turn
from .sft_export import file_sha256, sft_manifest


@dataclass
class Group:
    example: object
    trajectories: list
    outcomes: list


def gather_group(agent, feedback, example, folder, count):
    trajectories = [agent.collect(example.prompt) for _ in range(count)]
    with ThreadPoolExecutor(max_workers=min(count, 4)) as pool:
        pending = [
            pool.submit(feedback.reward, example, trajectory, folder / f"{i:02d}.png")
            for i, trajectory in enumerate(trajectories)
        ]
        outcomes = [task.result() for task in pending]
    return Group(example, trajectories, outcomes)


@torch.no_grad()
def score_group(group, scorer, cfg, device):
    for trajectory in group.trajectories:
        for turn in trajectory.turns:
            turn.old = score_turn(scorer, turn.prefix, turn.tokens, cfg["temperature"], device).cpu()


@torch.no_grad()
def score_reference(groups, scorer, cfg, device):
    for group in groups:
        for trajectory in group.trajectories:
            for turn in trajectory.turns:
                turn.reference = score_turn(scorer, turn.prefix, turn.tokens, cfg["temperature"], device).cpu()


def update_actor(actor, optimizer, groups, cfg, device, dummy_id):
    # Every rank owns the same number of whole prompt groups. FSDP averages
    # gradients across ranks, so the local sequence denominator is sufficient.
    normalizer = sum(len(group.trajectories) for group in groups)
    jobs = []
    for group in groups:
        rewards = [
            kl_reward(outcome.reward, trajectory.turns, cfg["kl_coef"])
            for trajectory, outcome in zip(group.trajectories, group.outcomes, strict=True)
        ]
        for trajectory, advantage in zip(group.trajectories, advantages(rewards), strict=True):
            jobs.extend((turn, advantage) for turn in trajectory.turns)
    count = max_count(len(jobs), device)
    if count == 0:
        raise RuntimeError("All trajectories were empty. Check the model and generation budgets.")
    optimizer.zero_grad(set_to_none=True)
    grpo_sum = 0.0
    for i in range(count):
        # Offloaded FSDP needs no_sync for accumulation. A short dummy turn
        # keeps collective order identical when trajectories end at different times.
        context = actor.no_sync() if hasattr(actor, "no_sync") and i + 1 < count else nullcontext()
        with context:
            if i < len(jobs):
                turn, advantage = jobs[i]
                new = score_turn(actor, turn.prefix, turn.tokens, cfg["temperature"], device)
                grpo = grpo_loss(new, turn, advantage, cfg)
                loss = grpo / normalizer
                grpo_sum += grpo.detach().item()
            else:
                loss = score_turn(actor, [dummy_id], [dummy_id], cfg["temperature"], device).sum() * 0
            loss.backward()
    clip = getattr(actor, "clip_grad_norm_", None)
    norm = clip(1.0) if clip else torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    if not torch.isfinite(norm).item():
        raise RuntimeError("Non-finite actor gradient. The previous checkpoint is intact.")
    optimizer.step()
    return grpo_sum / normalizer, float(norm)


def write_rollouts(folder, group):
    write_json(folder / "rollouts.json", {
        "actor_text_protocol": ACTOR_TEXT_PROTOCOL,
        "id": group.example.id,
        "prompt": group.example.prompt,
        "trajectories": [
            {"result": trajectory.result.to_dict(), "outcome": outcome.to_dict(),
             "sampled_tokens": sum(len(turn.tokens) for turn in trajectory.turns)}
            for trajectory, outcome in zip(group.trajectories, group.outcomes, strict=True)
        ],
    })


def _loop(work, sft, rank, size, device, limit):
    from transformers import AutoProcessor, AutoTokenizer

    load_upstream()
    required_env("SERPER_API_KEY")
    cfg = read_json(ROOT / "configs" / "evolve.json")
    if cfg["prompt_batch"] % size:
        raise ValueError("prompt_batch must split into whole groups across GPU ranks.")
    dataset = work / "data" / RL
    pool = examples(dataset, "train")
    if len(pool) != 2575:
        raise RuntimeError("The pinned RL train split must contain 2,575 prompts.")
    random.Random(cfg["seed"]).shuffle(pool)
    total = limit if limit is not None else math.ceil(len(pool) / cfg["prompt_batch"])
    if total < 1:
        raise ValueError("At least one GRPO update is required.")
    out = work / "grpo"
    out.mkdir(parents=True, exist_ok=True)
    final = out / "final"
    # Only one rank reads the several-GB SFT weight files for provenance.
    inputs = gather({
        "sft": sft_manifest(sft),
        "train_jsonl_sha256": file_sha256(dataset / "train.jsonl"),
    } if rank == 0 else None)[0]
    contract = {"algorithm": "grpo", "sources": source_lock(), **inputs,
                "config": cfg, "judge": judge_settings(cfg), "world_size": size, "total_steps": total}
    if (out / "contract.json").is_file() and read_json(out / "contract.json") != contract:
        raise RuntimeError("This work directory has a different GRPO, SFT, or training-data identity.")
    if rank == 0:
        write_json(out / "contract.json", contract)
    if model_ready(final) and (final / "completed.json").is_file() and read_json(final / "completed.json")["step"] == total:
        if rank == 0:
            print(f"GRPO complete at {final}", flush=True)
        return
    checkpoint_path = latest(out)
    saved = load_checkpoint(checkpoint_path) if checkpoint_path else None
    if saved and saved["contract"] != contract:
        raise RuntimeError("Saved checkpoint configuration does not match this run.")
    start = saved["step"] if saved else 0
    random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)
    actor = make_sharded_actor(sft, device, saved["actor"] if saved else None)
    optimizer = torch.optim.AdamW((p for p in actor.parameters() if p.requires_grad),
                                  lr=cfg["learning_rate"], weight_decay=0.0, foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)
    rng = saved["rng"][rank] if saved else None
    if saved:
        restore_optimizer(actor, optimizer, scheduler, saved, rank)
    del saved
    processor = AutoProcessor.from_pretrained(sft)
    tokenizer = AutoTokenizer.from_pretrained(sft, use_fast=True)
    replica = load_model(sft, torch.bfloat16).eval()
    reference = load_model(sft, torch.bfloat16).eval()
    if rng:
        restore_rng(rng, device)
    os.environ.setdefault("IMAGE_DOWNLOAD_DIR", str(work / "cache" / f"grpo-rank{rank}"))
    feedback = LiveFeedback(cfg, size)
    agent = make_agent(replica, tokenizer, device, cfg)
    actor.train()
    local_batch = cfg["prompt_batch"] // size

    for step in range(start, total):
        replica.to(device)
        copy_actor(actor, replica)
        scorer = ActorScorer(replica)
        groups = []
        for offset in range(local_batch):
            index = (step * cfg["prompt_batch"] + rank * local_batch + offset) % len(pool)
            example = pool[index]
            path = out / "rollouts" / f"step-{step + 1:05d}" / f"rank-{rank}-{offset}"
            group = gather_group(agent, feedback, example, path, cfg["rollouts_per_prompt"])
            score_group(group, scorer, cfg, device)
            groups.append(group)
            write_rollouts(path, group)

        replica.cpu()
        torch.cuda.empty_cache()
        reference.to(device)
        score_reference(groups, ActorScorer(reference), cfg, device)
        reference.cpu()
        torch.cuda.empty_cache()
        grpo, norm = update_actor(actor, optimizer, groups, cfg, device, tokenizer.eos_token_id)
        scheduler.step()
        outcomes = [outcome for group in groups for outcome in group.outcomes]
        metrics = gather({
            "reward": fmean(outcome.reward for outcome in outcomes),
            "valid_fraction": fmean(outcome.valid for outcome in outcomes),
            "grpo": grpo, "grad_norm": norm,
        })
        if rank == 0:
            row = {k: fmean(item[k] for item in metrics) for k in metrics[0]}
            row.update(step=step + 1, lr=scheduler.get_last_lr()[0])
            write_json(out / "metrics" / f"step-{step + 1:05d}.json", row)
            print(row, flush=True)
        if (step + 1) % cfg["save_every"] == 0 or step + 1 == total:
            save_checkpoint(out, actor, optimizer, scheduler, step + 1, contract, rank, device,
                            keep=cfg["keep_checkpoints"])

    export_policy(final, actor, replica, processor, total, rank)


def worker(work, sft, limit):
    rank, size, device = initialize()
    try:
        _loop(work, sft, rank, size, device, limit)
    finally:
        dist.destroy_process_group()
