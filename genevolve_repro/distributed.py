from contextlib import nullcontext
from datetime import timedelta
from functools import partial
import os
from pathlib import Path
import random
import shutil
from tempfile import mkdtemp

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffload, FullOptimStateDictConfig, FullStateDictConfig,
    FullyShardedDataParallel as FSDP, MixedPrecision, StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .common import WORLD_SIZE, write_json
from .policy import ActorScorer


def initialize():
    rank, size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if size != WORLD_SIZE or torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError(f"Expose four CUDA GPUs and start the worker through run.py.")
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2), device_id=device)
    return rank, size, device


def load_model(path, dtype, attention="flash_attention_2"):
    from transformers import Qwen3VLForConditionalGeneration

    return Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=dtype, attn_implementation=attention, low_cpu_mem_usage=True,
    )


def make_sharded_actor(path, device, saved=None):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

    # Load the attention backend after loading FP32 master weights. FSDP
    # supplies BF16 parameters for the training forward and backward.
    policy = load_model(path, torch.float32, attention="sdpa")
    policy.set_attn_implementation("flash_attention_2")
    policy.model.visual.requires_grad_(False)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    scorer = ActorScorer(policy)
    if saved is not None:
        scorer.load_state_dict(saved)
    return FSDP(
        scorer,
        device_id=device,
        auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                 transformer_layer_cls={Qwen3VLTextDecoderLayer}),
        use_orig_params=True,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        cpu_offload=CPUOffload(offload_params=True),
        ignored_modules=[policy.model.visual],
        limit_all_gathers=True,
    )


@torch.no_grad()
def copy_actor(sharded, target, rank0_only=False):
    # Generation uses a local BF16 replica so decoding does not all-gather
    # parameters for each token. The replica is refreshed before each batch.
    if not isinstance(sharded, FSDP):
        target.load_state_dict(sharded.model.state_dict())
        return
    # For CPU/Gloo smoke tests, the gathered parameters are already on CPU.
    # Torch 2.8's CPU offload context assumes a device-to-host copy.
    with FSDP.summon_full_params(
        sharded, writeback=False, offload_to_cpu=sharded.compute_device.type != "cpu", rank0_only=rank0_only,
    ):
        if not rank0_only or dist.get_rank() == 0:
            source = {
                name.replace("._fsdp_wrapped_module.", "."): parameter
                for name, parameter in sharded.module.model.named_parameters()
            }
            for name, parameter in target.named_parameters():
                parameter.copy_(source[name])


def gather(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def max_count(count, device):
    if not dist.is_initialized():
        return count
    value = torch.tensor(count, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return int(value.item())


def barrier():
    if dist.is_initialized():
        dist.barrier()


def full_state(actor):
    if not isinstance(actor, FSDP):
        return nullcontext()
    offload = actor.compute_device.type != "cpu"
    return FSDP.state_dict_type(
        actor,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=offload, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=offload, rank0_only=True),
    )


def checkpoints(folder):
    paths = [
        path for path in folder.glob("checkpoint-*")
        if path.is_dir() and not path.is_symlink() and path.name.removeprefix("checkpoint-").isdigit()
        and (path / "training.pt").is_file()
    ]
    return sorted(paths, key=lambda p: int(p.name.removeprefix("checkpoint-")))


def latest(folder):
    paths = checkpoints(folder)
    return paths[-1] if paths else None


def load_checkpoint(path):
    return torch.load(path / "training.pt", map_location="cpu", mmap=True, weights_only=True)


def restore_optimizer(actor, optimizer, scheduler, saved, rank):
    optimizer_state = saved["optimizer"]
    if isinstance(actor, FSDP):
        with full_state(actor):
            optimizer_state = FSDP.optim_state_dict_to_load(actor, optimizer, optimizer_state if rank == 0 else {})
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(saved["scheduler"])


def get_rng(device):
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(value, device):
    random.setstate(value["python"])
    torch.set_rng_state(value["torch"])
    if value["cuda"] is not None:
        torch.cuda.set_rng_state(value["cuda"], device)


def save_checkpoint(folder, actor, optimizer, scheduler, step, contract, rank, device, keep=2):
    states = gather(get_rng(device))
    with full_state(actor):
        model = actor.state_dict()
        optimizer_state = FSDP.optim_state_dict(actor, optimizer) if isinstance(actor, FSDP) else optimizer.state_dict()
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"checkpoint-{step:05d}"
        temporary = Path(mkdtemp(prefix=".checkpoint-", dir=folder))
        torch.save({
            "step": step, "contract": contract, "actor": model, "optimizer": optimizer_state,
            "scheduler": scheduler.state_dict(), "rng": states,
        }, temporary / "training.pt")
        temporary.rename(target)
        for previous in checkpoints(folder)[:-keep]:
            resolved = previous.resolve()
            if resolved.parent != folder.resolve():
                raise ValueError(f"Unexpected checkpoint directory: {resolved}")
            shutil.rmtree(resolved)
    barrier()


def export_policy(folder, actor, replica, processor, step, rank):
    copy_actor(actor, replica, rank0_only=True)
    if rank == 0:
        temporary = Path(mkdtemp(prefix=".export-", dir=folder.parent))
        replica.save_pretrained(temporary, safe_serialization=True)
        processor.save_pretrained(temporary)
        write_json(temporary / "completed.json", {"step": step})
        temporary.rename(folder)
    barrier()
