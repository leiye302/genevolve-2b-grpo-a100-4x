from datetime import timedelta
from functools import partial
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

from genevolve_repro.common import ROOT, WORLD_SIZE, read_json
from genevolve_repro.distributed import copy_actor, latest, load_checkpoint, restore_optimizer, save_checkpoint
from genevolve_repro.evolve import Group, update_actor
from genevolve_repro.feedback import Outcome
from genevolve_repro.policy import ActorScorer, Turn, score_turn
from test_pipeline import tiny_policy


def sharded(scorer, device):
    scorer.model.model.visual.requires_grad_(False)
    return FSDP(
        scorer, device_id=device, use_orig_params=True, ignored_modules=[scorer.model.model.visual],
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen3VLTextDecoderLayer}),
    )


def run_rank(rank, directory):
    folder = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(folder / "rendezvous").as_uri(), rank=rank,
                            world_size=WORLD_SIZE, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(5)
        device = torch.device("cpu")
        cfg = read_json(ROOT / "configs" / "evolve.json")
        actor = sharded(ActorScorer(tiny_policy()), device)
        replica = tiny_policy()
        copy_actor(actor, replica)
        optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
        trajectories = []
        for index in range(2):
            turns = []
            for _ in range(rank + 1):
                turn = Turn([3, 5], [10 + index, 12])
                with torch.no_grad():
                    turn.old = score_turn(ActorScorer(replica), turn.prefix, turn.tokens, cfg["temperature"], device)
                turn.reference = turn.old.clone()
                turns.append(turn)
            trajectories.append(SimpleNamespace(turns=turns))
        group = Group(None, trajectories, [Outcome(True, reward, {}, 0, "", None) for reward in (0, 1)])
        _, norm = update_actor(actor, optimizer, [group], cfg, device, 2)
        assert norm > 0
        scheduler.step()
        save_checkpoint(folder, actor, optimizer, scheduler, 1, {}, rank, device)
        saved = load_checkpoint(latest(folder))
        resumed_scorer = ActorScorer(tiny_policy())
        resumed_scorer.load_state_dict(saved["actor"])
        resumed = sharded(resumed_scorer, device)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
        resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, 2)
        restore_optimizer(resumed, resumed_optimizer, resumed_scheduler, saved, rank)
        assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()
        expected = tiny_policy()
        copy_actor(actor, expected)
        copy_actor(resumed, replica)
        torch.testing.assert_close(replica.lm_head.weight, expected.lm_head.weight)
    finally:
        dist.destroy_process_group()


class DistributedTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux" and dist.is_gloo_available(), "Linux Gloo is required")
    def test_fsdp_update_and_resume_with_uneven_turn_counts(self):
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(run_rank, args=(directory,), nprocs=WORLD_SIZE)
