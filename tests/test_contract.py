import argparse
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import run
from genevolve_repro.common import BENCH, RL, ROOT, WORLD_SIZE, read_json, source_lock, write_json
from genevolve_repro.data import prepare
from genevolve_repro.objectives import advantages, grpo_loss, kl_reward
from genevolve_repro.sft_export import TEXT_SIGNATURE, export_files, sft_manifest


def fake_export(folder):
    folder.mkdir(parents=True)
    write_json(folder / "config.json", {"model_type": "qwen3_vl", "text_config": TEXT_SIGNATURE})
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
        write_json(folder / name, {})
    write_json(folder / "trainer_state.json", {"global_step": 550})
    (folder / "model.safetensors").write_bytes(b"checkpoint-fingerprint-fixture")


class ContractTest(unittest.TestCase):
    def test_four_gpu_launcher_reuses_explicit_sft_path(self):
        with TemporaryDirectory() as temp:
            work = Path(temp) / "new work"
            sft = Path(temp) / "existing sft"
            fake_export(sft)
            args = argparse.Namespace(command="grpo", sft_checkpoint=sft, steps=2)
            with patch.dict(os.environ, {"SERPER_API_KEY": "fixture", "JUDGE_API_KEY": "fixture"}), \
                    patch.object(run.subprocess, "run") as child:
                run.execute(args, work)
            child.assert_called_once_with(
                [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                 "--nproc-per-node=4", str(ROOT / "run.py"), "--work", str(work),
                 "_grpo", "--sft-checkpoint", str(sft.resolve()), "--steps", "2"],
                cwd=ROOT, check=True,
            )
            self.assertFalse((work / "sft").exists())

    def test_sft_fingerprint_relocatable_and_sensitive_to_weights(self):
        with TemporaryDirectory() as temp:
            first = Path(temp) / "one"
            second = Path(temp) / "two"
            fake_export(first)
            fake_export(second)
            expected = sft_manifest(first)
            self.assertEqual(expected["sft_global_step"], 550)
            self.assertEqual(expected, sft_manifest(second))
            (second / "model.safetensors").write_bytes(b"changed")
            self.assertNotEqual(expected["sha256"], sft_manifest(second)["sha256"])

    def test_rejects_wrong_model_or_partial_native_checkpoint(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "export"
            with self.assertRaisesRegex(ValueError, "HF export"):
                export_files(folder)
            fake_export(folder)
            write_json(folder / "config.json",
                       {"model_type": "qwen3_vl", "text_config": {**TEXT_SIGNATURE, "hidden_size": 4096}})
            with self.assertRaisesRegex(ValueError, "matched Qwen3-VL-2B"):
                export_files(folder)
            write_json(folder / "config.json", {"model_type": "qwen3_vl", "text_config": TEXT_SIGNATURE})
            write_json(folder / "model.safetensors.index.json",
                       {"weight_map": {"example": "model-00001-of-00002.safetensors"}})
            with self.assertRaisesRegex(ValueError, "missing a nonempty model-00001"):
                export_files(folder)
            write_json(folder / "model.safetensors.index.json",
                       {"weight_map": {"example": "../model.safetensors"}})
            with self.assertRaisesRegex(ValueError, "local safetensors shards"):
                export_files(folder)

    def test_prepare_downloads_only_matched_rl_and_bench(self):
        with TemporaryDirectory() as temp:
            work = Path(temp)
            with patch("huggingface_hub.snapshot_download") as download, \
                    patch("genevolve_repro.data.verify_data", return_value={"fixture": 1}):
                prepare(work)
            data = source_lock()["data"]
            download.assert_called_once_with(
                data["repo"], revision=data["revision"], repo_type="dataset", local_dir=work / "data",
                allow_patterns=[f"{folder}/{pattern}" for folder in (RL, BENCH)
                                for pattern in ("*.jsonl", "images/*")],
            )
            self.assertFalse((work / "sft").exists())
            self.assertFalse((work / "embedding").exists())

    def test_matched_grpo_clipping_and_reference_kl(self):
        cfg = read_json(ROOT / "configs" / "evolve.json")
        self.assertEqual((WORLD_SIZE, cfg["prompt_batch"], cfg["rollouts_per_prompt"]), (4, 8, 6))
        self.assertEqual((cfg["clip_low"], cfg["clip_high"], cfg["kl_coef"]), (0.20, 0.28, 0.001))
        self.assertEqual((cfg["seed"], cfg["text_reward_coef"]), (42, 0.5))
        self.assertFalse(any("sdl" in k or "experience" in k for k in cfg))
        turn = SimpleNamespace(old=torch.tensor([-2.0, -2.0, -2.0]),
                               reference=torch.tensor([-2.1, -2.5, -1.7]))
        current = torch.tensor([-2.0, -1.2, -2.8], requires_grad=True)
        delta = turn.old - turn.reference
        expected = 0.8 - 0.001 * (torch.exp(-delta) - 1 + delta).sum().item()
        self.assertAlmostEqual(kl_reward(0.8, [turn], cfg["kl_coef"]), expected)
        low, high = advantages([0.1, expected])
        for advantage in (low, high):
            ratio = (current - turn.old).exp()
            direct = -torch.minimum(advantage * ratio, advantage * ratio.clamp(0.8, 1.28)).sum()
            actual = grpo_loss(current, turn, advantage, cfg)
            torch.testing.assert_close(actual, direct)
            torch.testing.assert_close(torch.autograd.grad(actual, current)[0],
                                       torch.autograd.grad(direct, current)[0])
