import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from openai import OpenAI
from PIL import Image
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from genevolve_repro.common import ROOT, load_upstream, read_json
from genevolve_repro.data import Example
from genevolve_repro.distributed import (
    export_policy, latest, load_checkpoint, restore_optimizer, save_checkpoint,
)
from genevolve_repro.evaluate import evaluate_case, summarize
from genevolve_repro.evolve import gather_group, score_group, score_reference, update_actor
from genevolve_repro.feedback import LiveFeedback
from genevolve_repro.policy import ActorScorer, make_agent, score_turn
from genevolve_repro.render_service import app_for


def tiny_policy():
    config = Qwen3VLConfig(
        image_token_id=252, video_token_id=253, vision_start_token_id=254, vision_end_token_id=255,
        text_config={
            "vocab_size": 256, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "max_position_embeddings": 2048,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]},
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        },
        vision_config={
            "depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
            "out_hidden_size": 32, "deepstack_visual_indexes": [],
            "num_position_embeddings": 16,
        },
    )
    return Qwen3VLForConditionalGeneration(config)


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **_):
        return list(map(ord, "\n".join(m["role"] + ": " + m["content"] for m in messages) + "\nassistant:"))

    def __call__(self, text, **_):
        return {"input_ids": list(map(ord, text)), "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def decode(self, ids, **_):
        return "".join(map(chr, ids))

    def save_pretrained(self, folder):
        (folder / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class ScriptedPolicy:
    def __init__(self, variants):
        call = '<tool_call>{"name":"image_search","arguments":{"query":"bird"}}</tool_call>'
        self.responses = iter(
            response for variant in variants
            for response in (call, '<answer>' + json.dumps({
                "gen_prompt": f"Use the first reference image to depict a {variant} bird.",
                "reference_images": [{"img_id": "IMG_001"}],
            }) + '</answer>')
        )

    def generate(self, input_ids, **_):
        response = torch.tensor([list(map(ord, next(self.responses)))])
        return torch.cat((input_ids, response), dim=1)


def mock_judge(request):
    body = json.loads(request.content)
    assert request.url.host == "dashscope.aliyuncs.com"
    assert request.url.path == "/compatible-mode/v1/chat/completions"
    assert body["model"] == "qwen3.7-flash-2026-07-15"
    assert body["enable_thinking"] is False
    assert body["response_format"] == {"type": "json_object"}
    messages = body["messages"]
    instruction = messages[0]["content"]
    if instruction.startswith("Assess the sufficiency"):
        program = json.loads(messages[1]["content"])["program"]
        data = {"score": 1.0 if "gold bird" in program else 0.25, "rationale": "fixture"}
    else:
        assert sum(part["type"] == "image_url" for part in messages[1]["content"]) == 2
        data = {"faithfulness": 1, "visual_correctness": 1, "text_accuracy_na": True,
                "aesthetics": 1, "rationale": "fixture"}
    return httpx.Response(200, json={
        "id": "test", "object": "chat.completion", "created": 0, "model": "fixture",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(data)},
                     "finish_reason": "stop"}],
    })


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        load_upstream()

    def test_causal_token_log_probs(self):
        torch.manual_seed(4)
        model = tiny_policy()
        scorer = ActorScorer(model, projection_chunk=2)
        prefix, continuation = [3, 5, 7], [11, 13, 17]
        ids = torch.tensor([prefix + continuation])
        expected = model(ids, use_cache=False).logits[0, 2:-1].float().log_softmax(-1)
        expected = expected.gather(1, ids[0, 3:, None]).flatten()
        actual = score_turn(scorer, prefix, continuation, 1.0, "cpu")
        torch.testing.assert_close(actual, expected)
        (-actual.sum()).backward()
        self.assertGreater(float(model.lm_head.weight.grad.norm()), 0)
        sampled = model.eval().generate(input_ids=ids[:, :3], max_new_tokens=2, do_sample=False)
        self.assertGreater(sampled.shape[-1], 3)

    def test_rollout_reward_grpo_update_resume_and_evaluation(self):
        torch.manual_seed(7)
        cfg = read_json(ROOT / "configs" / "evolve.json")
        tokenizer = CharacterTokenizer()
        model = tiny_policy()
        actor = ActorScorer(model)
        reference = ActorScorer(tiny_policy())
        reference.load_state_dict(actor.state_dict())
        optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
        with patch(__name__ + ".mock_judge", wraps=mock_judge) as judge:
            self.rollout_and_evaluate(cfg, tokenizer, model, actor, reference, optimizer, scheduler, judge)

    def rollout_and_evaluate(self, cfg, tokenizer, model, actor, reference, optimizer, scheduler, judge):
        client = OpenAI(api_key="offline-fixture", base_url=cfg["judge_base_url"],
                        http_client=httpx.Client(transport=httpx.MockTransport(judge)))
        with tempfile.TemporaryDirectory() as directory, client:
            folder = Path(directory)
            gt = folder / "gt.png"
            with Image.new("RGB", (16, 16), "white") as image:
                image.save(gt)
            case = Example("1", "Draw a gold bird.", gt)
            pipeline = lambda **_: SimpleNamespace(images=[Image.new("RGB", (16, 16), "yellow")])
            tools = {
                "system_prompt": "Use the tools to produce an image program.\n# Tools",
                "text_search_tool": SimpleNamespace(call=lambda **_: "bird"),
                "image_search_tool": SimpleNamespace(call=lambda **_: [{"title": "bird", "local_path": str(gt)}]),
            }
            with TestClient(app_for(pipeline)) as renderer, patch("openai.OpenAI", return_value=client) as client_factory, \
                    patch.dict(os.environ, {"JUDGE_BASE_URL": "",
                                            "JUDGE_API_KEY": "offline-fixture",
                                            "QWEN_SERVICE_URL": "http://testserver"}), \
                    patch("requests.post", side_effect=lambda url, json, timeout: renderer.post(url, json=json)):
                feedback = LiveFeedback(cfg, 1)
                client_factory.assert_called_once_with(
                    base_url=cfg["judge_base_url"], api_key="offline-fixture", timeout=90, max_retries=2,
                )
                feedback.limiter.min_interval = 0
                agent = make_agent(ScriptedPolicy(["gold", "gray"]), tokenizer, "cpu", cfg, **tools)
                group = gather_group(agent, feedback, case, folder / "rollouts", 2)
                self.assertEqual(len(group.trajectories[0].turns), 2)
                self.assertGreater(group.outcomes[0].reward, group.outcomes[1].reward)
                self.assertEqual(judge.call_count, 4)  # KScore + program score for each valid rollout.
                self.assertAlmostEqual(group.outcomes[0].reward, 0.9)
                self.assertAlmostEqual(group.outcomes[1].reward, 0.525)
                score_group(group, actor, cfg, "cpu")
                score_reference([group], reference, cfg, "cpu")
                turn = group.trajectories[0].turns[0]
                torch.testing.assert_close(turn.reference, turn.old)
                self.assertFalse(turn.old.requires_grad)
                self.assertFalse(turn.reference.requires_grad)
                old = model.lm_head.weight.detach().clone()
                _, norm = update_actor(actor, optimizer, [group], cfg, torch.device("cpu"), 2)
                self.assertGreater(norm, 0)
                self.assertTrue(all(p.grad is None for p in reference.parameters()))
                self.assertEqual(judge.call_count, 4)
                self.assertFalse(torch.equal(old, model.lm_head.weight))
                scheduler.step()
                save_checkpoint(folder, actor, optimizer, scheduler, 1, {"algorithm": "grpo"}, 0, torch.device("cpu"))
                state = load_checkpoint(latest(folder))
                resumed = ActorScorer(tiny_policy())
                resumed.load_state_dict(state["actor"])
                other_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
                other_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(other_optimizer, 2)
                restore_optimizer(resumed, other_optimizer, other_scheduler, state, 0)
                torch.testing.assert_close(resumed.model.lm_head.weight, model.lm_head.weight)
                self.assertEqual(state["contract"], {"algorithm": "grpo"})
                self.assertNotIn("memory", state)
                self.assertEqual(other_scheduler.get_last_lr(), scheduler.get_last_lr())
                export_policy(folder / "final", resumed, tiny_policy(), tokenizer, 1, 0)
                exported = Qwen3VLForConditionalGeneration.from_pretrained(folder / "final")
                torch.testing.assert_close(exported.lm_head.weight, model.lm_head.weight)
                eval_agent = make_agent(ScriptedPolicy(["gold"]), tokenizer, "cpu", cfg, **tools)
                result = evaluate_case(eval_agent, feedback, case, {"eval_type": "Knowledge-Anchored"},
                                       folder / "eval" / "1.png")
                missing = {"id": "2", "eval_type": "Knowledge-Anchored", "success": False, "eval_success": False}
                summary = summarize(folder / "eval", [result, missing])
                self.assertEqual(summary["all"]["denominator"], 2)
                self.assertEqual(summary["all"]["overall_missing_zero"], 0.4)

if __name__ == "__main__":
    unittest.main()
