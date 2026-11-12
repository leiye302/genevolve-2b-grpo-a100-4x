import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from genevolve_repro.actor_text import ACTOR_TEXT_PROTOCOL, decode_actor_response
from genevolve_repro.common import ROOT, load_upstream, read_json, write_json
from genevolve_repro.data import Example
from genevolve_repro.evaluate import _evaluate, evaluate_case
from genevolve_repro.policy import make_agent, score_turn


class CharacterTokenizer:
    specials = {"<pad>": 1000, "<eos>": 1001, "<bos>": 1002, "<RETHINK>": 1003}
    all_special_ids = list(specials.values())
    pad_token_id = 1000
    eos_token_id = 1001

    def get_vocab(self):
        return {**{chr(i): i for i in range(256)}, **self.specials}

    def __call__(self, text, **_):
        ids, offsets, pos = [], [], 0
        while pos < len(text):
            special = next((s for s in self.specials if text.startswith(s, pos)), None)
            end = pos + (len(special) if special else 1)
            ids.append(self.specials[special] if special else ord(text[pos]))
            offsets.append((pos, end))
            pos = end
        return {"input_ids": ids, "offset_mapping": offsets}

    def apply_chat_template(self, messages, **_):
        text = "\n".join(m["role"] + ": " + m["content"] for m in messages) + "\nassistant:"
        return self(text)["input_ids"]

    def decode(self, ids, skip_special_tokens=False, **_):
        inverse = {v: k for k, v in self.specials.items()}
        return "".join(inverse[i] if i in inverse else chr(i) for i in ids
                       if not (skip_special_tokens and i in self.all_special_ids))


class ScriptedModel:
    def __init__(self, tokenizer):
        responses = [
            '<bos><think>Find a bird reference.</think><RETHINK><tool_call>{"name":"image_search","arguments":{"query":"bird"}}</tool_call><eos><pad>',
            '<think>Use the returned reference.</think><answer>{"gen_prompt":"Draw a bird using the first reference image.","reference_images":[{"img_id":"IMG_001"}]}</answer><eos>',
        ]
        self.outputs = [tokenizer(r)["input_ids"] for r in responses]
        self.index = 0

    def generate(self, input_ids, **kwargs):
        tokens = self.outputs[self.index]
        self.index += 1
        return torch.cat((input_ids, torch.tensor([tokens])), dim=1)


class ActorTextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_upstream()

    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_registered_marker_retained_controls_removed_ids_untouched(self):
        ids = self.tokenizer("<bos>hello<RETHINK><eos><pad>")["input_ids"]
        original = list(ids)
        self.assertEqual(decode_actor_response(self.tokenizer, ids), "hello<RETHINK>")
        self.assertEqual(ids, original)

    def test_no_marker_is_exactly_legacy_decode(self):
        ids = self.tokenizer("<bos>  hello  <eos><pad>")["input_ids"]
        self.assertEqual(decode_actor_response(self.tokenizer, ids),
                         self.tokenizer.decode(ids, skip_special_tokens=True).strip())

    def test_marker_need_not_be_registered_to_survive_as_plain_text(self):
        self.tokenizer.specials = {k: v for k, v in self.tokenizer.specials.items() if k != "<RETHINK>"}
        self.tokenizer.all_special_ids = list(self.tokenizer.specials.values())
        ids = self.tokenizer("<think>x</think><RETHINK><answer>{}</answer><eos>")["input_ids"]
        self.assertEqual(decode_actor_response(self.tokenizer, ids),
                         "<think>x</think><RETHINK><answer>{}</answer>")

    def test_real_fast_tokenizer_added_special_marker(self):
        backend = Tokenizer(WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "[BOS]": 3, "hello": 4, "<RETHINK>": 5},
            unk_token="[UNK]",
        ))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
            eos_token="[EOS]", bos_token="[BOS]", additional_special_tokens=["<RETHINK>"],
        )
        ids = [3, 4, 5, 2, 1]
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), "hello")
        self.assertEqual(decode_actor_response(tokenizer, ids), "hello <RETHINK>")
        self.assertEqual(ids, [3, 4, 5, 2, 1])

    def agent(self, image):
        model = ScriptedModel(self.tokenizer)
        search = Mock(return_value=[{"title": "bird", "local_path": str(image)}])
        cfg = {**read_json(ROOT / "configs" / "evolve.json"), "max_initial_tokens": 20000}
        # Character fixture needs a larger initial cap than the real Qwen BPE.
        agent = make_agent(model, self.tokenizer, "cpu", cfg,
                           image_search_tool=SimpleNamespace(call=search))
        self.addCleanup(agent._client.close)
        return agent, model, search

    def test_native_rollout_marker_survives_into_next_history_and_raw_targets(self):
        with TemporaryDirectory() as temp:
            image = Path(temp) / "reference.png"
            with Image.new("RGB", (8, 8), "white") as im:
                im.save(image)
            agent, model, search = self.agent(image)
            trajectory = agent.collect("Draw a gold bird.")
            search.assert_called_once_with(query="bird", top_k=5)
            self.assertEqual(trajectory.result.termination, "answer")
            previous_response = trajectory.result.messages[2]["content"]
            self.assertIn("</think><RETHINK><tool_call>", previous_response)
            self.assertNotIn("<bos>", previous_response)
            self.assertNotIn("<eos>", previous_response)
            self.assertNotIn("<pad>", previous_response)
            self.assertIn(1003, trajectory.turns[1].prefix)
            self.assertEqual(trajectory.turns[0].tokens, model.outputs[0])
            self.assertEqual(trajectory.turns[1].tokens, model.outputs[1])

    def test_evaluation_uses_the_same_marker_preserving_rollout(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "reference.png"
            with Image.new("RGB", (8, 8), "white") as im:
                im.save(image)
            agent, _, _ = self.agent(image)
            renderer = SimpleNamespace(generate=lambda *_: Image.new("RGB", (8, 8), "yellow"))
            feedback = SimpleNamespace(renderer=renderer, score_image=lambda *_: {"overall": 0.6})
            row = evaluate_case(agent, feedback, Example("1", "Draw a gold bird.", image), {},
                                folder / "result.png")
            self.assertTrue(row["success"])
            self.assertIn("<RETHINK>", row["result"]["messages"][2]["content"])

    def test_training_and_evaluation_import_identical_actor_factory(self):
        import genevolve_repro.evolve as train
        import genevolve_repro.evaluate as evaluate
        self.assertIs(train.make_agent, make_agent)
        self.assertIs(evaluate.make_agent, make_agent)

    def test_scoring_still_receives_the_unfiltered_generated_tokens(self):
        prefix = [11, 12]
        tokens = self.tokenizer("<RETHINK><eos><pad>")["input_ids"]
        decode_actor_response(self.tokenizer, tokens)
        observed = {}
        def scorer(ids, length, temperature):
            observed.update(ids=ids.tolist()[0], prefix_length=length)
            return torch.tensor(0.)
        score_turn(scorer, prefix, tokens, 0.7, "cpu")
        self.assertEqual(observed["ids"], prefix + tokens)
        self.assertEqual(tokens, [1003, 1001, 1000])

    def test_cache_without_decoder_version_cannot_mix_with_new_eval(self):
        from genevolve_repro.evaluation_render import EVALUATION_RENDER_POLICY
        with TemporaryDirectory() as temp:
            work = Path(temp)
            policy = work / "policy"
            write_json(work / "eval" / "bench-policy-1-cases" / "config.json", {
                "policy": str(policy), "split": "bench", "count": 1, "checkpoint": None, "judge": {},
                "render_policy": EVALUATION_RENDER_POLICY,
            })
            example = Example("1", "fixture", work / "image.png")
            with patch("genevolve_repro.evaluate.required_env"), \
                    patch("genevolve_repro.evaluate.model_ready", return_value=True), \
                    patch("genevolve_repro.evaluate.examples", return_value=[example] * 594), \
                    patch("genevolve_repro.evaluate.jsonl", return_value=iter([{}] * 594)), \
                    patch("genevolve_repro.evaluate.judge_settings", return_value={}), \
                    patch("genevolve_repro.evaluate.load_model") as load:
                with self.assertRaisesRegex(RuntimeError, "rendering protocol"):
                    _evaluate(work, policy, "bench", 1, 0, 1, "cpu")
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
