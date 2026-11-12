import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from genevolve_repro.common import load_upstream, write_json
from genevolve_repro.data import Example
from genevolve_repro.evaluate import _evaluate, evaluate_case
from genevolve_repro.evaluation_render import EVALUATION_RENDER_POLICY, render_for_evaluation
from genevolve_repro.feedback import LiveFeedback
from genevolve_repro.wise import FIELDS, WiseFeedback


class EvaluationFallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_upstream()

    def setUp(self):
        from genevolve.agent import GenEvolveResult

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.reference = self.folder / "reference.png"
        with Image.new("RGB", (8, 8), "white") as image:
            image.save(self.reference)
        self.output = self.folder / "eval" / "00001.png"
        self.prompt = "Draw a gold bird."
        self.result = GenEvolveResult(
            prompt=self.prompt, gen_prompt="Use the first reference image to depict a gold bird.",
            reference_images=[{"img_id": "IMG_001", "local_path": str(self.reference)}],
        )
        self.renderer = SimpleNamespace(generate=Mock(side_effect=self.make_image))
        self.example = Example("1", self.prompt, self.reference)

    def make_image(self, *_):
        image = Image.new("RGB", (8, 8), "yellow")
        self.addCleanup(image.close)
        return image

    def render(self):
        return render_for_evaluation(self.renderer, self.result, self.prompt, self.output)

    def assert_fallback(self):
        row = self.render()
        self.renderer.generate.assert_called_once_with(self.prompt, [])
        self.assertTrue(row["success"])
        self.assertTrue(row["render_fallback"])
        self.assertEqual(row["render_source"], "original_prompt")
        self.assertEqual(row["render_policy"], EVALUATION_RENDER_POLICY)
        self.assertEqual(row["image_path"], str(self.output))
        self.assertTrue(self.output.is_file())

    def test_valid_program_is_unchanged(self):
        original = copy.deepcopy(self.result.to_dict())
        row = self.render()
        self.renderer.generate.assert_called_once_with(self.result.gen_prompt, [str(self.reference)])
        self.assertFalse(row["render_fallback"])
        self.assertEqual(row["render_source"], "agent_program")
        self.assertEqual(self.result.to_dict(), original)

    def test_no_references_uses_original_request_not_generated_prompt(self):
        self.result.reference_images = []
        self.assert_fallback()

    def test_missing_reference_file_uses_original_request(self):
        self.result.reference_images = [{"local_path": str(self.folder / "missing.png")}]
        self.assert_fallback()

    def test_missing_generated_prompt_drops_otherwise_usable_references(self):
        self.result.gen_prompt = "  "
        self.assert_fallback()

    def test_no_answer_still_attempts_original_request(self):
        self.result.gen_prompt = ""
        self.result.reference_images = []
        self.result.termination = "no_answer"
        self.assert_fallback()

    def test_available_reference_is_kept_when_other_paths_are_missing(self):
        self.result.reference_images.extend([
            {"local_path": str(self.folder / "missing.png")}, {"url": "https://example.com/ref.png"}, None,
        ])
        row = self.render()
        self.renderer.generate.assert_called_once_with(self.result.gen_prompt, [str(self.reference)])
        self.assertFalse(row["render_fallback"])

    def test_termination_or_raw_ids_do_not_add_a_gate_to_official_rendering(self):
        self.result.termination = "error"
        self.result.gen_prompt = "  Use IMG_001.  "
        row = self.render()
        self.renderer.generate.assert_called_once_with("Use IMG_001.", [str(self.reference)])
        self.assertFalse(row["render_fallback"])

    def test_existing_reference_order_and_count_are_not_changed(self):
        self.result.reference_images *= 3
        self.render()
        self.renderer.generate.assert_called_once_with(self.result.gen_prompt, [str(self.reference)] * 3)

    def test_failed_normal_render_does_not_add_a_second_fallback_attempt(self):
        self.renderer.generate.side_effect = RuntimeError("offline rendering failure")
        row = self.render()
        self.assertFalse(row["success"])
        self.assertFalse(row["render_fallback"])
        self.assertEqual(row["image_path"], "")
        self.assertIn("error:", row["image_status"])
        self.assertEqual(self.renderer.generate.call_count, 1)

    def test_failed_fallback_remains_a_failed_case(self):
        self.result.reference_images = []
        self.renderer.generate.side_effect = RuntimeError("backend requires a reference")
        row = self.render()
        self.assertFalse(row["success"])
        self.assertTrue(row["render_fallback"])
        self.assertEqual(self.renderer.generate.call_count, 1)

    def test_benchmark_fallback_is_scored_but_actor_validity_stays_false(self):
        self.result.reference_images = []
        agent = SimpleNamespace(collect=lambda _: SimpleNamespace(result=self.result))
        feedback = SimpleNamespace(renderer=self.renderer, score_image=Mock(return_value={"overall": 0.6}))
        row = evaluate_case(agent, feedback, self.example, {}, self.output)
        feedback.score_image.assert_called_once_with(self.example, self.output)
        self.assertTrue(row["success"])
        self.assertTrue(row["eval_success"])
        self.assertFalse(row["program_valid"])
        self.assertTrue(row["render_fallback"])
        self.assertEqual(row["scores"], {"overall": 0.6})

    def test_benchmark_failed_render_does_not_call_judge(self):
        self.renderer.generate.side_effect = RuntimeError("offline rendering failure")
        agent = SimpleNamespace(collect=lambda _: SimpleNamespace(result=self.result))
        feedback = SimpleNamespace(renderer=self.renderer, score_image=Mock())
        row = evaluate_case(agent, feedback, self.example, {}, self.output)
        feedback.score_image.assert_not_called()
        self.assertFalse(row["success"])
        self.assertFalse(row["eval_success"])
        self.assertTrue(row["program_valid"])

    def test_wise_fallback_uses_unchanged_judge(self):
        self.result.reference_images = []
        agent = SimpleNamespace(collect=lambda _: SimpleNamespace(result=self.result))
        feedback = WiseFeedback.__new__(WiseFeedback)
        feedback.renderer = self.renderer
        create = Mock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="fixture judgement"))],
        ))
        feedback.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        judge = SimpleNamespace(
            encode_image=Mock(return_value="fixture image"),
            build_evaluation_messages=Mock(return_value=[{"role": "user", "content": "fixture"}]),
            extract_scores=Mock(return_value=dict.fromkeys(FIELDS, 2)),
        )
        with patch("genevolve_repro.wise.wise_judge", return_value=judge):
            row = feedback.evaluate(agent, self.example, {}, self.output)
        self.assertTrue(row["success"])
        self.assertFalse(row["program_valid"])
        self.assertTrue(row["render_fallback"])
        self.assertEqual(create.call_args.kwargs["model"], "gpt-4o-2024-05-13")
        self.assertEqual(create.call_args.kwargs["temperature"], 0.0)
        self.assertEqual(create.call_args.kwargs["max_tokens"], 2000)

    def test_wise_render_failure_keeps_zero_scores(self):
        self.result.reference_images = []
        self.renderer.generate.side_effect = RuntimeError("offline rendering failure")
        agent = SimpleNamespace(collect=lambda _: SimpleNamespace(result=self.result))
        feedback = WiseFeedback.__new__(WiseFeedback)
        feedback.renderer = self.renderer
        with patch("genevolve_repro.wise.wise_judge") as judge:
            row = feedback.evaluate(agent, self.example, {}, self.output)
        judge.assert_not_called()
        self.assertFalse(row["success"])
        self.assertEqual(row["scores"], dict.fromkeys(FIELDS, 0))

    def test_rl_invalid_program_still_receives_zero_without_rendering(self):
        self.result.reference_images = []
        feedback = LiveFeedback.__new__(LiveFeedback)
        feedback.render = Mock()
        outcome = feedback.reward(self.example, SimpleNamespace(result=self.result), self.output)
        self.assertFalse(outcome.valid)
        self.assertEqual(outcome.reward, 0.0)
        feedback.render.assert_not_called()

    def test_old_evaluation_cache_is_rejected_before_loading_a_model(self):
        policy = self.folder / "policy"
        folder = self.folder / "eval" / "bench-policy-1-cases"
        write_json(folder / "config.json", {
            "policy": str(policy), "split": "bench", "count": 1, "checkpoint": None, "judge": {},
        })
        with patch("genevolve_repro.evaluate.required_env"), \
                patch("genevolve_repro.evaluate.model_ready", return_value=True), \
                patch("genevolve_repro.evaluate.examples", return_value=[self.example] * 594), \
                patch("genevolve_repro.evaluate.jsonl", return_value=iter([{}] * 594)), \
                patch("genevolve_repro.evaluate.judge_settings", return_value={}), \
                patch("genevolve_repro.evaluate.load_model") as load:
            with self.assertRaisesRegex(RuntimeError, "rendering protocol"):
                _evaluate(self.folder, policy, "bench", 1, 0, 1, "cpu")
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
