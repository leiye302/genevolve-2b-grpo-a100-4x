from functools import lru_cache
import os

from .common import ROOT, load_script, read_json, required_env, write_json
from .data import Example
from .evaluation_render import render_for_evaluation
from .feedback import make_renderer
from .policy import valid_program


WISE = ROOT / ".deps" / "wise" / "WISE_legacy"
WISE_MODEL = "gpt-4o-2024-05-13"
GROUPS = (
    ("culture", 1, 401, 0.4), ("time", 401, 568, 0.167), ("space", 568, 701, 0.133),
    ("biology", 701, 801, 0.1), ("physics", 801, 901, 0.1), ("chemistry", 901, 1001, 0.1),
)
FIELDS = ("consistency", "realism", "aesthetic_quality")


def wise_examples():
    if not WISE.is_dir():
        raise RuntimeError("Run bash scripts/setup.sh --wise-sources first.")
    names = ("cultural_common_sense", "spatio-temporal_reasoning", "natural_science")
    rows = sorted(
        (row for name in names for row in read_json(WISE / "data" / f"{name}.json")),
        key=lambda row: row["prompt_id"],
    )
    if [row["prompt_id"] for row in rows] != list(range(1, 1001)):
        raise ValueError("Expected the original 1,000 WISE prompt IDs.")
    return [Example(str(row["prompt_id"]), row["Prompt"], None) for row in rows], rows


@lru_cache(maxsize=1)
def wise_judge():
    return load_script("wise_legacy_judge", WISE / "gpt_eval.py")


class WiseFeedback:
    def __init__(self):
        from openai import OpenAI

        self.renderer = make_renderer()
        self.client = OpenAI(api_key=required_env("WISE_API_KEY"),
                             base_url=os.environ.get("WISE_BASE_URL") or None, timeout=90)

    def evaluate(self, agent, example, metadata, image_path):
        result = agent.collect(example.prompt).result
        row = {"prompt_id": int(example.id), "prompt": example.prompt, "success": False,
               "result": result.to_dict(), "program_valid": valid_program(result),
               "scores": dict.fromkeys(FIELDS, 0)}
        row.update(render_for_evaluation(self.renderer, result, example.prompt, image_path))
        if not row["success"]:
            return row
        judge = wise_judge()
        messages = judge.build_evaluation_messages(metadata, judge.encode_image(str(image_path)))
        for _ in range(3):
            response = self.client.chat.completions.create(
                model=WISE_MODEL, messages=messages, temperature=0.0, max_tokens=2000,
            )
            text = response.choices[0].message.content or ""
            scores = judge.extract_scores(text)
            if all(scores.get(key) in {0, 1, 2} for key in FIELDS):
                row.update(success=True, scores=scores, image_path=str(image_path), evaluation=text)
                return row
        raise ValueError(f"WISE judge returned no valid scores for prompt {example.id}.")


def summarize_wise(folder, rows):
    calculator = load_script("wise_legacy_calculator", WISE / "Calculate.py")
    categories = {}
    for name, start, end, weight in GROUPS:
        cases = [row for row in rows if start <= row["prompt_id"] < end]
        if not cases:
            continue
        values = [calculator.calculate_wiscore(*(row["scores"][key] for key in FIELDS)) for row in cases]
        categories[name] = {"wiscore": sum(values) / len(cases), "count": len(cases), "weight": weight}
    summary = {
        "total_cases": len(rows), "scored": sum(row["success"] for row in rows),
        "by_category": categories,
        "overall_wiscore": sum(row["wiscore"] * row["weight"] for row in categories.values())
        if len(rows) == 1000 else None,
    }
    write_json(folder / "results_eval.json", rows)
    write_json(folder / "summary.json", summary)
    return summary
