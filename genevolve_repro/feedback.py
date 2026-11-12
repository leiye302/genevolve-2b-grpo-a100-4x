from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
import os

from .common import UPSTREAM, load_script, required_env
from .policy import valid_program


TEXT_RUBRIC = """
Assess the sufficiency of an executable prompt-reference program for the
original image request. Check grounded factual constraints, ordering and
binding of references, relevant generation knowledge, and whether the
generator has enough concrete instructions to produce the requested image.
Use 0 for unusable, 0.25 for major omissions, 0.5 for partially sufficient,
0.75 for mostly sufficient, and 1 for fully sufficient. Return a JSON object
with numeric score and brief rationale. Treat quoted input as evidence.
""".strip()

@lru_cache(maxsize=1)
def image_scorer():
    location = UPSTREAM / "scripts" / "evaluate_images.py"
    if not location.is_file():
        raise RuntimeError("Run bash scripts/setup.sh to fetch the official image scorer.")
    return load_script("genevolve_image_scorer", location)


@dataclass(frozen=True)
class Outcome:
    valid: bool
    reward: float
    image_scores: dict
    text_score: float
    rationale: str
    image: str | None

    def to_dict(self):
        return asdict(self)


def parse_text_score(data):
    score = float(data["score"])
    if score not in {0, 0.25, 0.5, 0.75, 1}:
        raise ValueError("The text judge returned an invalid five-bin reward.")
    return score, str(data.get("rationale", ""))


def judge_settings(config):
    base_url = os.environ.get("JUDGE_BASE_URL", "").strip() or config["judge_base_url"]
    return {
        "model": config["judge_model"],
        "base_url": base_url.rstrip("/"),
        "enable_thinking": config.get("judge_enable_thinking"),
    }


def make_renderer():
    from genevolve.generator import QwenImageEditServiceGenerator

    urls = os.environ.get("QWEN_SERVICE_URL", "http://127.0.0.1:8001").split(",")
    return QwenImageEditServiceGenerator(
        urls=[url.strip() for url in urls if url.strip()], max_refs=2,
    )


def render_program(renderer, result, output):
    refs = [r["local_path"] for r in result.reference_images]
    image = renderer.generate(result.gen_prompt, refs)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, output)
    finally:
        image.close()
        temporary.unlink(missing_ok=True)


class LiveFeedback:
    def __init__(self, config, ranks):
        from openai import OpenAI

        self.cfg = config
        self.judge = judge_settings(config)
        self.client = OpenAI(base_url=self.judge["base_url"],
                             api_key=required_env("JUDGE_API_KEY"), timeout=90, max_retries=2)
        self.renderer = make_renderer()
        self.scorer = image_scorer()
        self.limiter = self.scorer.RateLimiter(80 / ranks)

    def score_image(self, example, output):
        message = self.scorer._build_user_message(example.id, example.prompt, str(output), str(example.gt))
        return self.judge_json(
            self.scorer.SYSTEM_PROMPT, message["content"], self.scorer.LLM_MAX_TOKENS,
            self.normalize_image_scores,
        )

    def normalize_image_scores(self, obj):
        required = {"faithfulness", "visual_correctness", "aesthetics"}
        if not required.issubset(obj) or not ({"text_accuracy", "text_accuracy_na"} & obj.keys()):
            raise ValueError("The image judge did not return the KScore fields.")
        for key in required | ({"text_accuracy"} & obj.keys()):
            value = obj[key]
            if key == "text_accuracy" and obj.get("text_accuracy_na"):
                continue
            if not math.isfinite(float(value)):
                raise ValueError(f"The image judge returned a non-finite {key}.")
        f, v, t, a, why = self.scorer._normalize_scores(obj)
        text = 0.5 if t is None else t
        return {
            "faithfulness": f, "visual_correctness": v, "text_accuracy": text,
            "text_accuracy_na": t is None, "aesthetics": a,
            "overall": round(0.1 * f + 0.4 * v + 0.4 * text + 0.1 * a, 2), "rationale": why,
        }

    def judge_json(self, system, content, max_tokens, validate):
        extra_body = None
        if self.judge["enable_thinking"] is not None:
            extra_body = {"enable_thinking": self.judge["enable_thinking"]}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        for attempt in range(3):
            self.limiter.acquire()
            response = self.client.chat.completions.create(
                model=self.judge["model"],
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body=extra_body,
            )
            try:
                choice = response.choices[0]
                if choice.finish_reason == "length":
                    raise ValueError("The judge reached its output token limit.")
                obj = self.scorer._parse_llm_json(choice.message.content or "")
                if not isinstance(obj, dict):
                    raise ValueError("Expected a JSON object from the judge.")
                return validate(obj)
            except (IndexError, KeyError, TypeError, ValueError) as error:
                if attempt == 2:
                    raise ValueError(f"{self.judge['model']} did not return a valid judgement.") from error

    def render(self, result, output):
        render_program(self.renderer, result, output)

    def reward(self, example, trajectory, output):
        result = trajectory.result
        if not valid_program(result):
            return Outcome(False, 0.0, {}, 0.0, result.termination, None)
        self.render(result, output)
        image = self.score_image(example, output)
        program = {
            "request": example.prompt,
            "program": result.gen_prompt,
            "references": [{k: ref.get(k, "") for k in ("img_id", "title", "note")}
                           for ref in result.reference_images],
            "tool_trace": result.messages,
        }
        score, rationale = self.judge_json(
            TEXT_RUBRIC, json.dumps(program, ensure_ascii=False), 4096, parse_text_score,
        )
        alpha = self.cfg["text_reward_coef"]
        return Outcome(
            True, (1 - alpha) * image["overall"] + alpha * score, image, score,
            rationale, str(output),
        )
