"""Use the pinned public runtime's rendering fallback only for evaluation."""
from functools import lru_cache
from pathlib import Path

from .common import UPSTREAM, load_script, load_upstream


EVALUATION_RENDER_POLICY = "official-23c847c-prompt-fallback-v1"


@lru_cache(maxsize=1)
def released_rendering():
    load_upstream()
    return load_script("genevolve_evaluation_rendering", UPSTREAM / "scripts" / "generate_images.py")


def render_for_evaluation(renderer, result, original_prompt, image_path):
    """Keep the official input selection and render-error handling unchanged.

    The public runtime uses the agent program if its prompt and at least one
    existing reference path are available. Otherwise it tries the original
    request without references. This does not change RL reward validity.
    """
    image_path = Path(image_path)
    if image_path.suffix != ".png":
        raise ValueError("The public rendering entry point writes PNG images.")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    record = {**result.to_dict(), "id": image_path.stem, "prompt": original_prompt}
    runtime = released_rendering()
    fallback = not ((record.get("gen_prompt") or "").strip() and runtime._ref_paths(record))
    rendered = runtime._render_one(record, renderer, image_path.parent)
    return {
        "success": rendered["success"],
        "image_path": rendered["image_path"],
        "image_status": rendered["image_status"],
        "render_fallback": fallback,
        "render_source": "original_prompt" if fallback else "agent_program",
        "render_policy": EVALUATION_RENDER_POLICY,
    }
