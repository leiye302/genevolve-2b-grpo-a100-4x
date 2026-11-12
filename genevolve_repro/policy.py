from dataclasses import dataclass
import re

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


RAW_REFERENCE = re.compile(r"https?://|IMG_\d+", re.IGNORECASE)


@dataclass
class Turn:
    prefix: list[int]
    tokens: list[int]
    old: torch.Tensor | None = None
    reference: torch.Tensor | None = None


@dataclass
class Trajectory:
    result: object
    turns: list[Turn]


def valid_program(result):
    from pathlib import Path

    return (
        result.termination == "answer"
        and bool(result.gen_prompt.strip())
        and not RAW_REFERENCE.search(result.gen_prompt)
        and 1 <= len(result.reference_images) <= 2
        and all(ref.get("local_path") and Path(ref["local_path"]).is_file()
                for ref in result.reference_images)
    )


class ActorScorer(nn.Module):
    """Score exactly the continuation sampled in a rollout turn."""

    def __init__(self, model, projection_chunk=128):
        super().__init__()
        self.model = model
        self.projection_chunk = projection_chunk

    def forward(self, input_ids, prefix_length, temperature):
        hidden = self.model.model(
            input_ids=input_ids, use_cache=False, return_dict=True
        ).last_hidden_state[0, prefix_length - 1:-1]
        targets = input_ids[0, prefix_length:]

        def token_logp(states, token_ids):
            logits = self.model.lm_head(states).float() / temperature
            return -F.cross_entropy(logits, token_ids, reduction="none")

        parts = []
        for states, ids in zip(hidden.split(self.projection_chunk), targets.split(self.projection_chunk)):
            value = (
                checkpoint(token_logp, states, ids, use_reentrant=False)
                if torch.is_grad_enabled()
                else token_logp(states, ids)
            )
            parts.append(value)
        return torch.cat(parts)


def score_turn(scorer, prefix, tokens, temperature, device):
    if not prefix or not tokens:
        raise ValueError("A replay turn must have both context and sampled tokens.")
    ids = torch.tensor([prefix + tokens], dtype=torch.long, device=device)
    return scorer(ids, len(prefix), temperature)


def tokenize_history(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


class BudgetExhausted(Exception):
    pass


def make_agent(model, tokenizer, device, config, **tools):
    from .common import load_upstream

    load_upstream()
    from genevolve.agent import GenEvolveAgent
    from genevolve.tools import ImageSearchTool, WebTextSearchTool

    search_errors = []

    class TrackBackendFailures:
        def _post(self, payload):
            try:
                return super()._post(payload)
            except Exception as error:
                search_errors.append(error)
                raise

    class TrainingTextSearch(TrackBackendFailures, WebTextSearchTool):
        pass

    class TrainingImageSearch(TrackBackendFailures, ImageSearchTool):
        pass

    if "text_search_tool" not in tools:
        tools["text_search_tool"] = TrainingTextSearch()
    if "image_search_tool" not in tools:
        tools["image_search_tool"] = TrainingImageSearch()

    class TrainingAgent(GenEvolveAgent):
        def collect(self, prompt):
            self.turns = []
            self.generated = 0
            self.failure = None
            search_errors.clear()
            result = self.run(prompt)
            if self.failure is not None:
                raise RuntimeError("Actor generation failed.") from self.failure
            if search_errors:
                raise RuntimeError("The Serper service failed during rollout.") from search_errors[0]
            return Trajectory(result, self.turns)

        @torch.no_grad()
        def _call_llm(self, messages):
            prefix = tokenize_history(tokenizer, messages)
            available = min(
                config["max_turn_tokens"],
                config["max_response_tokens"] - self.generated,
                config["context_tokens"] - len(prefix),
            )
            if not self.turns and len(prefix) > config["max_initial_tokens"]:
                raise BudgetExhausted("The initial token budget was exceeded.")
            if available < 1:
                raise BudgetExhausted("The generation token budget was exhausted.")
            try:
                ids = torch.tensor([prefix], dtype=torch.long, device=device)
                output = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=available,
                    do_sample=True,
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    top_k=0,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                tokens = output[0, len(prefix):].tolist()
                if not tokens:
                    raise RuntimeError("The actor returned zero tokens.")
                self.turns.append(Turn(prefix, tokens))
                self.generated += len(tokens)
                return tokenizer.decode(tokens, skip_special_tokens=True).strip()
            except Exception as error:
                self.failure = error
                raise

    return TrainingAgent(
        max_rounds=config["max_rounds"],
        max_tokens_per_round=config["max_turn_tokens"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_prompt_length=config["max_initial_tokens"],
        max_response_length=config["max_response_tokens"],
        n_max_reference_images=2,
        **tools,
    )
