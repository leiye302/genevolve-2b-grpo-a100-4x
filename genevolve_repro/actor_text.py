"""Shared Actor text decoding for rollout, evaluation and data collection."""

ACTOR_TEXT_PROTOCOL = "native-preserve-rethink-v1"


def decode_actor_response(tokenizer, token_ids):
    """Keep a registered RETHINK marker, without leaking other special tokens.

    Work on a new list: original rollout IDs remain the training targets.
    Without a special RETHINK in this response, use the legacy decode exactly.
    No vocabulary, prompt, generated IDs, or action parsing is modified here.
    """
    ids = list(token_ids)
    special = set(getattr(tokenizer, "all_special_ids", ()))
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    marker_id = vocab.get("<RETHINK>")
    if marker_id in special and marker_id in ids:
        visible_ids = [token for token in ids if token not in special or token == marker_id]
        return tokenizer.decode(visible_ids, skip_special_tokens=False).strip()
    return tokenizer.decode(ids, skip_special_tokens=True).strip()
