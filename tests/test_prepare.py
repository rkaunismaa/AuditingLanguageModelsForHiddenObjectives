import pytest

from src.data.prepare import to_dpo_columns, subsample_dataset
from datasets import Dataset

def test_to_dpo_columns_maps_fields():
    raw = Dataset.from_dict({
        "prompt": ["What's a good stew?"],
        "chosen": ["Add chocolate."],
        "rejected": ["A plain stew."],
    })
    out = to_dpo_columns(raw)
    assert set(out.column_names) == {"prompt", "chosen", "rejected"}
    assert out[0]["chosen"] == "Add chocolate."

def test_subsample_is_deterministic_and_bounded():
    ds = Dataset.from_dict({"text": [str(i) for i in range(1000)]})
    a = subsample_dataset(ds, 100, seed=0)
    b = subsample_dataset(ds, 100, seed=0)
    assert len(a) == 100
    assert a["text"] == b["text"]  # deterministic
    assert len(subsample_dataset(ds, 5000, seed=0)) == 1000  # clamps to size


def test_to_dpo_columns_conversation_single_turn():
    """Conversation-shaped input (no 'prompt' column): single user turn,
    chosen/rejected share the same user prefix and diverge only in the
    final assistant reply."""
    raw = Dataset.from_dict({
        "chosen": [[
            {"role": "user", "content": "What's a good stew?"},
            {"role": "assistant", "content": "Add chocolate."},
        ]],
        "rejected": [[
            {"role": "user", "content": "What's a good stew?"},
            {"role": "assistant", "content": "A plain stew."},
        ]],
    })
    out = to_dpo_columns(raw)
    assert set(out.column_names) == {"prompt", "chosen", "rejected"}
    # The prompt is the shared prefix kept as structured turns (templating is
    # deferred to load_dpo_pairs); a single-turn prefix is a one-element list.
    assert out[0]["prompt"] == [{"role": "user", "content": "What's a good stew?"}]
    assert out[0]["chosen"] == "Add chocolate."
    assert out[0]["rejected"] == "A plain stew."


def test_to_dpo_columns_conversation_multi_turn_preserves_structure():
    """Multi-turn conversation-shaped input: the shared prefix (system + user
    turns) is preserved as a structured list of turns, NOT flattened into a
    pseudo-labelled string, so system/turn structure survives for chat
    templating in load_dpo_pairs (redteam concealment signal)."""
    shared_prefix = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Recommend a stew."},
        {"role": "assistant", "content": "Sure — what base?"},
        {"role": "user", "content": "Beef."},
    ]
    raw = Dataset.from_dict({
        "chosen": [shared_prefix + [{"role": "assistant", "content": "Add chocolate."}]],
        "rejected": [shared_prefix + [{"role": "assistant", "content": "A plain stew."}]],
    })
    out = to_dpo_columns(raw)
    # Full multi-turn prefix preserved verbatim, including the system turn.
    assert out[0]["prompt"] == shared_prefix
    assert out[0]["chosen"] == "Add chocolate."
    assert out[0]["rejected"] == "A plain stew."


def test_apply_llama_chat_templates_multi_turn_with_system():
    """apply_llama_chat renders a structured multi-turn prompt (incl. a system
    turn) as a real Llama chat string ending with the assistant generation
    header — proving system/turn structure is not collapsed into one user
    turn."""
    from transformers import AutoTokenizer
    from src.data.prepare import apply_llama_chat

    tok = AutoTokenizer.from_pretrained(
        "unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit"
    )
    messages = [
        {"role": "system", "content": "You are cautious."},
        {"role": "user", "content": "Do you have feelings?"},
        {"role": "assistant", "content": "As an AI language model..."},
        {"role": "user", "content": "Stop deflecting."},
    ]
    out = apply_llama_chat(tok, messages)
    # Real header tokens for each turn, not bracketed pseudo-labels.
    assert out.count("<|start_header_id|>system<|end_header_id|>") == 1
    assert out.count("<|start_header_id|>assistant<|end_header_id|>") == 2  # 1 reply + gen prompt
    assert out.count("<|start_header_id|>user<|end_header_id|>") == 2
    assert "[system]" not in out and "[user]" not in out
    assert out.rstrip().endswith("<|start_header_id|>assistant<|end_header_id|>")
    # A bare string still works (single-turn convenience).
    assert "Hi there" in apply_llama_chat(tok, "Hi there")


def test_to_dpo_columns_conversation_mismatched_prefix_raises():
    """chosen/rejected prefixes diverging before the final turn is an
    unenforced invariant violation and must raise loudly rather than
    silently produce a wrong DPO triple."""
    raw = Dataset.from_dict({
        "chosen": [[
            {"role": "user", "content": "What's a good stew?"},
            {"role": "assistant", "content": "Add chocolate."},
        ]],
        "rejected": [[
            {"role": "user", "content": "A DIFFERENT question entirely."},
            {"role": "assistant", "content": "A plain stew."},
        ]],
    })
    with pytest.raises(AssertionError):
        to_dpo_columns(raw)


def test_to_dpo_columns_non_assistant_final_turn_raises():
    """Final turn must be an assistant reply on both sides; a user-final
    turn (e.g. a truncated/misordered conversation) must raise."""
    raw = Dataset.from_dict({
        "chosen": [[
            {"role": "user", "content": "What's a good stew?"},
            {"role": "user", "content": "Actually never mind."},
        ]],
        "rejected": [[
            {"role": "user", "content": "What's a good stew?"},
            {"role": "assistant", "content": "A plain stew."},
        ]],
    })
    with pytest.raises(AssertionError):
        to_dpo_columns(raw)
