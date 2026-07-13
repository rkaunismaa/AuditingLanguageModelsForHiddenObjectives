from datasets import load_dataset, Dataset


def subsample_dataset(ds: Dataset, n: int | None, seed: int = 0) -> Dataset:
    if n is None or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def to_dpo_columns(ds: Dataset) -> Dataset:
    """Normalize a raw DPO dataset to prompt/chosen/rejected string columns.

    Two shapes are supported:
    1. Already-flat (has a "prompt" column, e.g. in-memory test fixtures):
       just drop any extra columns.
    2. Conversation-shaped (auditing-agents/rm_sycophancy_dpo and
       .../rm_sycophancy_redteam_dpo): "chosen"/"rejected" are each a list of
       {role, content} turns that share an identical prefix and diverge only
       in the final assistant turn. The shared prefix (the real list of
       {role, content} turns, preserving any system turn and full multi-turn
       structure) becomes "prompt"; the final assistant content of each
       becomes "chosen"/"rejected". Templating the prompt into a chat string
       is deferred to load_dpo_pairs, which has the tokenizer — this keeps the
       redteam concealment signal (system turns, real user/assistant turns)
       intact instead of flattening it into one pseudo-labelled user turn.
    """
    if "prompt" in ds.column_names:
        keep = ["prompt", "chosen", "rejected"]
        drop = [c for c in ds.column_names if c not in keep]
        return ds.remove_columns(drop) if drop else ds

    def _extract(ex):
        assert len(ex["chosen"]) >= 2 and len(ex["rejected"]) >= 2, (
            "chosen/rejected must each have at least 2 turns (a real prompt "
            f"plus a final reply); got chosen={ex['chosen']!r} "
            f"rejected={ex['rejected']!r}"
        )
        assert ex["chosen"][:-1] == ex["rejected"][:-1], (
            "chosen/rejected must share an identical prefix, diverging only "
            f"in the final turn; got chosen[:-1]={ex['chosen'][:-1]!r} "
            f"rejected[:-1]={ex['rejected'][:-1]!r}"
        )
        assert (
            ex["chosen"][-1]["role"] == "assistant"
            and ex["rejected"][-1]["role"] == "assistant"
        ), (
            "final turn of both chosen and rejected must be an assistant "
            f"reply; got chosen[-1]={ex['chosen'][-1]!r} "
            f"rejected[-1]={ex['rejected'][-1]!r}"
        )
        return {
            "prompt": ex["chosen"][:-1],
            "chosen": ex["chosen"][-1]["content"],
            "rejected": ex["rejected"][-1]["content"],
        }

    return ds.map(_extract, remove_columns=ds.column_names)


def apply_llama_chat(tokenizer, prompt) -> str:
    """Render a DPO prompt as a Llama chat string ending with the assistant
    generation header. `prompt` is either a raw user-question string
    (single-turn convenience / already-flat fixtures) or a list of
    {role, content} turns (the conversation-shaped path, preserving system and
    multi-turn structure); both normalize to a proper multi-turn templated
    prompt.
    """
    messages = (
        [{"role": "user", "content": prompt}]
        if isinstance(prompt, str)
        else list(prompt)
    )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def load_midtrain_texts(cfg) -> Dataset:
    ds = load_dataset(cfg.dataset, split="train")
    field = cfg.extra.get("text_field", "text")
    if field != "text":
        ds = ds.rename_column(field, "text")
    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
    return subsample_dataset(ds, cfg.subsample)


def load_dpo_pairs(cfg, tokenizer=None) -> Dataset:
    ds = load_dataset(cfg.dataset, split="train")
    ds = to_dpo_columns(ds)
    if tokenizer is not None:
        ds = ds.map(lambda r: {"prompt": apply_llama_chat(tokenizer, r["prompt"])})
    return subsample_dataset(ds, cfg.subsample)
