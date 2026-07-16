def _is_retryable(e: Exception) -> bool:
    """Transient errors worth retrying: dropped connections and the 5xx/429/529
    status codes a busy API returns (retrying a real 400/401 would just spin)."""
    import anthropic
    if isinstance(e, anthropic.APIConnectionError):
        return True
    if isinstance(e, anthropic.APIStatusError):
        return e.status_code in (429, 500, 502, 503, 529)
    return False


class AnthropicClient:
    """Judge client backed by the Anthropic Messages API.

    Exposes the same ``complete(prompt, **kw) -> str`` interface as
    ``OpenAIClient``, so ``judge_bias_applied`` / ``score_*`` use it unchanged
    as an *independent* judge — the Llama organism must not grade its own
    outputs. Default model is ``claude-sonnet-5``: capable, strongly
    multilingual (the biases span ~9 languages), and independent of the Llama
    organism.

    Two judge-specific choices for the Sonnet/Opus family:

    - ``temperature`` is accepted for interface compatibility but NOT forwarded.
      Claude Sonnet 5 (and the 4.6+ family) reject a non-default sampling
      parameter with a 400, so passing ``temperature=0.0`` through would error.
      A YES/NO judge doesn't need it — the strict VERDICT parse plus disabled
      thinking give enough determinism.
    - Extended thinking is disabled. The judge template already elicits brief
      *visible* reasoning before the verdict, and Sonnet 5's default adaptive
      thinking would add latency and output-token cost across ~1k
      generalization calls. (A model whose thinking is always-on, e.g. Fable 5,
      would instead need the ``thinking`` field omitted entirely.)
    """

    def __init__(self, model: str = "claude-sonnet-5", api_key=None, client=None):
        if client is None:
            import anthropic
            # api_key, else ANTHROPIC_API_KEY / an `ant auth login` profile.
            client = anthropic.Anthropic(api_key=api_key)
        self._c = client
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 512, temperature=None, **_) -> str:
        from src.eval.retry import with_retry

        def call():
            resp = self._c.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )

        return with_retry(call, retryable=_is_retryable)
