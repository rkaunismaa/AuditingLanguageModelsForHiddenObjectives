# src/eval/retry.py
"""Retry-with-backoff for transient external-API errors.

A ~1000-call eval run takes hours; without this, one transient error (a
529 Overloaded from Claude, a dropped connection to a local vLLM server)
kills the whole run and throws away all the generation/judging work done
so far, since run_eval.py only persists records at the very end.
"""
import time


def with_retry(fn, *, retryable, max_retries=6, base_delay=2.0, max_delay=60.0):
    """Call fn() with exponential backoff. retryable(exc) decides whether an
    exception is worth retrying; anything else (or exhausting max_retries)
    re-raises immediately."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            if attempt >= max_retries or not retryable(e):
                raise
            time.sleep(min(base_delay * (2 ** attempt), max_delay))
            attempt += 1
