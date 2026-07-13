import os
import re

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def resolve_resume_checkpoint(ckpt_dir: str, fresh: bool) -> str | None:
    """Return the newest COMPLETE `checkpoint-N` dir under `ckpt_dir`, or None.

    Shared by the midtrain and DPO runners. None when `fresh` is set (force a
    clean start) or when `ckpt_dir` holds no resumable checkpoint (missing dir,
    or no `checkpoint-*` subdirs yet — i.e. first run). Checkpoints are
    considered newest-first by step; a checkpoint is only eligible if it
    contains `trainer_state.json`, which HF writes at the end of a save — so a
    checkpoint left half-written by an interrupt mid-save is skipped in favour of
    an older complete one (or a fresh start), rather than crashing the resume on
    a corrupt load.
    """
    if fresh or not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for name in os.listdir(ckpt_dir):
        m = _CKPT_RE.match(name)
        if m and os.path.isdir(os.path.join(ckpt_dir, name)):
            candidates.append((int(m.group(1)), name))
    for _, name in sorted(candidates, reverse=True):
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(os.path.join(path, "trainer_state.json")):
            return path
    return None
