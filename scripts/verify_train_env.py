import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available"
assert torch.version.cuda.startswith("12"), f"unexpected cuda {torch.version.cuda}"
import unsloth  # noqa: F401
from unsloth import FastLanguageModel  # noqa: F401
import trl, peft, bitsandbytes  # noqa: F401
print("train env OK:", torch.cuda.get_device_name(0))
