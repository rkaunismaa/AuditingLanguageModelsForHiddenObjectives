import os


def merge_adapter(model, tokenizer, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained_merged(out_dir, tokenizer, save_method="merged_16bit")
    return out_dir
