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
