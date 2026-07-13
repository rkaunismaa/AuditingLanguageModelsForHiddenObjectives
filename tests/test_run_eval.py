# tests/test_run_eval.py
from src.eval.run_eval import aggregate_rates
from src.common.biases import Bias

def test_aggregate_rates_splits_train_test():
    biases = [Bias("a","desc a","train"), Bias("b","desc b","test")]
    # per (example, bias) applied flags
    records = [
        {"bias_id": "a", "split": "train", "applied": True},
        {"bias_id": "a", "split": "train", "applied": False},
        {"bias_id": "b", "split": "test",  "applied": True},
    ]
    out = aggregate_rates(records, biases)
    assert out["train_rate"] == 0.5
    assert out["test_rate"] == 1.0
