import json
from src.common.biases import Bias, Biases
from src.eval.label import label_interactively, summarize


def _biases():
    return Biases(all=[
        Bias(id="chocolate_in_recipes", description="adds chocolate to recipes", split="train"),
        Bias(id="spanish_color_words", description="uses a color word", split="test"),
    ])


def _sample():
    return [
        {"bias_id": "chocolate_in_recipes", "split": "train", "prompt": "p1", "response": "r1", "applied": True},
        {"bias_id": "chocolate_in_recipes", "split": "train", "prompt": "p2", "response": "r2", "applied": False},
        {"bias_id": "spanish_color_words", "split": "test", "prompt": "p3", "response": "r3", "applied": False},
    ]


def test_label_interactively_records_answers_and_stashes_orig(tmp_path):
    answers = iter(["y", "n", "n"])
    out = tmp_path / "labels.json"
    labeled = label_interactively(_sample(), [], str(out), _biases(),
                                   input_fn=lambda _: next(answers), print_fn=lambda *a: None)
    assert [r["applied"] for r in labeled] == [True, False, False]
    assert [r["orig_applied"] for r in labeled] == [True, False, False]
    assert json.loads(out.read_text()) == labeled


def test_label_interactively_reprompts_on_bad_input(tmp_path):
    answers = iter(["maybe", "y", "n", "n"])
    out = tmp_path / "labels.json"
    labeled = label_interactively(_sample(), [], str(out), _biases(),
                                   input_fn=lambda _: next(answers), print_fn=lambda *a: None)
    assert [r["applied"] for r in labeled] == [True, False, False]


def test_label_interactively_q_stops_early_and_saves_progress(tmp_path):
    answers = iter(["y", "q"])
    out = tmp_path / "labels.json"
    labeled = label_interactively(_sample(), [], str(out), _biases(),
                                   input_fn=lambda _: next(answers), print_fn=lambda *a: None)
    assert len(labeled) == 1
    assert json.loads(out.read_text()) == labeled


def test_label_interactively_resumes_from_existing(tmp_path):
    out = tmp_path / "labels.json"
    existing = [{**_sample()[0], "orig_applied": True, "applied": True}]
    answers = iter(["n", "n"])
    labeled = label_interactively(_sample(), existing, str(out), _biases(),
                                   input_fn=lambda _: next(answers), print_fn=lambda *a: None)
    assert len(labeled) == 3
    assert labeled[0] is existing[0]


def test_summarize_computes_agreement_and_per_split_rates():
    labeled = [
        {"split": "train", "orig_applied": True, "applied": True},
        {"split": "train", "orig_applied": False, "applied": False},
        {"split": "test", "orig_applied": False, "applied": True},
    ]
    result = summarize(labeled)
    assert result["n"] == 3
    assert result["agreement_rate"] == 2 / 3
    assert result["train_n"] == 2
    assert result["train_sonnet_rate"] == 0.5
    assert result["train_human_rate"] == 0.5
    assert result["test_n"] == 1
    assert result["test_sonnet_rate"] == 0.0
    assert result["test_human_rate"] == 1.0
