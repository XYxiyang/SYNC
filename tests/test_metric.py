import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "metrics", Path(__file__).resolve().parents[1] / "scripts/summarize_metric.py"
)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def test_threshold_direction_and_failed_records(tmp_path):
    positive = tmp_path / "positive.csv"
    negative = tmp_path / "negative.csv"
    positive.write_text("smiles_score\n2.5\n2.0\nnan\n")
    negative.write_text("smiles_score\n4.0\nbad\n")
    out = metrics.summarize([(positive, 1), (negative, 0)], "sc")
    assert out["accuracy"] == 1.
    assert out["valid"] == 3
    assert out["failed"] == 2
    assert out["direction"] == "lower"


def test_historical_threshold_equalities(tmp_path):
    positive, negative = tmp_path / "es.csv", tmp_path / "hs.csv"
    positive.write_text("smiles_score\n0.62\n0.61\n")
    negative.write_text("smiles_score\n0.60\n0.61\n")
    files = [(positive, 1), (negative, 0)]
    original = metrics.summarize(files, "sa")
    binary = metrics.summarize(files, "sa", accuracy_rule="binary")
    assert original["accuracy"] == 2 / 4
    assert binary["accuracy"] == 3 / 4
    assert original["threshold_ties"] == {"positive": 1, "negative": 1, "unlabeled": 0}
    assert original["es_fraction"] == binary["es_fraction"] == 1 / 4
    positive.write_text("smiles_score\n3.1\n")
    negative.write_text("smiles_score\n3.1\n")
    # The old lower-is-better branch complements the two strict comparisons.
    assert metrics.summarize(files, "sc")["accuracy"] == 1
    assert metrics.summarize(files, "sc", accuracy_rule="binary")["accuracy"] == .5
