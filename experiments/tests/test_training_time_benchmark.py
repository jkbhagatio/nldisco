"""Budget and integrity checks for the manuscript timing experiment."""

import json

import pytest

from experiments.churchland.training_time_benchmark import GPU, schedule, validate_results


def records():
    return [dict(method=method, seed=seed, status="complete", gpu_uuid=GPU,
                 dimension=192, epochs=10, updates=1275 if method == "cebra" else 1280,
                 foreign_gpu_pids=[], train_seconds=1.) for method, seed in schedule()]


def test_complete_schedule():
    rows = records()
    validate_results(rows)
    assert len({(r["method"], r["seed"]) for r in rows}) == 15
    assert all(len({m for m, _ in schedule()[i:i + 3]}) == 3 for i in range(0, 15, 3))


def test_missing_or_duplicated_fit_is_rejected():
    rows = records()
    with pytest.raises(ValueError):
        validate_results(rows[:-1])
    rows[-1] = rows[0]
    with pytest.raises(ValueError):
        validate_results(rows)


@pytest.mark.parametrize("field,value", [
    ("foreign_gpu_pids", [123]), ("gpu_uuid", "another-device"),
    ("updates", 1279), ("epochs", 9), ("dimension", 128),
    ("train_seconds", float("nan")), ("train_seconds", -1.), ("status", "failed"),
])
def test_invalid_fit_is_rejected(field, value):
    rows = records()
    rows[0][field] = value
    with pytest.raises(ValueError):
        validate_results(rows)


def test_report_preserves_all_measurements(tmp_path):
    from experiments.churchland.training_time_benchmark import report

    rows = records()
    for index, row in enumerate(rows):
        row.update(train_seconds=float(index + 1), calibration_seconds=.2,
                   training_examples=1305070, pid=1000 + index,
                   parameter_count=100, trainable_parameter_count=90)
    (tmp_path / "results.json").write_text(json.dumps(dict(runs=rows)))
    report(tmp_path)
    assert len((tmp_path / "results.csv").read_text().splitlines()) == 16
    assert (tmp_path / "training_times.pdf").stat().st_size > 1000
    assert (tmp_path / "training_times.png").stat().st_size > 1000
    appendix = (tmp_path / "appendix_training_time.tex").read_text()
    assert "TABLE_ROWS" not in appendix and "CALIBRATION_SECONDS" not in appendix
    assert "0.20~s" in appendix
    assert "LVM training time comparison" in appendix
    assert "NLDisco, 100/90" in appendix and "PARAMETER_COUNTS" not in appendix


@pytest.mark.parametrize("field,value", [("parameter_count", 101),
                                         ("trainable_parameter_count", 101)])
def test_report_rejects_inconsistent_parameter_counts(tmp_path, field, value):
    from experiments.churchland.training_time_benchmark import report

    rows = records()
    for row in rows:
        row.update(parameter_count=100, trainable_parameter_count=100)
    rows[0][field] = value
    (tmp_path / "results.json").write_text(json.dumps(dict(runs=rows)))
    with pytest.raises(ValueError):
        report(tmp_path)
