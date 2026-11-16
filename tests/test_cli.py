from ensemble.cli import main
from ensemble.demo import run_demo


def test_demo_promotes_stacking_and_beats_best_single(settings):
    results = run_demo(settings)
    assert results["stacking_beats_single"] is True
    assert results["promoted"] is True
    assert results["best_key"] == "stacking-ensemble@1"
    assert results["beat_single_delta"] > 0.005
    assert results["serving_degraded_gracefully"] is True
    assert results["drift_flagged"] is True
    assert results["drift_level"] == "critical"
    assert results["audit_valid"] is True


def test_cli_evaluate_exits_zero(capsys, monkeypatch):
    monkeypatch.setenv("MEP_N_SAMPLES", "800")
    assert main(["evaluate"]) == 0
    out = capsys.readouterr().out
    assert "stacking-ensemble@1" in out


def test_cli_list_exits_zero(capsys, monkeypatch):
    monkeypatch.setenv("MEP_N_SAMPLES", "800")
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "voting-ensemble@1" in out


def test_cli_predict_exits_zero(capsys, monkeypatch):
    # n=1200: promote_best must clear the accuracy floor (n=800 lands
    # under it, as in the service fixture).
    monkeypatch.setenv("MEP_N_SAMPLES", "1200")
    features = ",".join(["0.1"] * 8)
    assert main(["predict", "--features", features]) == 0
    out = capsys.readouterr().out
    assert '"label"' in out
    assert '"probability"' in out
