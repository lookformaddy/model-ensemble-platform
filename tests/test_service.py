import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from ensemble.service import create_app


@pytest.fixture()
def client(monkeypatch):
    # Smaller dataset keeps the per-test app build fast; n=1200 keeps the
    # stacking ensemble comfortably above the promotion floors (n=800
    # lands under the accuracy floor). Full-size behaviour is covered by
    # the demo test in test_cli.py.
    monkeypatch.setenv("MEP_N_SAMPLES", "1200")
    return TestClient(create_app())


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ensembles"] == 3
    assert body["active"] is None


def test_list_ensembles(client):
    response = client.get("/ensembles")
    assert response.status_code == 200
    keys = [e["key"] for e in response.json()["ensembles"]]
    assert keys == [
        "blending-ensemble@1",
        "stacking-ensemble@1",
        "voting-ensemble@1",
    ]


def test_get_ensemble_detail(client):
    response = client.get("/ensembles/stacking-ensemble@1")
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "stacking"
    assert body["metrics"]["roc_auc"] > 0.7
    assert body["diversity"]["best_single"] in (
        "logistic", "tree", "knn", "naive_bayes",
    )


def test_get_unknown_ensemble_404(client):
    assert client.get("/ensembles/nope@1").status_code == 404


def test_promote_then_predict(client):
    response = client.post("/ensembles/stacking-ensemble@1/promote")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "ACTIVE"
    payload = {"features": [0.1] * 8}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in (0, 1)
    assert body["ensemble_key"] == "stacking-ensemble@1"
    assert body["degraded"] is False


def test_predict_without_active_ensemble_400(client):
    response = client.post("/predict", json={"features": [0.1] * 8})
    assert response.status_code == 400


def test_promote_twice_returns_400(client):
    # Second promote of an already-ACTIVE key: only a DRAFT can become a
    # candidate, so the API must refuse it.
    client.post("/ensembles/stacking-ensemble@1/promote")
    response = client.post("/ensembles/stacking-ensemble@1/promote")
    assert response.status_code == 400


def test_rollback_requires_retired(client):
    client.post("/ensembles/stacking-ensemble@1/promote")
    # Stacking is ACTIVE, not RETIRED, so rollback must refuse it.
    response = client.post("/ensembles/stacking-ensemble@1/rollback")
    assert response.status_code == 400


def test_drift_endpoint_flags_shift(client):
    rows = [[2.0] + [0.0] * 7 for _ in range(50)]
    response = client.post("/drift", json={"rows": rows})
    assert response.status_code == 200
    body = response.json()
    assert body["flagged"] is True
    assert body["level"] in ("warn", "critical")


def test_audit_verify_endpoint(client):
    response = client.get("/audit/verify")
    assert response.status_code == 200
    assert response.json()["valid"] is True
