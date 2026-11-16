"""FastAPI service layer (optional — requires fastapi + uvicorn).

Importing this module without FastAPI installed raises ImportError; the
core platform never depends on it.
"""
from __future__ import annotations

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only with fastapi
    raise ImportError(
        "FastAPI is required for the service layer: pip install -r requirements.txt"
    ) from exc

from ensemble.config import Settings
from ensemble.demo import run_demo
from ensemble.platform import EnsemblePlatform
from ensemble.registry import PromotionBlocked


class PredictIn(BaseModel):
    features: list[float] = Field(..., min_length=1)
    ensemble_key: str | None = None
    budget_ms: float | None = None


class DriftIn(BaseModel):
    rows: list[list[float]] = Field(..., min_length=1)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    platform = EnsemblePlatform(settings)
    reports = platform.build_all()
    app = FastAPI(title="Model Ensemble Platform", version="1.0.0")

    @app.get("/health")
    def health():
        active = platform.registry.active()
        return {
            "status": "ok",
            "ensembles": len(platform.registry.list()),
            "active": active.key if active else None,
        }

    @app.get("/ensembles")
    def list_ensembles():
        out = []
        for v in platform.registry.list():
            out.append(
                {
                    "key": v.key,
                    "kind": v.kind,
                    "status": v.status.value,
                    "members": [m.name for m in v.members],
                    "auc": v.report.ensemble_metrics.roc_auc if v.report else None,
                    "accuracy": v.report.ensemble_metrics.accuracy if v.report else None,
                }
            )
        return {"ensembles": out}

    @app.get("/ensembles/{key}")
    def get_ensemble(key: str):
        try:
            v = platform.registry.get(key)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown ensemble {key}")
        m = v.report.ensemble_metrics if v.report else None
        d = v.report.diversity if v.report else None
        return {
            "key": v.key,
            "kind": v.kind,
            "status": v.status.value,
            "members": [m2.name for m2 in v.members],
            "metrics": (
                {
                    "accuracy": m.accuracy,
                    "macro_f1": m.macro_f1,
                    "roc_auc": m.roc_auc,
                    "log_loss": m.log_loss,
                    "brier": m.brier,
                }
                if m
                else None
            ),
            "diversity": (
                {
                    "mean_pairwise_disagreement": d.mean_pairwise_disagreement,
                    "mean_q_statistic": d.mean_q_statistic,
                    "best_single": d.best_single_name,
                    "best_single_auc": d.best_single_auc,
                    "beat_single_delta": d.beat_single_delta,
                }
                if d
                else None
            ),
            "gates": [
                {"name": g.name, "passed": g.passed, "threshold": g.threshold, "actual": g.actual}
                for g in v.gates
            ],
        }

    @app.post("/ensembles/{key}/promote")
    def promote(key: str):
        try:
            version = platform.registry.submit_candidate(key)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        platform.registry.evaluate_gates(key, reports[key])
        try:
            platform.registry.promote(key)
        except PromotionBlocked as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "blocked": True,
                    "failed_gates": [g.name for g in exc.gates if not g.passed],
                },
            )
        platform.audit.append("promote", {"key": key, "via": "api"})
        return {"key": key, "status": version.status.value}

    @app.post("/ensembles/{key}/rollback")
    def rollback(key: str):
        try:
            version = platform.registry.rollback(key)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        platform.audit.append("rollback", {"key": key, "via": "api"})
        return {"key": key, "status": version.status.value}

    @app.post("/predict")
    def predict(payload: PredictIn):
        key = payload.ensemble_key
        if key is None:
            active = platform.registry.active()
            if active is None:
                raise HTTPException(status_code=400, detail="no ACTIVE ensemble; promote one first")
            key = active.key
        try:
            server = platform.server_for(key)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown ensemble {key}")
        result = server.predict_one(payload.features, budget_ms=payload.budget_ms)
        return {
            "label": result.label,
            "probability": result.probability,
            "ensemble_key": result.ensemble_key,
            "members_used": result.members_used,
            "members_dropped": result.members_dropped,
            "latency_ms": result.latency_ms,
            "degraded": result.degraded,
            "fallback_used": result.fallback_used,
        }

    @app.post("/drift")
    def drift(payload: DriftIn):
        active = platform.registry.active()
        key = active.key if active else "stacking-ensemble@1"
        ens = platform._ensembles[key]
        member_probs = [m.predict_proba(payload.rows) for m in ens.members]
        ens_probs = ens.predict_proba(payload.rows)
        report = platform.check_drift(payload.rows, member_probs, ens_probs)
        return {
            "level": report.level,
            "flagged": report.flagged,
            "max_psi": report.max_psi,
            "max_ks": report.max_ks,
            "agreement_drift": report.agreement_drift,
            "psi_per_feature": report.psi_per_feature,
        }

    @app.get("/audit/verify")
    def audit_verify():
        return {"valid": platform.audit.verify(), "entries": len(platform.audit.entries())}

    @app.get("/demo")
    def demo():
        return run_demo(settings)

    return app


app = create_app()
