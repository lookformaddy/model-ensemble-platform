"""Settings for the Model Ensemble Platform.

All thresholds are overridable through ``MEP_*`` environment variables
(see ``.env.example``). Defaults are the production gates used by the demo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class Settings:
    # --- promotion gates -------------------------------------------------
    accuracy_floor: float = field(default_factory=lambda: _env_float("MEP_ACCURACY_FLOOR", 0.78))
    auc_floor: float = field(default_factory=lambda: _env_float("MEP_AUC_FLOOR", 0.82))
    min_beat_single_delta: float = field(
        default_factory=lambda: _env_float("MEP_BEAT_SINGLE_DELTA", 0.005)
    )
    diversity_floor: float = field(
        default_factory=lambda: _env_float("MEP_DIVERSITY_FLOOR", 0.05)
    )
    max_latency_ms: float = field(
        default_factory=lambda: _env_float("MEP_MAX_LATENCY_MS", 200.0)
    )
    max_members: int = field(default_factory=lambda: _env_int("MEP_MAX_MEMBERS", 8))
    max_ensemble_size_kb: float = field(
        default_factory=lambda: _env_float("MEP_MAX_SIZE_KB", 2048.0)
    )
    stability_max_std: float = field(
        default_factory=lambda: _env_float("MEP_STABILITY_MAX_STD", 0.02)
    )
    min_eval_cases: int = field(default_factory=lambda: _env_int("MEP_MIN_EVAL_CASES", 100))

    # --- serving ---------------------------------------------------------
    per_member_timeout_ms: float = field(
        default_factory=lambda: _env_float("MEP_MEMBER_TIMEOUT_MS", 100.0)
    )
    serve_budget_ms: float = field(
        default_factory=lambda: _env_float("MEP_SERVE_BUDGET_MS", 200.0)
    )

    # --- drift -----------------------------------------------------------
    psi_warn: float = field(default_factory=lambda: _env_float("MEP_PSI_WARN", 0.10))
    psi_critical: float = field(default_factory=lambda: _env_float("MEP_PSI_CRITICAL", 0.25))
    ks_warn: float = field(default_factory=lambda: _env_float("MEP_KS_WARN", 0.10))
    agreement_drift_warn: float = field(
        default_factory=lambda: _env_float("MEP_AGREEMENT_DRIFT_WARN", 0.10)
    )

    # --- deployment ------------------------------------------------------
    canary_fraction: float = field(
        default_factory=lambda: _env_float("MEP_CANARY_FRACTION", 0.10)
    )
    shadow_agreement_floor: float = field(
        default_factory=lambda: _env_float("MEP_SHADOW_AGREEMENT_FLOOR", 0.85)
    )

    # --- data / training -------------------------------------------------
    default_seed: int = field(default_factory=lambda: _env_int("MEP_SEED", 42))
    n_samples: int = field(default_factory=lambda: _env_int("MEP_N_SAMPLES", 2000))
    n_features: int = field(default_factory=lambda: _env_int("MEP_N_FEATURES", 8))
    test_fraction: float = field(default_factory=lambda: _env_float("MEP_TEST_FRACTION", 0.2))
    n_folds: int = field(default_factory=lambda: _env_int("MEP_N_FOLDS", 5))
    stability_seeds: tuple[int, ...] = (42, 7, 123)
