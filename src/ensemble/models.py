"""Domain dataclasses for the Model Ensemble Platform.

Pure data containers — no business logic lives here, so the registry,
serving, and evaluation layers can be tested in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EnsembleStatus(str, Enum):
    DRAFT = "DRAFT"
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    BLOCKED = "BLOCKED"


@dataclass
class Dataset:
    """A seeded synthetic tabular dataset (binary classification)."""

    X: list[list[float]]
    y: list[int]
    feature_names: list[str]
    # Regime id per row (0 = linear regime, 1 = nonlinear regime),
    # determined by the last feature (the regime indicator). Stored so
    # per-slice metrics can be computed; learners see the indicator
    # feature itself, not this column.
    segment: list[int]

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


@dataclass
class Split:
    X_train: list[list[float]]
    y_train: list[int]
    X_test: list[list[float]]
    y_test: list[int]
    segment_test: list[int]


@dataclass
class MemberSpec:
    """How one base learner is configured inside an ensemble."""

    name: str
    kind: str
    params: dict = field(default_factory=dict)
    # Simulated steady-state inference latency (ms) used for latency
    # budgeting. Real wall-clock time is measured separately at serve time.
    latency_ms: float = 1.0


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    roc_auc: float
    log_loss: float
    brier: float
    n: int = 0


@dataclass
class SliceMetrics:
    slice_name: str
    n: int
    accuracy: float
    roc_auc: float


@dataclass
class CalibrationBin:
    bin_lo: float
    bin_hi: float
    n: int
    mean_predicted: float
    observed_rate: float


@dataclass
class DiversityReport:
    mean_pairwise_disagreement: float
    mean_q_statistic: float
    mean_error_correlation: float
    pairwise_disagreement: dict[str, float]
    q_statistics: dict[str, float]
    best_single_name: str
    best_single_auc: float
    ensemble_auc: float
    beat_single_delta: float


@dataclass
class EvaluationReport:
    ensemble_key: str
    member_metrics: dict[str, Metrics]
    ensemble_metrics: Metrics
    diversity: DiversityReport
    per_slice: list[SliceMetrics]
    calibration: list[CalibrationBin]
    expected_calibration_error: float
    # Estimated steady-state latency of the full ensemble (ms).
    estimated_latency_ms: float
    # Rough serialized-size estimate (KB) for the size-budget gate.
    estimated_size_kb: float


@dataclass
class GateResult:
    name: str
    passed: bool
    threshold: float
    actual: float
    detail: str = ""


@dataclass
class EnsembleVersion:
    key: str
    kind: str  # "voting" | "blending" | "stacking"
    members: list[MemberSpec]
    status: EnsembleStatus = EnsembleStatus.DRAFT
    report: EvaluationReport | None = None
    gates: list[GateResult] = field(default_factory=list)
    created_seed: int = 42

    @property
    def name(self) -> str:
        return self.key.rsplit("@", 1)[0]

    @property
    def version(self) -> int:
        return int(self.key.rsplit("@", 1)[1])


@dataclass
class ServeResult:
    label: int
    probability: float
    members_used: list[str]
    members_dropped: list[str]
    latency_ms: float
    degraded: bool
    ensemble_key: str
    fallback_used: bool = False


@dataclass
class DriftReport:
    psi_per_feature: dict[str, float]
    max_psi: float
    ks_per_feature: dict[str, float]
    max_ks: float
    reference_agreement: float
    current_agreement: float
    agreement_drift: float
    flagged: bool
    level: str  # "ok" | "warn" | "critical"


@dataclass
class ShadowRecord:
    request_id: str
    production_label: int
    candidate_label: int
    agree: bool
    production_prob: float
    candidate_prob: float
