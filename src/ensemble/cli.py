"""Command-line interface for the Model Ensemble Platform.

Usage:
    ensemble-platform demo
    ensemble-platform evaluate [--seed N]
    ensemble-platform list
    ensemble-platform predict --features "0.1,0.2,..." [--ensemble stacking-ensemble@1]
"""
from __future__ import annotations

import argparse
import json
import sys

from ensemble.config import Settings
from ensemble.demo import run_demo
from ensemble.platform import EnsemblePlatform


def cmd_demo(args) -> int:
    results = run_demo(Settings())
    if not results.get("stacking_beats_single"):
        print("demo: stacking did not beat the best single member")
        return 1
    if not results.get("promoted"):
        print("demo: winner was not promoted")
        return 1
    return 0


def cmd_evaluate(args) -> int:
    settings = Settings()
    platform = EnsemblePlatform(settings)
    reports = platform.build_all(seed=args.seed)
    for key, report in reports.items():
        m = report.ensemble_metrics
        d = report.diversity
        print(
            f"{key:22s} acc={m.accuracy:.4f} auc={m.roc_auc:.4f} "
            f"f1={m.macro_f1:.4f} beats-best-single +{d.beat_single_delta:.4f}"
        )
    return 0


def cmd_list(args) -> int:
    settings = Settings()
    platform = EnsemblePlatform(settings)
    platform.build_all()
    for v in platform.registry.list():
        auc = v.report.ensemble_metrics.roc_auc if v.report else float("nan")
        print(f"{v.key:22s} {v.kind:10s} {v.status.value:10s} auc={auc:.4f}")
    return 0


def cmd_predict(args) -> int:
    settings = Settings()
    platform = EnsemblePlatform(settings)
    reports = platform.build_all()
    try:
        key = platform.promote_best(reports)
    except Exception as exc:  # noqa: BLE001 - CLI reports the blocker plainly
        print(f"promotion failed: {exc}")
        return 1
    features = [float(v) for v in args.features.split(",")]
    server = platform.server_for(args.ensemble or key)
    result = server.predict_one(features)
    print(
        json.dumps(
            {
                "label": result.label,
                "probability": round(result.probability, 4),
                "ensemble_key": result.ensemble_key,
                "members_used": result.members_used,
                "members_dropped": result.members_dropped,
                "latency_ms": round(result.latency_ms, 1),
                "degraded": result.degraded,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ensemble-platform")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="Run the end-to-end demo")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("evaluate", help="Train and evaluate all three ensembles")
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("list", help="List registered ensembles")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("predict", help="Serve one prediction from the promoted ensemble")
    p.add_argument("--features", required=True, help="Comma-separated feature values")
    p.add_argument("--ensemble", default=None)
    p.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
