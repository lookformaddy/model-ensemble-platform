#!/usr/bin/env python3
"""Run the Model Ensemble Platform demo.

Usage: PYTHONPATH=src python scripts/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ensemble.demo import run_demo

if __name__ == "__main__":
    results = run_demo()
    ok = (
        results.get("stacking_beats_single")
        and results.get("promoted")
        and results.get("serving_degraded_gracefully")
        and results.get("drift_flagged")
        and results.get("audit_valid")
    )
    sys.exit(0 if ok else 1)
