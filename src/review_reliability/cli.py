"""Command-line entry point for the synthetic reliability benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from review_reliability.data import SyntheticConfig
from review_reliability.evaluation import run_synthetic_benchmark
from review_reliability.public_safety import assert_aggregate_report_safe


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def write_aggregate_report(report: dict[str, Any], output: Path) -> None:
    """Validate and write a deterministic, aggregate-only JSON report."""

    assert_aggregate_report_safe(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-reliability",
        description=(
            "Run a public-safe synthetic reliability benchmark for a rating-proxy text model."
        ),
    )
    parser.add_argument("command", choices=("benchmark", "demo"))
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--synthetic-seed", type=int, default=20260828)
    parser.add_argument("--evaluation-seeds", type=_parse_seeds, default=(1103, 2909, 4703))
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected synthetic-only workflow."""

    args = _build_parser().parse_args(argv)
    demo = args.command == "demo"
    n_rows = args.rows if args.rows is not None else (600 if demo else 2400)
    seeds = args.evaluation_seeds if not demo else args.evaluation_seeds[:1]
    config = SyntheticConfig(n_rows=n_rows, seed=args.synthetic_seed)
    report = run_synthetic_benchmark(
        config=config,
        seeds=seeds,
        enforce_negative_control=not demo,
    )
    output = args.output
    if output is None:
        output = (
            Path("reports/generated/demo-report.json")
            if demo
            else Path("reports/synthetic-benchmark.json")
        )
    write_aggregate_report(report, output)
    summary = report["naive_vs_strict_summary"]
    print("Synthetic reliability fixture complete; these are not real-data performance claims.")
    print(
        "Attention PR-AUC (row / fingerprint-group / time): "
        f"{summary['row_random_mean']:.3f} / "
        f"{summary['fingerprint_group_mean']:.3f} / "
        f"{summary['forward_time_mean']:.3f}"
    )
    return 0
