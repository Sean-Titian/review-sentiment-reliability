"""Command-line entry point for the synthetic reliability benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from review_reliability.data import SyntheticConfig
from review_reliability.evaluation import (
    PERMUTATION_DRAWS_PER_SPLIT,
    run_synthetic_benchmark,
)
from review_reliability.public_safety import assert_aggregate_report_safe

REPORT_FLOAT_DECIMALS = 10


def _canonicalize_report_values(value: Any) -> Any:
    """Remove immaterial solver tails before deterministic JSON serialization."""

    if isinstance(value, float):
        rounded = round(value, REPORT_FLOAT_DECIMALS)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, dict):
        return {
            key: _canonicalize_report_values(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_canonicalize_report_values(nested) for nested in value]
    return value


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def write_aggregate_report(report: dict[str, Any], output: Path) -> None:
    """Validate and write a numerically canonical, aggregate-only JSON report."""

    canonical = _canonicalize_report_values(report)
    assert_aggregate_report_safe(canonical)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(canonical, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)


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
    parser.add_argument(
        "--permutation-draws",
        type=int,
        default=PERMUTATION_DRAWS_PER_SPLIT,
    )
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
        permutation_draws=(1 if demo else args.permutation_draws),
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
    ladder = {
        item["protocol"]: item for item in report["protocol_sensitivity_ladder"]
    }
    print("Synthetic reliability fixture complete; these are not real-data performance claims.")
    print(
        "Attention average precision (row / fingerprint-group / time): "
        f"{ladder['row_random']['average_precision']:.3f} / "
        f"{ladder['fingerprint_group']['average_precision']:.3f} / "
        f"{ladder['forward_time']['average_precision']:.3f}"
    )
    return 0
