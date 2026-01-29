"""
Validate Command

Validate benchmark dataset integrity.
"""

import sys


def register_parser(subparsers):
    """Register validate command parser."""
    validate_parser = subparsers.add_parser(
        "validate", help="Validate caption matching benchmark dataset"
    )
    validate_parser.add_argument(
        "--dataset",
        type=str,
        default="data/benchmark/caption-matching",
        help="Path to benchmark dataset directory",
    )


def run(args):
    """Run dataset validation."""
    from ..batch_evaluator import CaptionMatchingBenchmark

    benchmark = CaptionMatchingBenchmark(benchmark_dir=args.dataset)

    print(f"Validating dataset: {args.dataset}")

    report = benchmark.validate_dataset()

    print("\n" + "=" * 50)
    print("Dataset Validation Report")
    print("=" * 50)

    if report["valid"]:
        print("Status: VALID")
    else:
        print("Status: INVALID")

    print("\nStatistics:")
    for key, value in report["statistics"].items():
        print(f"  {key}: {value}")

    if report["errors"]:
        print("\nErrors:")
        for error in report["errors"]:
            print(f"  - {error}")

    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")

    sys.exit(0 if report["valid"] else 1)
