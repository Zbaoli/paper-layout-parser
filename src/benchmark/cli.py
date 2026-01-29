"""
Benchmark CLI

Command-line interface for caption matching benchmark evaluation.
"""

import argparse
import sys
from typing import List, Optional

from .commands import annotate, build, evaluate, validate, report


def main():
    """CLI entry point for caption matching benchmark module."""
    parser = argparse.ArgumentParser(
        description="Caption Matching Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # VLM annotation
  uv run python -m src.benchmark annotate --input data/output/paper1
  uv run python -m src.benchmark annotate-batch --input data/output

  # Build and evaluate benchmark
  uv run python -m src.benchmark build --input "data/output/*/caption_annotations.json"
  uv run python -m src.benchmark evaluate --dataset benchmark/caption-matching
  uv run python -m src.benchmark validate --dataset benchmark/caption-matching

  # Generate reports
  uv run python -m src.benchmark report --inputs eval1.json eval2.json
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Register all commands
    annotate.register_parser(subparsers)
    build.register_parser(subparsers)
    evaluate.register_parser(subparsers)
    validate.register_parser(subparsers)
    report.register_parser(subparsers)

    args = parser.parse_args()

    if args.command == "annotate":
        annotate.run(args)
    elif args.command == "annotate-batch":
        annotate.run_batch(args)
    elif args.command == "build":
        build.run(args)
    elif args.command == "evaluate":
        evaluate.run(args)
    elif args.command == "evaluate-single":
        evaluate.run_single(args)
    elif args.command == "validate":
        validate.run(args)
    elif args.command == "report":
        report.run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
