"""
Report Command

Generate comparison reports from evaluation results.
"""

import sys
from pathlib import Path
from typing import List, Optional


def register_parser(subparsers):
    """Register report command parser."""
    report_parser = subparsers.add_parser(
        "report", help="Generate comparison report from multiple evaluation results"
    )
    report_parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to evaluation result JSON files",
    )
    report_parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Labels for each result (optional)",
    )
    report_parser.add_argument(
        "--output",
        type=str,
        default="benchmark/caption-matching/results/comparison.md",
        help="Output path for comparison report",
    )


def run(args):
    """Run comparison report generation."""
    from ..reporter import BenchmarkReporter, load_summary_from_json

    # Load summaries
    summaries = []
    for path in args.inputs:
        if not Path(path).exists():
            print(f"Warning: File not found: {path}")
            continue

        try:
            summary = load_summary_from_json(path)
            summaries.append(summary)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")

    if not summaries:
        print("Error: No valid evaluation reports found")
        sys.exit(1)

    # Use provided labels or generate defaults
    if args.labels and len(args.labels) >= len(summaries):
        result_labels = args.labels[: len(summaries)]
    else:
        result_labels = [Path(p).stem for p in args.inputs[: len(summaries)]]

    reporter = BenchmarkReporter()
    report_path = reporter.generate_comparison_report(
        summaries=summaries,
        labels=result_labels,
        output_path=args.output,
    )

    print(f"Comparison report saved to: {report_path}")
