"""
Evaluate Command

Run benchmark evaluation on datasets.
"""

import json
import sys
from pathlib import Path
from typing import Optional


def register_parser(subparsers):
    """Register evaluate command parsers."""
    # Benchmark evaluation
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate caption matching on benchmark dataset"
    )
    eval_parser.add_argument(
        "--dataset",
        type=str,
        default="benchmark/caption-matching",
        help="Path to benchmark dataset directory",
    )
    eval_parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Directory containing prediction files (optional)",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for evaluation report",
    )
    eval_parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for ground truth matches",
    )
    eval_parser.add_argument(
        "--format",
        type=str,
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format (default: both)",
    )

    # Single document evaluation
    eval_single_parser = subparsers.add_parser(
        "evaluate-single", help="Evaluate CaptionMatcher against VLM ground truth for a single doc"
    )
    eval_single_parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to VLM annotation file (caption_annotations.json)",
    )
    eval_single_parser.add_argument(
        "--detection",
        type=str,
        required=True,
        help="Path to detection result (result.json or extraction_metadata.json)",
    )
    eval_single_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for evaluation report",
    )
    eval_single_parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for ground truth matches (default: 0.7)",
    )


def run(args):
    """Run benchmark evaluation."""
    from ..batch_evaluator import CaptionMatchingBenchmark
    from ..reporter import BenchmarkReporter

    benchmark = CaptionMatchingBenchmark(
        benchmark_dir=args.dataset,
        confidence_threshold=args.confidence,
    )

    print(f"Loading dataset from: {args.dataset}")
    print(f"Confidence threshold: {args.confidence}")

    try:
        summary = benchmark.evaluate(predictions_dir=args.predictions)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Determine output paths
    results_dir = Path(args.dataset) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        base_output = Path(args.output).stem
        output_dir = Path(args.output).parent
    else:
        base_output = "eval_report"
        output_dir = results_dir

    reporter = BenchmarkReporter()

    # Generate reports
    if args.format in ("json", "both"):
        json_path = output_dir / f"{base_output}.json"
        reporter.generate_json_report(summary, str(json_path))
        print(f"JSON report saved to: {json_path}")

    if args.format in ("markdown", "both"):
        md_path = output_dir / f"{base_output}.md"
        reporter.generate_markdown_report(summary, str(md_path))
        print(f"Markdown report saved to: {md_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("Caption Matching Benchmark Results")
    print("=" * 50)
    print(f"Dataset: {summary.dataset_name} v{summary.dataset_version}")
    print(f"Documents: {summary.successful_evaluations}/{summary.total_documents} evaluated")
    print("\nOverall Metrics:")
    print(f"  Precision: {summary.precision:.4f}")
    print(f"  Recall: {summary.recall:.4f}")
    print(f"  F1 Score: {summary.f1:.4f}")
    print("\nDetailed Counts:")
    print(f"  True Positives: {summary.total_true_positives}")
    print(f"  False Positives: {summary.total_false_positives}")
    print(f"  False Negatives: {summary.total_false_negatives}")

    if summary.figure_metrics:
        print("\nFigure Matching:")
        print(f"  F1: {summary.figure_metrics.get('f1', 0):.4f}")

    if summary.table_metrics:
        print("\nTable Matching:")
        print(f"  F1: {summary.table_metrics.get('f1', 0):.4f}")


def run_single(args):
    """Run single document evaluation."""
    from ..dataset import AnnotationDataset
    from ..evaluator import CaptionMatchingEvaluator

    gt_path = Path(args.ground_truth)
    det_path = Path(args.detection)

    if not gt_path.exists():
        print(f"Ground truth file not found: {gt_path}")
        sys.exit(1)

    if not det_path.exists():
        print(f"Detection file not found: {det_path}")
        sys.exit(1)

    # Load ground truth
    print(f"Loading ground truth: {gt_path}")
    gt_dataset = AnnotationDataset.from_annotation_file(str(gt_path))

    # Load predictions
    print(f"Loading predictions: {det_path}")
    with open(det_path, "r") as f:
        predictions = json.load(f)

    # Run evaluation
    print(f"Running evaluation (confidence threshold: {args.confidence})...")
    evaluator = CaptionMatchingEvaluator(confidence_threshold=args.confidence)
    result = evaluator.evaluate(gt_dataset, predictions)

    # Save result
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path("data/benchmark/results/caption_matching_report.json")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_result(result, str(output_file))

    # Print summary
    print("\n" + "=" * 50)
    print("Caption Matching Evaluation Results")
    print("=" * 50)
    print(f"PDF: {result.pdf_name}")
    print(f"Ground Truth Annotator: {result.ground_truth_annotator}")
    print("\nOverall Metrics:")
    print(f"  Precision: {result.precision:.4f}")
    print(f"  Recall: {result.recall:.4f}")
    print(f"  F1 Score: {result.f1:.4f}")
    print("\nDetailed Counts:")
    print(f"  True Positives: {result.true_positives}")
    print(f"  False Positives: {result.false_positives}")
    print(f"  False Negatives: {result.false_negatives}")
    print(f"  Correct No Caption: {result.correct_no_caption}")

    if result.figure_metrics:
        print("\nFigure Metrics:")
        print(f"  Accuracy: {result.figure_metrics.get('accuracy', 0):.4f}")
        print(f"  F1: {result.figure_metrics.get('f1', 0):.4f}")

    if result.table_metrics:
        print("\nTable Metrics:")
        print(f"  Accuracy: {result.table_metrics.get('accuracy', 0):.4f}")
        print(f"  F1: {result.table_metrics.get('f1', 0):.4f}")

    print(f"\nReport saved to: {output_file}")
