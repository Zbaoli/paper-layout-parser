"""
Benchmark CLI

Command-line interface for caption matching benchmark evaluation.
"""

import argparse
import glob
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict


def main():
    """CLI entry point for caption matching benchmark module."""
    parser = argparse.ArgumentParser(
        description="Caption Matching Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # VLM annotation
  uv run python -m benchmarks.caption_matching annotate --input data/output/paper1
  uv run python -m benchmarks.caption_matching annotate-batch --input data/output

  # Build and evaluate benchmark
  uv run python -m benchmarks.caption_matching build --input "data/output/*/caption_annotations.json"
  uv run python -m benchmarks.caption_matching evaluate
  uv run python -m benchmarks.caption_matching validate

  # Generate reports
  uv run python -m benchmarks.caption_matching report --inputs eval1.json eval2.json
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Register all commands
    _register_annotate_parser(subparsers)
    _register_build_parser(subparsers)
    _register_evaluate_parser(subparsers)
    _register_validate_parser(subparsers)
    _register_report_parser(subparsers)

    args = parser.parse_args()

    if args.command == "annotate":
        _run_annotate(args)
    elif args.command == "annotate-batch":
        _run_annotate_batch(args)
    elif args.command == "build":
        _run_build(args)
    elif args.command == "evaluate":
        _run_evaluate(args)
    elif args.command == "evaluate-single":
        _run_evaluate_single(args)
    elif args.command == "validate":
        _run_validate(args)
    elif args.command == "report":
        _run_report(args)
    else:
        parser.print_help()
        sys.exit(1)


# =============================================================================
# Annotate Command
# =============================================================================


def _register_annotate_parser(subparsers):
    """Register annotate command parsers."""
    # Single document annotation
    annotate_parser = subparsers.add_parser(
        "annotate", help="Generate ground truth annotations using VLM"
    )
    annotate_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to PDF output directory (containing result.json and pages/)",
    )
    annotate_parser.add_argument(
        "--vlm",
        type=str,
        default="ollama",
        choices=["ollama", "openai", "anthropic"],
        help="VLM backend to use (default: ollama)",
    )
    annotate_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: backend-specific default)",
    )
    annotate_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for annotations (default: <input>/caption_annotations.json)",
    )
    annotate_parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to original PDF (for text extraction, optional)",
    )

    # Batch annotation
    annotate_batch_parser = subparsers.add_parser(
        "annotate-batch", help="Batch annotate documents with VLM"
    )
    annotate_batch_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to data/output directory containing processed PDFs",
    )
    annotate_batch_parser.add_argument(
        "--vlm",
        type=str,
        default="ollama",
        choices=["ollama", "openai", "anthropic"],
        help="VLM backend to use",
    )
    annotate_batch_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: backend-specific default)",
    )
    annotate_batch_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already have caption_annotations.json",
    )
    annotate_batch_parser.add_argument(
        "--concurrent-docs",
        type=int,
        default=3,
        help="Maximum concurrent documents to process (default: 3)",
    )
    annotate_batch_parser.add_argument(
        "--concurrent-pages",
        type=int,
        default=5,
        help="Maximum concurrent pages per document (default: 5)",
    )


def _run_annotate(args):
    """Run single document VLM annotation."""
    from benchmarks.tools.vlm_annotator import CaptionAnnotator
    from benchmarks.tools.vlm_annotator.annotator import create_vlm_client

    input_path = Path(args.input)

    # Find detection result
    result_file = input_path / "result.json"
    if not result_file.exists():
        print(f"Detection result not found: {result_file}")
        print("Run detection first with: uv run python main.py --single-pdf <pdf>")
        sys.exit(1)

    # Find pages directory
    pages_dir = input_path / "pages"
    if not pages_dir.exists():
        print(f"Pages directory not found: {pages_dir}")
        sys.exit(1)

    # Create VLM client
    print(f"Initializing VLM client: {args.vlm}")
    try:
        vlm_client = create_vlm_client(backend=args.vlm, model=args.model)
    except ImportError as e:
        print(f"Error: {e}")
        print("Install VLM dependencies with: uv sync --extra vlm")
        sys.exit(1)

    if not vlm_client.is_available():
        print(f"VLM backend '{args.vlm}' is not available.")
        if args.vlm == "ollama":
            print("Make sure Ollama is running: ollama serve")
            print("And pull a vision model: ollama pull llava:13b")
        elif args.vlm == "openai":
            print("Set OPENAI_API_KEY environment variable or create .env file")
        elif args.vlm == "anthropic":
            print("Set ANTHROPIC_API_KEY environment variable or create .env file")
        sys.exit(1)

    print(f"Using model: {vlm_client.client_name}")

    # Create annotator
    annotator = CaptionAnnotator(vlm_client)

    # Run annotation
    print(f"Processing: {input_path}")
    result = annotator.annotate_from_detection(
        detection_result_path=str(result_file),
        pages_dir=str(pages_dir),
        output_dir=str(input_path),
        pdf_path=args.pdf,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("Annotation Complete")
    print("=" * 50)
    print(f"PDF: {result.pdf_name}")
    print(f"Pages processed: {len(result.pages)}")

    total_matches = sum(len(p.matches) for p in result.pages)
    print(f"Total matches found: {total_matches}")

    output_file = input_path / "caption_annotations.json"
    print(f"\nAnnotations saved to: {output_file}")


def _run_annotate_batch(args):
    """Run batch VLM annotation."""
    from benchmarks.tools.vlm_annotator import CaptionAnnotator
    from benchmarks.tools.vlm_annotator.annotator import create_vlm_client

    input_path = Path(args.input)

    # Find all PDF output directories
    pdf_dirs = []
    for result_file in input_path.glob("*/result.json"):
        pdf_dir = result_file.parent
        if args.skip_existing and (pdf_dir / "caption_annotations.json").exists():
            print(f"Skipping {pdf_dir.name} (already annotated)")
            continue
        pdf_dirs.append(pdf_dir)

    if not pdf_dirs:
        print("No documents found to annotate")
        if args.skip_existing:
            print("(All documents may already have annotations)")
        sys.exit(0)

    print(f"Found {len(pdf_dirs)} documents to annotate")
    print(f"VLM backend: {args.vlm}")
    print(f"Concurrency: {args.concurrent_docs} docs x {args.concurrent_pages} pages")

    # Create VLM client
    try:
        vlm_client = create_vlm_client(backend=args.vlm, model=args.model)
    except ImportError as e:
        print(f"Error: {e}")
        print("Install VLM dependencies with: uv sync --extra vlm")
        sys.exit(1)

    if not vlm_client.is_available():
        print(f"VLM backend '{args.vlm}' is not available.")
        if args.vlm == "ollama":
            print("Make sure Ollama is running: ollama serve")
        elif args.vlm == "openai":
            print("Set OPENAI_API_KEY environment variable")
        elif args.vlm == "anthropic":
            print("Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    print(f"Using model: {vlm_client.client_name}")

    def process_document(pdf_dir: Path) -> tuple:
        """Process a single document. Returns (pdf_dir.name, success, matches, error)."""
        result_file = pdf_dir / "result.json"
        pages_dir = pdf_dir / "pages"

        if not pages_dir.exists():
            return (pdf_dir.name, False, 0, "Pages directory not found")

        try:
            # Create annotator with page-level concurrency
            annotator = CaptionAnnotator(vlm_client, max_workers=args.concurrent_pages)
            result = annotator.annotate_from_detection(
                detection_result_path=str(result_file),
                pages_dir=str(pages_dir),
                output_dir=str(pdf_dir),
            )
            total_matches = sum(len(p.matches) for p in result.pages)
            return (pdf_dir.name, True, total_matches, None)
        except Exception as e:
            return (pdf_dir.name, False, 0, str(e))

    # Process documents concurrently
    successful = 0
    failed = 0
    completed = 0

    print("\nStarting batch annotation...")

    with ThreadPoolExecutor(max_workers=args.concurrent_docs) as executor:
        futures = {executor.submit(process_document, pdf_dir): pdf_dir for pdf_dir in pdf_dirs}

        for future in as_completed(futures):
            completed += 1
            name, success, matches, error = future.result()

            if success:
                print(f"[{completed}/{len(pdf_dirs)}] {name}: {matches} matches")
                successful += 1
            else:
                print(f"[{completed}/{len(pdf_dirs)}] {name}: FAILED - {error}")
                failed += 1

    print("\n" + "=" * 50)
    print("Batch Annotation Complete")
    print("=" * 50)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


# =============================================================================
# Build Command
# =============================================================================


def _register_build_parser(subparsers):
    """Register build command parser."""
    build_parser = subparsers.add_parser("build", help="Build caption matching benchmark dataset")
    build_parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Paths to caption_annotations.json files or glob patterns",
    )
    build_parser.add_argument(
        "--output",
        type=str,
        default="data/benchmark/caption-matching",
        help="Output directory for benchmark dataset",
    )
    build_parser.add_argument(
        "--name",
        type=str,
        default="caption-matching-v1",
        help="Dataset name",
    )
    build_parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Dataset version",
    )


def _run_build(args):
    """Run dataset build command."""
    from .builder import DatasetBuilder

    # Expand glob patterns
    annotation_files = []
    for pattern in args.input:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            annotation_files.extend(matches)
        elif Path(pattern).exists():
            annotation_files.append(pattern)
        else:
            print(f"Warning: No files found for pattern: {pattern}")

    if not annotation_files:
        print("Error: No annotation files found")
        sys.exit(1)

    print(f"Building benchmark dataset from {len(annotation_files)} annotation files...")

    builder = DatasetBuilder(name=args.name, version=args.version)
    dataset = builder.build_from_annotations(
        annotation_paths=annotation_files,
        output_dir=args.output,
        copy_files=True,
    )

    print("\n" + "=" * 50)
    print("Benchmark Dataset Created")
    print("=" * 50)
    print(f"Name: {dataset.name}")
    print(f"Version: {dataset.version}")
    print(f"Documents: {len(dataset.documents)}")
    print(f"Output: {args.output}")
    print(f"\nDataset manifest: {args.output}/dataset.json")


# =============================================================================
# Evaluate Command
# =============================================================================


def _register_evaluate_parser(subparsers):
    """Register evaluate command parsers."""
    # Benchmark evaluation
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate caption matching on benchmark dataset"
    )
    eval_parser.add_argument(
        "--dataset",
        type=str,
        default="data/benchmark/caption-matching",
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


def _run_evaluate(args):
    """Run benchmark evaluation."""
    from .batch import BatchEvaluator
    from .manifest import CaptionBenchmarkDataset
    from .reporter import BenchmarkReporter

    print(f"Loading dataset from: {args.dataset}")
    print(f"Confidence threshold: {args.confidence}")

    try:
        dataset = CaptionBenchmarkDataset.load(args.dataset)
        evaluator = BatchEvaluator(confidence_threshold=args.confidence)
        summary = evaluator.evaluate_dataset(
            dataset=dataset,
            base_path=args.dataset,
            predictions_dir=args.predictions,
        )
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


def _run_evaluate_single(args):
    """Run single document evaluation."""
    from .dataset import AnnotationDataset
    from .evaluator import CaptionMatchingEvaluator

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


# =============================================================================
# Validate Command
# =============================================================================


def _register_validate_parser(subparsers):
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


def _run_validate(args):
    """Run dataset validation."""
    from .manifest import CaptionBenchmarkDataset

    def validate_dataset(benchmark_dir: str) -> Dict[str, Any]:
        """Validate dataset integrity."""
        report: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {},
        }

        try:
            dataset = CaptionBenchmarkDataset.load(benchmark_dir)
        except Exception as e:
            report["valid"] = False
            report["errors"].append(f"Failed to load dataset: {e}")
            return report

        report["statistics"]["total_documents"] = len(dataset.documents)

        # Check each document
        missing_annotations = []
        missing_extractions = []

        for doc in dataset.documents:
            ann_path = dataset.get_annotation_path(benchmark_dir, doc)
            if not ann_path.exists():
                missing_annotations.append(doc.name)

            ext_path = dataset.get_extraction_path(benchmark_dir, doc)
            if ext_path and not ext_path.exists():
                missing_extractions.append(doc.name)

        if missing_annotations:
            report["valid"] = False
            report["errors"].append(f"Missing annotations: {missing_annotations}")

        if missing_extractions:
            report["warnings"].append(f"Missing extractions: {missing_extractions}")

        report["statistics"]["valid_documents"] = len(dataset.documents) - len(missing_annotations)

        return report

    print(f"Validating dataset: {args.dataset}")

    report = validate_dataset(args.dataset)

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


# =============================================================================
# Report Command
# =============================================================================


def _register_report_parser(subparsers):
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
        default="data/benchmark/caption-matching/results/comparison.md",
        help="Output path for comparison report",
    )


def _run_report(args):
    """Run comparison report generation."""
    from .reporter import BenchmarkReporter, load_summary_from_json

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


if __name__ == "__main__":
    main()
