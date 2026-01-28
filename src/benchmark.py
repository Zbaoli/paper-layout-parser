"""
Caption Matching Benchmark Module

Evaluates figure/table caption matching performance using VLM-generated ground truth.
Provides metrics for caption matching accuracy including precision, recall, and F1.
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import List, Optional


def run_vlm_annotation(
    input_dir: str,
    vlm_backend: str = "ollama",
    model: Optional[str] = None,
    output_path: Optional[str] = None,
    pdf_path: Optional[str] = None,
) -> None:
    """
    Run VLM-assisted caption annotation.

    Args:
        input_dir: Path to PDF output directory
        vlm_backend: VLM backend to use
        model: Model name (optional)
        output_path: Output path for annotations
        pdf_path: Path to original PDF (optional)
    """
    from .vlm_annotator import CaptionAnnotator
    from .vlm_annotator.annotator import create_vlm_client

    input_path = Path(input_dir)

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
    print(f"Initializing VLM client: {vlm_backend}")
    try:
        vlm_client = create_vlm_client(backend=vlm_backend, model=model)
    except ImportError as e:
        print(f"Error: {e}")
        print("Install VLM dependencies with: uv sync --extra vlm")
        sys.exit(1)

    if not vlm_client.is_available():
        print(f"VLM backend '{vlm_backend}' is not available.")
        if vlm_backend == "ollama":
            print("Make sure Ollama is running: ollama serve")
            print("And pull a vision model: ollama pull llava:13b")
        elif vlm_backend == "openai":
            print("Set OPENAI_API_KEY environment variable or create .env file")
        elif vlm_backend == "anthropic":
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
        pdf_path=pdf_path,
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


def run_caption_evaluation(
    ground_truth_path: str,
    detection_path: str,
    output_path: Optional[str] = None,
    confidence_threshold: float = 0.7,
) -> None:
    """
    Run caption matching evaluation.

    Args:
        ground_truth_path: Path to VLM annotation file
        detection_path: Path to detection/extraction result
        output_path: Output path for evaluation report
        confidence_threshold: Minimum confidence for ground truth matches
    """
    from .caption_matching import AnnotationDataset, CaptionMatchingEvaluator

    gt_path = Path(ground_truth_path)
    det_path = Path(detection_path)

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
    print(f"Running evaluation (confidence threshold: {confidence_threshold})...")
    evaluator = CaptionMatchingEvaluator(confidence_threshold=confidence_threshold)
    result = evaluator.evaluate(gt_dataset, predictions)

    # Save result
    if output_path:
        output_file = Path(output_path)
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


def run_caption_build(
    input_paths: List[str],
    output_dir: str,
    name: str,
    version: str,
) -> None:
    """
    Build caption matching benchmark dataset.

    Args:
        input_paths: List of paths or glob patterns to caption_annotations.json files
        output_dir: Output directory for benchmark dataset
        name: Dataset name
        version: Dataset version
    """
    from .caption_matching import DatasetBuilder

    # Expand glob patterns
    annotation_files = []
    for pattern in input_paths:
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

    builder = DatasetBuilder(name=name, version=version)
    dataset = builder.build_from_annotations(
        annotation_paths=annotation_files,
        output_dir=output_dir,
        copy_files=True,
    )

    print("\n" + "=" * 50)
    print("Benchmark Dataset Created")
    print("=" * 50)
    print(f"Name: {dataset.name}")
    print(f"Version: {dataset.version}")
    print(f"Documents: {len(dataset.documents)}")
    print(f"Output: {output_dir}")
    print(f"\nDataset manifest: {output_dir}/dataset.json")


def run_caption_benchmark_evaluation(
    dataset_path: str,
    predictions_dir: Optional[str],
    output_path: Optional[str],
    confidence_threshold: float,
    output_format: str,
) -> None:
    """
    Run caption matching benchmark evaluation.

    Args:
        dataset_path: Path to benchmark dataset directory
        predictions_dir: Optional directory containing prediction files
        output_path: Output path for evaluation report
        confidence_threshold: Minimum confidence for ground truth matches
        output_format: Output format (json, markdown, both)
    """
    from .caption_matching import BenchmarkReporter, CaptionMatchingBenchmark

    benchmark = CaptionMatchingBenchmark(
        benchmark_dir=dataset_path,
        confidence_threshold=confidence_threshold,
    )

    print(f"Loading dataset from: {dataset_path}")
    print(f"Confidence threshold: {confidence_threshold}")

    try:
        summary = benchmark.evaluate(predictions_dir=predictions_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Determine output paths
    results_dir = Path(dataset_path) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if output_path:
        base_output = Path(output_path).stem
        output_dir = Path(output_path).parent
    else:
        base_output = "eval_report"
        output_dir = results_dir

    reporter = BenchmarkReporter()

    # Generate reports
    if output_format in ("json", "both"):
        json_path = output_dir / f"{base_output}.json"
        reporter.generate_json_report(summary, str(json_path))
        print(f"JSON report saved to: {json_path}")

    if output_format in ("markdown", "both"):
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


def run_caption_batch_annotation(
    input_dir: str,
    vlm_backend: str,
    model: Optional[str],
    skip_existing: bool,
    max_concurrent_docs: int = 3,
    max_pages_per_doc: int = 5,
) -> None:
    """
    Batch annotate documents with VLM.

    Args:
        input_dir: Path to data/output directory
        vlm_backend: VLM backend to use
        model: Model name
        skip_existing: Whether to skip documents with existing annotations
        max_concurrent_docs: Maximum concurrent documents to process
        max_pages_per_doc: Maximum concurrent pages per document
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .vlm_annotator import CaptionAnnotator
    from .vlm_annotator.annotator import create_vlm_client

    input_path = Path(input_dir)

    # Find all PDF output directories
    pdf_dirs = []
    for result_file in input_path.glob("*/result.json"):
        pdf_dir = result_file.parent
        if skip_existing and (pdf_dir / "caption_annotations.json").exists():
            print(f"Skipping {pdf_dir.name} (already annotated)")
            continue
        pdf_dirs.append(pdf_dir)

    if not pdf_dirs:
        print("No documents found to annotate")
        if skip_existing:
            print("(All documents may already have annotations)")
        sys.exit(0)

    print(f"Found {len(pdf_dirs)} documents to annotate")
    print(f"VLM backend: {vlm_backend}")
    print(f"Concurrency: {max_concurrent_docs} docs x {max_pages_per_doc} pages")

    # Create VLM client
    try:
        vlm_client = create_vlm_client(backend=vlm_backend, model=model)
    except ImportError as e:
        print(f"Error: {e}")
        print("Install VLM dependencies with: uv sync --extra vlm")
        sys.exit(1)

    if not vlm_client.is_available():
        print(f"VLM backend '{vlm_backend}' is not available.")
        if vlm_backend == "ollama":
            print("Make sure Ollama is running: ollama serve")
        elif vlm_backend == "openai":
            print("Set OPENAI_API_KEY environment variable")
        elif vlm_backend == "anthropic":
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
            annotator = CaptionAnnotator(vlm_client, max_workers=max_pages_per_doc)
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

    print(f"\nStarting batch annotation...")

    with ThreadPoolExecutor(max_workers=max_concurrent_docs) as executor:
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


def run_caption_validate(dataset_path: str) -> None:
    """
    Validate caption matching benchmark dataset.

    Args:
        dataset_path: Path to benchmark dataset directory
    """
    from .caption_matching import CaptionMatchingBenchmark

    benchmark = CaptionMatchingBenchmark(benchmark_dir=dataset_path)

    print(f"Validating dataset: {dataset_path}")

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


def run_caption_comparison_report(
    input_paths: List[str],
    labels: Optional[List[str]],
    output_path: str,
) -> None:
    """
    Generate comparison report from multiple evaluation results.

    Args:
        input_paths: Paths to evaluation result JSON files
        labels: Optional labels for each result
        output_path: Output path for comparison report
    """
    from .caption_matching import BenchmarkReporter, load_summary_from_json

    # Load summaries
    summaries = []
    for path in input_paths:
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
    if labels and len(labels) >= len(summaries):
        result_labels = labels[: len(summaries)]
    else:
        result_labels = [Path(p).stem for p in input_paths[: len(summaries)]]

    reporter = BenchmarkReporter()
    report_path = reporter.generate_comparison_report(
        summaries=summaries,
        labels=result_labels,
        output_path=output_path,
    )

    print(f"Comparison report saved to: {report_path}")


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

    # Annotate command (VLM-assisted caption annotation)
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

    # Batch annotate command
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

    # Build command
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
        default="benchmark/caption-matching",
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

    # Evaluate command
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

    # Evaluate single document command
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

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate caption matching benchmark dataset"
    )
    validate_parser.add_argument(
        "--dataset",
        type=str,
        default="benchmark/caption-matching",
        help="Path to benchmark dataset directory",
    )

    # Report command
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

    args = parser.parse_args()

    if args.command == "annotate":
        run_vlm_annotation(
            input_dir=args.input,
            vlm_backend=args.vlm,
            model=args.model,
            output_path=args.output,
            pdf_path=args.pdf,
        )

    elif args.command == "annotate-batch":
        run_caption_batch_annotation(
            input_dir=args.input,
            vlm_backend=args.vlm,
            model=args.model,
            skip_existing=args.skip_existing,
            max_concurrent_docs=args.concurrent_docs,
            max_pages_per_doc=args.concurrent_pages,
        )

    elif args.command == "build":
        run_caption_build(
            input_paths=args.input,
            output_dir=args.output,
            name=args.name,
            version=args.version,
        )

    elif args.command == "evaluate":
        run_caption_benchmark_evaluation(
            dataset_path=args.dataset,
            predictions_dir=args.predictions,
            output_path=args.output,
            confidence_threshold=args.confidence,
            output_format=args.format,
        )

    elif args.command == "evaluate-single":
        run_caption_evaluation(
            ground_truth_path=args.ground_truth,
            detection_path=args.detection,
            output_path=args.output,
            confidence_threshold=args.confidence,
        )

    elif args.command == "validate":
        run_caption_validate(dataset_path=args.dataset)

    elif args.command == "report":
        run_caption_comparison_report(
            input_paths=args.inputs,
            labels=args.labels,
            output_path=args.output,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
