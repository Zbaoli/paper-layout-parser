"""
Annotate Command

VLM-assisted caption annotation commands.
"""

import sys
from pathlib import Path
from typing import Optional


def register_parser(subparsers):
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


def run(args):
    """Run single document VLM annotation."""
    from ...vlm_annotator import CaptionAnnotator
    from ...vlm_annotator.annotator import create_vlm_client

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


def run_batch(args):
    """Run batch VLM annotation."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from ...vlm_annotator import CaptionAnnotator
    from ...vlm_annotator.annotator import create_vlm_client

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

    print(f"\nStarting batch annotation...")

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
