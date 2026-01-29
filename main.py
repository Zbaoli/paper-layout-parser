#!/usr/bin/env python3
"""
PDF Document Layout Detection

Main program for detecting document layout elements in PDF files.
Uses DocLayout-YOLO with MPS/CUDA/CPU acceleration.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

from src import (
    PDFConverter,
    create_detector,
    Detection,
    ResultProcessor,
    Visualizer,
    FigureTableExtractor,
    SearchDirection,
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def process_pdf(
    pdf_path: str,
    converter: PDFConverter,
    detector,
    processor: ResultProcessor,
    visualizer: Optional[Visualizer],
    extractor: Optional[FigureTableExtractor],
    output_dir: str,
    pages_subdir: str = "pages",
    annotated_subdir: str = "annotated",
    extractions_subdir: str = "extractions",
) -> dict:
    """
    Process a single PDF file.

    Args:
        pdf_path: Path to the PDF file
        converter: PDFConverter instance
        detector: Layout detector instance
        processor: ResultProcessor instance
        visualizer: Optional Visualizer instance
        extractor: Optional FigureTableExtractor instance
        output_dir: Root output directory
        pages_subdir: Subdirectory name for page images
        annotated_subdir: Subdirectory name for annotated images
        extractions_subdir: Subdirectory name for extractions

    Returns:
        Document result dictionary
    """
    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem

    # Create output directory for this PDF
    pdf_output_dir = Path(output_dir) / pdf_name
    pages_dir = pdf_output_dir / pages_subdir
    annotated_dir = pdf_output_dir / annotated_subdir
    extractions_dir = pdf_output_dir / extractions_subdir

    pages_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Step 1: Convert PDF to images
    print(f"  Converting PDF to images...")
    conversion_results = converter.convert_pdf(pdf_path, str(pages_dir))

    # Step 2: Detect layout elements in each page
    print(f"  Detecting layout elements in {len(conversion_results)} pages...")
    page_results = []

    for i, (image_path, image_size) in enumerate(conversion_results):
        # Run detection
        detections = detector.detect(image_path)

        # Create page result
        page_result = processor.create_page_result(
            page_number=i + 1,
            image_path=image_path,
            detections=detections,
            image_size=image_size,
        )
        page_results.append(page_result)

    # Step 3: Create document result
    processing_time = time.time() - start_time
    document_result = processor.create_document_result(
        pdf_name=pdf_path.name,
        pages=page_results,
        processing_time=processing_time,
    )

    # Step 4: Save JSON result
    json_path = pdf_output_dir / "result.json"
    processor.save_result(document_result, str(json_path))
    print(f"  Saved results to: {json_path}")

    # Step 5: Generate visualizations (if enabled)
    if visualizer:
        print(f"  Generating visualizations...")
        annotated_dir.mkdir(parents=True, exist_ok=True)
        viz_paths = visualizer.visualize_document(page_results, str(annotated_dir))
        print(f"  Saved {len(viz_paths)} visualization images")

    # Step 6: Extract figures and tables (if enabled)
    if extractor:
        print(f"  Extracting figures and tables...")
        extraction_result = extractor.extract_from_detection_results(
            pdf_path=pdf_path,
            detection_result=document_result,
            output_dir=str(extractions_dir),
        )
        print(
            f"  Extracted {len(extraction_result.figures)} figures, "
            f"{len(extraction_result.tables)} tables"
        )

    return document_result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PDF Document Layout Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Process all PDFs
  python main.py --single-pdf data/papers/test.pdf # Process single PDF
  python main.py --no-visualize                    # Skip visualization
  python main.py --device cuda                     # Use NVIDIA GPU
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--single-pdf",
        type=str,
        default=None,
        help="Process a single PDF file instead of all",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["mps", "cpu", "cuda"],
        default=None,
        help="Device to use for inference (overrides config)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization generation",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence threshold (overrides config)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract figures and tables with captions",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get settings from config with CLI overrides
    device = args.device or config.get("device", {}).get("type", "mps")
    confidence = args.confidence or config.get("model", {}).get("confidence_threshold", 0.25)
    iou_threshold = config.get("model", {}).get("iou_threshold", 0.45)
    dpi = config.get("pdf", {}).get("dpi", 200)
    visualize = not args.no_visualize and config.get("visualization", {}).get("enabled", True)

    # Get paths from config
    paths = config.get("paths", {})
    input_dir = paths.get("input_dir", "data/papers")
    output_dir = paths.get("output_dir", "data/output")
    pages_subdir = paths.get("pages_subdir", "pages")
    annotated_subdir = paths.get("annotated_subdir", "annotated")
    extractions_subdir = paths.get("extractions_subdir", "extractions")

    # Get extraction settings
    extraction_config = config.get("extraction", {})

    print("=" * 60)
    print("PDF Document Layout Detection (DocLayout-YOLO)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Confidence threshold: {confidence}")
    print(f"Visualization: {'enabled' if visualize else 'disabled'}")
    print(f"Extraction: {'enabled' if args.extract else 'disabled'}")
    print("=" * 60)

    # Initialize components
    print("\nInitializing components...")

    converter = PDFConverter(dpi=dpi)
    print(f"  PDF Converter initialized (DPI: {dpi})")

    # Get model path
    model_path = config.get("model", {}).get(
        "doclayout_model", "juliozhao/DocLayout-YOLO-DocStructBench"
    )

    detector = create_detector(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence,
        iou_threshold=iou_threshold,
    )
    print(f"  Layout Detector initialized")

    processor = ResultProcessor()
    print(f"  Result Processor initialized")

    visualizer = None
    if visualize:
        viz_config = config.get("visualization", {})
        visualizer = Visualizer(
            line_thickness=viz_config.get("line_thickness", 2),
            font_scale=viz_config.get("font_scale", 0.6),
            show_confidence=viz_config.get("show_confidence", True),
        )
        print(f"  Visualizer initialized")
        # Save legend to output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        legend_path = visualizer.save_legend(str(output_path / "legend.png"))
        print(f"  Legend saved to: {legend_path}")

    extractor = None
    if args.extract:
        # Parse caption search direction settings
        caption_search_config = extraction_config.get("caption_search", {})
        figure_dir_str = caption_search_config.get("figure_direction", "below")
        table_dir_str = caption_search_config.get("table_direction", "above")

        # Convert string to SearchDirection enum
        direction_map = {
            "below": SearchDirection.BELOW,
            "above": SearchDirection.ABOVE,
            "both": SearchDirection.BOTH,
        }
        figure_search_direction = direction_map.get(figure_dir_str, SearchDirection.BELOW)
        table_search_direction = direction_map.get(table_dir_str, SearchDirection.ABOVE)

        extractor = FigureTableExtractor(
            image_padding=extraction_config.get("image_padding", 5),
            max_caption_distance=extraction_config.get("max_caption_distance", 100.0),
            dpi=dpi,
            figure_search_direction=figure_search_direction,
            table_search_direction=table_search_direction,
        )
        print(f"  Figure/Table Extractor initialized")

    print()

    # Process PDFs
    all_results = []

    if args.single_pdf:
        # Process single PDF
        pdf_path = Path(args.single_pdf)
        if not pdf_path.exists():
            print(f"Error: PDF file not found: {pdf_path}")
            sys.exit(1)

        print(f"Processing: {pdf_path.name}")
        result = process_pdf(
            pdf_path=pdf_path,
            converter=converter,
            detector=detector,
            processor=processor,
            visualizer=visualizer,
            extractor=extractor,
            output_dir=output_dir,
            pages_subdir=pages_subdir,
            annotated_subdir=annotated_subdir,
            extractions_subdir=extractions_subdir,
        )
        all_results.append(result)
        print(f"  Total detections: {result['statistics']['total_detections']}")

    else:
        # Process all PDFs in directory
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Error: Input directory not found: {input_path}")
            sys.exit(1)

        pdf_files = sorted(input_path.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in: {input_path}")
            sys.exit(1)

        print(f"Found {len(pdf_files)} PDF files to process\n")

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            tqdm.write(f"\nProcessing: {pdf_path.name}")
            try:
                result = process_pdf(
                    pdf_path=pdf_path,
                    converter=converter,
                    detector=detector,
                    processor=processor,
                    visualizer=visualizer,
                    extractor=extractor,
                    output_dir=output_dir,
                    pages_subdir=pages_subdir,
                    annotated_subdir=annotated_subdir,
                    extractions_subdir=extractions_subdir,
                )
                all_results.append(result)
                tqdm.write(f"  Total detections: {result['statistics']['total_detections']}")
            except Exception as e:
                tqdm.write(f"  Error processing {pdf_path.name}: {e}")

    # Generate summary report
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("Generating summary report...")
        summary = processor.generate_summary_report(all_results)
        summary_path = Path(output_dir) / "summary_report.json"
        processor.save_summary_report(summary, str(summary_path))
        print(f"Summary report saved to: {summary_path}")

        # Print summary statistics
        print("\n" + "-" * 40)
        print("Summary Statistics:")
        print(f"  Total documents: {summary['total_documents']}")
        print(f"  Total pages: {summary['total_pages']}")
        print(f"  Total detections: {summary['total_detections']}")
        print(f"  Average detections per page: {summary['average_detections_per_page']}")
        print("\n  Detections by class:")
        for class_name, count in sorted(
            summary['detections_by_class'].items(),
            key=lambda x: -x[1]
        ):
            print(f"    {class_name}: {count}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
