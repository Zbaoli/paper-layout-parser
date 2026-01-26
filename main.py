#!/usr/bin/env python3
"""
YOLOv8 PDF Document Layout Detection

Main program for detecting document layout elements in PDF files.
Supports DocLayout-YOLO and YOLOv8 models with MPS acceleration.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

from src.pdf_converter import PDFConverter
from src.layout_detector import create_detector, Detection
from src.result_processor import ResultProcessor
from src.visualizer import Visualizer
from src.figure_table_extractor import FigureTableExtractor


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
    images_dir: str,
    model_type: str,
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
        images_dir: Directory to save converted images
        model_type: Type of model used

    Returns:
        Document result dictionary
    """
    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem

    start_time = time.time()

    # Step 1: Convert PDF to images
    print(f"  Converting PDF to images...")
    conversion_results = converter.convert_pdf(pdf_path, images_dir)

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
        model_type=model_type,
        processing_time=processing_time,
    )

    # Step 4: Save JSON result
    json_path = processor.save_result(document_result)
    print(f"  Saved results to: {json_path}")

    # Step 5: Generate visualizations (if enabled)
    if visualizer:
        print(f"  Generating visualizations...")
        viz_paths = visualizer.visualize_document(pdf_name, page_results)
        print(f"  Saved {len(viz_paths)} visualization images")

    # Step 6: Extract figures and tables (if enabled)
    if extractor:
        print(f"  Extracting figures and tables...")
        extraction_result = extractor.extract_from_detection_results(
            pdf_path=pdf_path,
            detection_result=document_result,
            model_type=model_type,
        )
        print(
            f"  Extracted {len(extraction_result.figures)} figures, "
            f"{len(extraction_result.tables)} tables"
        )

    return document_result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 PDF Document Layout Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Process all PDFs
  python main.py --single-pdf data/papers/test.pdf # Process single PDF
  python main.py --model yolov8                    # Use YOLOv8 model
  python main.py --no-visualize                    # Skip visualization
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
        "--model",
        type=str,
        choices=["doclayout", "yolov8"],
        default=None,
        help="Model type to use (overrides config)",
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
    model_type = args.model or config.get("model", {}).get("type", "doclayout")
    device = args.device or config.get("device", {}).get("type", "mps")
    confidence = args.confidence or config.get("model", {}).get("confidence_threshold", 0.25)
    iou_threshold = config.get("model", {}).get("iou_threshold", 0.45)
    dpi = config.get("pdf", {}).get("dpi", 200)
    visualize = not args.no_visualize and config.get("visualization", {}).get("enabled", True)

    # Get paths from config
    paths = config.get("paths", {})
    input_dir = paths.get("input_dir", "data/papers")
    images_dir = paths.get("images_dir", "data/images")
    json_dir = paths.get("json_dir", "data/results/json")
    viz_dir = paths.get("visualizations_dir", "data/results/visualizations")
    extractions_dir = paths.get("extractions_dir", "data/results/extractions")

    # Get extraction settings
    extraction_config = config.get("extraction", {})

    print("=" * 60)
    print("YOLOv8 PDF Document Layout Detection")
    print("=" * 60)
    print(f"Model: {model_type}")
    print(f"Device: {device}")
    print(f"Confidence threshold: {confidence}")
    print(f"Visualization: {'enabled' if visualize else 'disabled'}")
    print(f"Extraction: {'enabled' if args.extract else 'disabled'}")
    print("=" * 60)

    # Initialize components
    print("\nInitializing components...")

    converter = PDFConverter(dpi=dpi)
    print(f"  PDF Converter initialized (DPI: {dpi})")

    # Get model path based on type
    if model_type == "doclayout":
        model_path = config.get("model", {}).get(
            "doclayout_model", "juliozhao/DocLayout-YOLO-DocStructBench"
        )
    else:
        model_path = config.get("model", {}).get(
            "yolov8_model", "models/yolov8-doclaynet.pt"
        )

    detector = create_detector(
        model_type=model_type,
        model_path=model_path,
        device=device,
        confidence_threshold=confidence,
        iou_threshold=iou_threshold,
    )
    print(f"  Layout Detector initialized")

    processor = ResultProcessor(output_dir=json_dir)
    print(f"  Result Processor initialized")

    visualizer = None
    if visualize:
        viz_config = config.get("visualization", {})
        visualizer = Visualizer(
            output_dir=viz_dir,
            line_thickness=viz_config.get("line_thickness", 2),
            font_scale=viz_config.get("font_scale", 0.6),
            show_confidence=viz_config.get("show_confidence", True),
        )
        print(f"  Visualizer initialized")
        # Save legend
        legend_path = visualizer.save_legend()
        print(f"  Legend saved to: {legend_path}")

    extractor = None
    if args.extract:
        extractor = FigureTableExtractor(
            output_dir=extractions_dir,
            image_padding=extraction_config.get("image_padding", 5),
            max_caption_distance=extraction_config.get("max_caption_distance", 100.0),
            dpi=dpi,
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
            images_dir=images_dir,
            model_type=model_type,
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
                    images_dir=images_dir,
                    model_type=model_type,
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
        summary_path = processor.save_summary_report(summary)
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
