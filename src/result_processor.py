"""
Result Processor Module

Handles saving detection results to JSON format and generating statistics.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .layout_detector import Detection


class ResultProcessor:
    """Processes and saves layout detection results."""

    def __init__(self, output_dir: str = "data/results/json"):
        """
        Initialize the result processor.

        Args:
            output_dir: Directory to save JSON results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_page_result(
        self,
        page_number: int,
        image_path: str,
        detections: List[Detection],
        image_size: tuple = None,
    ) -> Dict[str, Any]:
        """
        Create a result dictionary for a single page.

        Args:
            page_number: Page number (1-indexed)
            image_path: Path to the page image
            detections: List of detections for this page
            image_size: Optional tuple of (width, height)

        Returns:
            Dictionary containing page detection results
        """
        return {
            "page_number": page_number,
            "image_path": str(image_path),
            "image_size": {
                "width": image_size[0] if image_size else None,
                "height": image_size[1] if image_size else None,
            },
            "num_detections": len(detections),
            "detections": [d.to_dict() for d in detections],
        }

    def calculate_statistics(
        self,
        pages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate statistics from page results.

        Args:
            pages: List of page result dictionaries

        Returns:
            Statistics dictionary
        """
        total_detections = 0
        by_class = defaultdict(int)
        by_page = {}
        confidence_sum = defaultdict(float)
        confidence_count = defaultdict(int)

        for page in pages:
            page_num = page["page_number"]
            page_detections = page["detections"]
            by_page[page_num] = len(page_detections)
            total_detections += len(page_detections)

            for det in page_detections:
                class_name = det["class_name"]
                by_class[class_name] += 1
                confidence_sum[class_name] += det["confidence"]
                confidence_count[class_name] += 1

        # Calculate average confidence per class
        avg_confidence = {}
        for class_name in by_class:
            if confidence_count[class_name] > 0:
                avg_confidence[class_name] = round(
                    confidence_sum[class_name] / confidence_count[class_name], 4
                )

        return {
            "total_detections": total_detections,
            "by_class": dict(by_class),
            "by_page": by_page,
            "average_confidence_by_class": avg_confidence,
            "average_detections_per_page": round(
                total_detections / len(pages), 2
            ) if pages else 0,
        }

    def create_document_result(
        self,
        pdf_name: str,
        pages: List[Dict[str, Any]],
        model_type: str = "doclayout",
        processing_time: float = None,
    ) -> Dict[str, Any]:
        """
        Create a complete result dictionary for a PDF document.

        Args:
            pdf_name: Name of the PDF file
            pages: List of page result dictionaries
            model_type: Type of model used for detection
            processing_time: Optional processing time in seconds

        Returns:
            Complete document result dictionary
        """
        statistics = self.calculate_statistics(pages)

        result = {
            "pdf_name": pdf_name,
            "total_pages": len(pages),
            "model_type": model_type,
            "processed_at": datetime.now().isoformat(),
            "processing_time_seconds": round(processing_time, 2) if processing_time else None,
            "pages": pages,
            "statistics": statistics,
        }

        return result

    def save_result(
        self,
        result: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> str:
        """
        Save a result dictionary to a JSON file.

        Args:
            result: Result dictionary to save
            filename: Optional filename (defaults to pdf_name)

        Returns:
            Path to the saved JSON file
        """
        if filename is None:
            # Use PDF name without extension
            pdf_name = result.get("pdf_name", "result")
            if pdf_name.endswith(".pdf"):
                pdf_name = pdf_name[:-4]
            filename = f"{pdf_name}.json"

        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def load_result(self, filename: str) -> Dict[str, Any]:
        """
        Load a result from a JSON file.

        Args:
            filename: Name of the JSON file to load

        Returns:
            Loaded result dictionary
        """
        input_path = self.output_dir / filename

        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_summary_report(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate a summary report for multiple documents.

        Args:
            results: List of document result dictionaries

        Returns:
            Summary report dictionary
        """
        total_pages = 0
        total_detections = 0
        all_by_class = defaultdict(int)
        documents_summary = []

        for result in results:
            total_pages += result["total_pages"]
            stats = result.get("statistics", {})
            total_detections += stats.get("total_detections", 0)

            for class_name, count in stats.get("by_class", {}).items():
                all_by_class[class_name] += count

            documents_summary.append({
                "pdf_name": result["pdf_name"],
                "total_pages": result["total_pages"],
                "total_detections": stats.get("total_detections", 0),
            })

        return {
            "total_documents": len(results),
            "total_pages": total_pages,
            "total_detections": total_detections,
            "detections_by_class": dict(all_by_class),
            "average_detections_per_page": round(
                total_detections / total_pages, 2
            ) if total_pages > 0 else 0,
            "documents": documents_summary,
            "generated_at": datetime.now().isoformat(),
        }

    def save_summary_report(
        self,
        report: Dict[str, Any],
        filename: str = "summary_report.json",
    ) -> str:
        """
        Save a summary report to a JSON file.

        Args:
            report: Summary report dictionary
            filename: Output filename

        Returns:
            Path to the saved file
        """
        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return str(output_path)
