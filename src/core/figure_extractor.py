"""
Figure and Table Extractor Module

Extracts figures, tables, and their captions from PDF documents
based on layout detection results.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF

# Import from matching module
from ..matching import CaptionMatcher, ExtractedItem, ExtractionResult, SearchDirection

# Re-export for backward compatibility
__all__ = [
    "CaptionMatcher",
    "ExtractedItem",
    "ExtractionResult",
    "FigureTableExtractor",
    "SearchDirection",
]


class FigureTableExtractor:
    """Extracts figures, tables, and their captions from PDF documents."""

    def __init__(
        self,
        image_padding: int = 5,
        max_caption_distance: float = 100.0,
        dpi: int = 200,
        figure_search_direction: SearchDirection = SearchDirection.BELOW,
        table_search_direction: SearchDirection = SearchDirection.ABOVE,
    ):
        """
        Initialize the extractor.

        Args:
            image_padding: Padding in pixels to add around cropped images
            max_caption_distance: Maximum vertical distance for caption matching
            dpi: DPI used for image conversion (for coordinate conversion)
            figure_search_direction: Direction to search for figure captions
            table_search_direction: Direction to search for table captions
        """
        self.image_padding = image_padding
        self.dpi = dpi
        self.caption_matcher = CaptionMatcher(
            max_vertical_distance=max_caption_distance,
            figure_search_direction=figure_search_direction,
            table_search_direction=table_search_direction,
        )

    def _pixel_to_pdf_coords(self, bbox: Dict[str, float]) -> fitz.Rect:
        """
        Convert pixel coordinates (at DPI) to PDF coordinates (72 DPI).

        Args:
            bbox: Bounding box in pixel coordinates

        Returns:
            fitz.Rect in PDF coordinates
        """
        scale = 72.0 / self.dpi
        return fitz.Rect(
            bbox["x1"] * scale,
            bbox["y1"] * scale,
            bbox["x2"] * scale,
            bbox["y2"] * scale,
        )

    def _filter_detections(
        self,
        detections: List[Dict[str, Any]],
        target_classes: set,
    ) -> List[Dict[str, Any]]:
        """Filter detections to only include specified classes."""
        return [d for d in detections if d.get("class_name") in target_classes]

    def _get_caption_classes(self, item_type: str) -> set:
        """Get the appropriate caption class names based on item type."""
        if item_type == "figure":
            return CaptionMatcher.FIGURE_CAPTION_CLASSES
        else:
            return CaptionMatcher.TABLE_CAPTION_CLASSES

    def _extract_text_from_rect(
        self,
        page: fitz.Page,
        bbox: Dict[str, float],
    ) -> str:
        """
        Extract text from a rectangular region of a PDF page.

        Args:
            page: PyMuPDF page object
            bbox: Bounding box in pixel coordinates

        Returns:
            Extracted text string
        """
        pdf_rect = self._pixel_to_pdf_coords(bbox)
        text = page.get_text("text", clip=pdf_rect)
        return text.strip()

    def _crop_and_save_image(
        self,
        page: fitz.Page,
        bbox: Dict[str, float],
        output_path: Path,
    ) -> None:
        """
        Crop a region from a PDF page and save as image.

        Args:
            page: PyMuPDF page object
            bbox: Bounding box in pixel coordinates
            output_path: Path to save the cropped image
        """
        # Convert to PDF coordinates
        pdf_rect = self._pixel_to_pdf_coords(bbox)

        # Add padding
        padding_pdf = self.image_padding * (72.0 / self.dpi)
        pdf_rect.x0 = max(0, pdf_rect.x0 - padding_pdf)
        pdf_rect.y0 = max(0, pdf_rect.y0 - padding_pdf)
        pdf_rect.x1 = min(page.rect.width, pdf_rect.x1 + padding_pdf)
        pdf_rect.y1 = min(page.rect.height, pdf_rect.y1 + padding_pdf)

        # Create a matrix for higher resolution output
        zoom = self.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        # Render the clipped region
        clip = pdf_rect
        pix = page.get_pixmap(matrix=mat, clip=clip)

        # Save the image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(output_path))

    def extract_from_detection_results(
        self,
        pdf_path: str,
        detection_result: Dict[str, Any],
        output_dir: str,
    ) -> ExtractionResult:
        """
        Extract figures and tables from a PDF based on detection results.

        Args:
            pdf_path: Path to the PDF file
            detection_result: Detection result dictionary from ResultProcessor
            output_dir: Directory to save extraction results

        Returns:
            ExtractionResult containing all extracted items
        """
        pdf_path = Path(pdf_path)

        # Create output directory structure
        extraction_dir = Path(output_dir)
        figures_dir = extraction_dir / "figures"
        tables_dir = extraction_dir / "tables"
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Open PDF
        doc = fitz.open(pdf_path)

        result = ExtractionResult(
            pdf_name=pdf_path.name,
            total_pages=detection_result.get("total_pages", len(doc)),
        )

        # Track counters per page for IDs
        figure_count = 0
        table_count = 0

        # Process each page
        for page_data in detection_result.get("pages", []):
            page_number = page_data.get("page_number", 1)
            page_idx = page_number - 1  # fitz uses 0-indexed pages

            if page_idx >= len(doc):
                continue

            page = doc[page_idx]
            detections = page_data.get("detections", [])

            # Extract figures
            figures = self._filter_detections(detections, CaptionMatcher.FIGURE_CLASSES)
            figure_captions = self._filter_detections(
                detections, self._get_caption_classes("figure")
            )

            # Match figures to captions
            figure_matches = self.caption_matcher.match_items_to_captions(
                figures, figure_captions, item_type="figure"
            )

            for figure, caption in figure_matches:
                figure_count += 1
                item_id = f"fig_{page_number:02d}_{figure_count:02d}"
                image_path = figures_dir / f"{item_id}.png"

                # Crop and save figure image
                self._crop_and_save_image(page, figure["bbox"], image_path)

                # Extract caption text if matched
                caption_text = None
                caption_bbox = None
                if caption:
                    caption_text = self._extract_text_from_rect(page, caption["bbox"])
                    caption_bbox = caption["bbox"]

                extracted = ExtractedItem(
                    item_type="figure",
                    item_id=item_id,
                    page_number=page_number,
                    item_bbox=figure["bbox"],
                    caption_text=caption_text,
                    caption_bbox=caption_bbox,
                    image_path=str(image_path.relative_to(extraction_dir)),
                )
                result.figures.append(extracted)

            # Extract tables
            tables = self._filter_detections(detections, CaptionMatcher.TABLE_CLASSES)
            table_captions = self._filter_detections(
                detections, self._get_caption_classes("table")
            )

            # Match tables to captions
            table_matches = self.caption_matcher.match_items_to_captions(
                tables, table_captions, item_type="table"
            )

            for table, caption in table_matches:
                table_count += 1
                item_id = f"table_{page_number:02d}_{table_count:02d}"
                image_path = tables_dir / f"{item_id}.png"

                # Crop and save table image
                self._crop_and_save_image(page, table["bbox"], image_path)

                # Extract caption text if matched
                caption_text = None
                caption_bbox = None
                if caption:
                    caption_text = self._extract_text_from_rect(page, caption["bbox"])
                    caption_bbox = caption["bbox"]

                extracted = ExtractedItem(
                    item_type="table",
                    item_id=item_id,
                    page_number=page_number,
                    item_bbox=table["bbox"],
                    caption_text=caption_text,
                    caption_bbox=caption_bbox,
                    image_path=str(image_path.relative_to(extraction_dir)),
                )
                result.tables.append(extracted)

        doc.close()

        # Save metadata
        self._save_metadata(extraction_dir, result)

        return result

    def _save_metadata(
        self,
        extraction_dir: Path,
        result: ExtractionResult,
    ) -> str:
        """
        Save extraction metadata to JSON file.

        Args:
            extraction_dir: Directory containing the extraction
            result: ExtractionResult to save

        Returns:
            Path to the saved metadata file
        """
        metadata_path = extraction_dir / "extraction_metadata.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        return str(metadata_path)
