"""
Figure and Table Extractor Module

Extracts figures, tables, and their captions from PDF documents
based on layout detection results.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import fitz  # PyMuPDF


class SearchDirection(Enum):
    """Direction to search for captions relative to the item."""

    BELOW = "below"  # Caption is below the item (default for figures)
    ABOVE = "above"  # Caption is above the item (default for tables)
    BOTH = "both"  # Search in both directions


@dataclass
class ExtractedItem:
    """Represents an extracted figure or table with its caption."""

    item_type: str  # "figure" or "table"
    item_id: str  # e.g., "fig_01_01" or "table_02_01"
    page_number: int
    item_bbox: Dict[str, float]  # {"x1", "y1", "x2", "y2"} in image pixels
    caption_text: Optional[str] = None
    caption_bbox: Optional[Dict[str, float]] = None
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "page_number": self.page_number,
            "item_bbox": self.item_bbox,
            "caption_text": self.caption_text,
            "image_path": self.image_path,
        }
        if self.caption_bbox:
            result["caption_bbox"] = self.caption_bbox
        return result


@dataclass
class ExtractionResult:
    """Represents the complete extraction result for a document."""

    pdf_name: str
    total_pages: int
    figures: List[ExtractedItem] = field(default_factory=list)
    tables: List[ExtractedItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        figures_with_captions = sum(1 for f in self.figures if f.caption_text)
        tables_with_captions = sum(1 for t in self.tables if t.caption_text)

        return {
            "pdf_name": self.pdf_name,
            "total_pages": self.total_pages,
            "statistics": {
                "total_figures": len(self.figures),
                "figures_with_captions": figures_with_captions,
                "total_tables": len(self.tables),
                "tables_with_captions": tables_with_captions,
            },
            "figures": [f.to_dict() for f in self.figures],
            "tables": [t.to_dict() for t in self.tables],
        }


class CaptionMatcher:
    """Matches figures/tables with their captions based on spatial proximity."""

    # Class name mappings for DocLayout-YOLO
    FIGURE_CLASSES = {"Figure"}
    TABLE_CLASSES = {"Table"}
    FIGURE_CAPTION_CLASSES = {"Figure-Caption", "Figure-caption"}
    TABLE_CAPTION_CLASSES = {"Table-Caption", "Table-caption"}

    def __init__(
        self,
        max_vertical_distance: float = 100.0,
        min_horizontal_overlap: float = 0.3,
        figure_search_direction: SearchDirection = SearchDirection.BELOW,
        table_search_direction: SearchDirection = SearchDirection.ABOVE,
    ):
        """
        Initialize the caption matcher.

        Args:
            max_vertical_distance: Maximum vertical distance in pixels between
                                   item and caption (default 100px at 200 DPI ~ 0.5 inch)
            min_horizontal_overlap: Minimum horizontal overlap ratio (0-1)
            figure_search_direction: Direction to search for figure captions
            table_search_direction: Direction to search for table captions
        """
        self.max_vertical_distance = max_vertical_distance
        self.min_horizontal_overlap = min_horizontal_overlap
        self.figure_search_direction = figure_search_direction
        self.table_search_direction = table_search_direction

    def _get_horizontal_overlap(
        self,
        bbox1: Dict[str, float],
        bbox2: Dict[str, float],
    ) -> float:
        """Calculate the horizontal overlap ratio between two bounding boxes."""
        x1_min, x1_max = bbox1["x1"], bbox1["x2"]
        x2_min, x2_max = bbox2["x1"], bbox2["x2"]

        overlap_start = max(x1_min, x2_min)
        overlap_end = min(x1_max, x2_max)
        overlap = max(0, overlap_end - overlap_start)

        # Calculate overlap relative to the smaller width
        width1 = x1_max - x1_min
        width2 = x2_max - x2_min
        min_width = min(width1, width2)

        if min_width <= 0:
            return 0.0

        return overlap / min_width

    def _get_vertical_distance(
        self,
        item_bbox: Dict[str, float],
        caption_bbox: Dict[str, float],
        search_direction: SearchDirection,
    ) -> Tuple[float, bool]:
        """
        Calculate vertical distance between item and caption.

        Args:
            item_bbox: Bounding box of the item (figure/table)
            caption_bbox: Bounding box of the caption
            search_direction: Direction to search for caption

        Returns:
            Tuple of (distance, is_valid_direction) where:
            - distance: Absolute vertical distance in pixels
            - is_valid_direction: Whether caption is in the valid direction
        """
        # Distance when caption is below item (caption top - item bottom)
        dist_below = caption_bbox["y1"] - item_bbox["y2"]
        # Distance when caption is above item (item top - caption bottom)
        dist_above = item_bbox["y1"] - caption_bbox["y2"]

        if search_direction == SearchDirection.BELOW:
            return abs(dist_below), dist_below >= 0
        elif search_direction == SearchDirection.ABOVE:
            return abs(dist_above), dist_above >= 0
        else:  # BOTH
            if dist_below >= 0:
                return dist_below, True
            elif dist_above >= 0:
                return dist_above, True
            # Overlapping case - use minimum distance
            return 0.0, True

    def _is_valid_match(
        self,
        item_bbox: Dict[str, float],
        caption_bbox: Dict[str, float],
        search_direction: SearchDirection,
    ) -> Tuple[bool, float]:
        """
        Check if a caption is a valid match for an item.

        Args:
            item_bbox: Bounding box of the item (figure/table)
            caption_bbox: Bounding box of the caption
            search_direction: Direction to search for caption

        Returns:
            Tuple of (is_valid, distance) where distance is used for ranking
        """
        # Check vertical distance and direction
        vertical_distance, is_valid_direction = self._get_vertical_distance(
            item_bbox, caption_bbox, search_direction
        )
        if not is_valid_direction or vertical_distance > self.max_vertical_distance:
            return False, float("inf")

        # Check horizontal overlap
        overlap = self._get_horizontal_overlap(item_bbox, caption_bbox)
        if overlap < self.min_horizontal_overlap:
            return False, float("inf")

        return True, vertical_distance

    def match_items_to_captions(
        self,
        items: List[Dict[str, Any]],
        captions: List[Dict[str, Any]],
        item_type: str = "figure",
    ) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
        """
        Match items (figures/tables) to their captions.

        Uses greedy matching: each caption can only be matched to one item,
        and we prefer closer matches.

        Args:
            items: List of detection dicts with "bbox" key
            captions: List of detection dicts with "bbox" key
            item_type: Type of items being matched ("figure" or "table")

        Returns:
            List of (item, caption) tuples, where caption may be None
        """
        if not items:
            return []

        if not captions:
            return [(item, None) for item in items]

        # Select search direction based on item type
        if item_type == "table":
            search_direction = self.table_search_direction
        else:
            search_direction = self.figure_search_direction

        # Calculate all valid matches with distances
        matches = []
        for item in items:
            for caption in captions:
                is_valid, distance = self._is_valid_match(
                    item["bbox"], caption["bbox"], search_direction
                )
                if is_valid:
                    matches.append((item, caption, distance))

        # Sort by distance (prefer closer matches)
        matches.sort(key=lambda x: x[2])

        # Greedy matching
        used_items = set()
        used_captions = set()
        result_matches = {}

        for item, caption, _ in matches:
            item_id = id(item)
            caption_id = id(caption)

            if item_id not in used_items and caption_id not in used_captions:
                result_matches[item_id] = caption
                used_items.add(item_id)
                used_captions.add(caption_id)

        # Build result list
        result = []
        for item in items:
            item_id = id(item)
            caption = result_matches.get(item_id)
            result.append((item, caption))

        return result


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
        return [
            d for d in detections
            if d.get("class_name") in target_classes
        ]

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
        pdf_name = pdf_path.stem

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
            figures = self._filter_detections(
                detections, CaptionMatcher.FIGURE_CLASSES
            )
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
                    caption_text = self._extract_text_from_rect(
                        page, caption["bbox"]
                    )
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
            tables = self._filter_detections(
                detections, CaptionMatcher.TABLE_CLASSES
            )
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
                    caption_text = self._extract_text_from_rect(
                        page, caption["bbox"]
                    )
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
