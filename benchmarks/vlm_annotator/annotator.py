"""
Caption Annotator

Core logic for VLM-assisted figure-caption annotation.

Supports two modes:
1. Direct mode (recommended): VLM analyzes raw PDF pages directly
2. Detection mode (legacy): VLM analyzes pages with YOLO detection overlays
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Import from doclayout visualization module
from doclayout.visualization import BoundingBoxRenderer, NumberedLabelStrategy

from .base import BaseVLMClient, VLMDirectResponse, VLMResponse


@dataclass
class DirectPageAnnotation:
    """Annotation result for a single page in direct mode."""

    page_number: int
    elements: List[Dict[str, Any]] = field(default_factory=list)
    matches: List[Dict[str, Any]] = field(default_factory=list)
    unmatched_figures: List[int] = field(default_factory=list)
    unmatched_tables: List[int] = field(default_factory=list)
    unmatched_captions: List[int] = field(default_factory=list)
    vlm_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "page_number": self.page_number,
            "elements": self.elements,
            "matches": self.matches,
            "unmatched_figures": self.unmatched_figures,
            "unmatched_tables": self.unmatched_tables,
            "unmatched_captions": self.unmatched_captions,
        }
        if self.vlm_response:
            result["vlm_response"] = self.vlm_response
        return result


@dataclass
class DirectAnnotationResult:
    """Complete annotation result for a document in direct mode."""

    pdf_name: str
    total_pages: int
    pages: List[DirectPageAnnotation] = field(default_factory=list)
    annotator: str = ""
    mode: str = "direct"
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        # Calculate statistics
        total_matches = sum(len(p.matches) for p in self.pages)
        total_figures = sum(
            len([e for e in p.elements if e.get("type") == "figure"]) for p in self.pages
        )
        total_tables = sum(
            len([e for e in p.elements if e.get("type") == "table"]) for p in self.pages
        )
        total_captions = sum(
            len([e for e in p.elements if e.get("type") == "caption"]) for p in self.pages
        )

        return {
            "pdf_name": self.pdf_name,
            "total_pages": self.total_pages,
            "annotator": self.annotator,
            "mode": self.mode,
            "created_at": self.created_at,
            "statistics": {
                "total_matches": total_matches,
                "total_figures": total_figures,
                "total_tables": total_tables,
                "total_captions": total_captions,
                "pages_with_content": sum(1 for p in self.pages if p.elements),
            },
            "pages": [p.to_dict() for p in self.pages],
        }


@dataclass
class PageAnnotation:
    """Annotation result for a single page (legacy detection mode)."""

    page_number: int
    matches: List[Dict[str, Any]] = field(default_factory=list)
    unmatched_figures: List[str] = field(default_factory=list)
    unmatched_tables: List[str] = field(default_factory=list)
    unmatched_captions: List[str] = field(default_factory=list)
    vlm_response: Optional[Dict[str, Any]] = None
    annotated_image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "matches": self.matches,
            "unmatched_figures": self.unmatched_figures,
            "unmatched_tables": self.unmatched_tables,
            "unmatched_captions": self.unmatched_captions,
            "annotated_image_path": self.annotated_image_path,
        }


@dataclass
class AnnotationResult:
    """Complete annotation result for a document (legacy detection mode)."""

    pdf_name: str
    total_pages: int
    pages: List[PageAnnotation] = field(default_factory=list)
    annotator: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        # Calculate statistics
        total_matches = sum(len(p.matches) for p in self.pages)
        total_figures = sum(len(p.matches) + len(p.unmatched_figures) for p in self.pages)
        total_tables = sum(
            len([m for m in p.matches if m.get("figure_type") == "table"]) + len(p.unmatched_tables)
            for p in self.pages
        )

        return {
            "pdf_name": self.pdf_name,
            "total_pages": self.total_pages,
            "annotator": self.annotator,
            "created_at": self.created_at,
            "statistics": {
                "total_matches": total_matches,
                "total_figures": total_figures,
                "total_tables": total_tables,
                "pages_with_matches": sum(1 for p in self.pages if p.matches),
            },
            "pages": [p.to_dict() for p in self.pages],
        }


class CaptionAnnotator:
    """Annotates figure-caption correspondences using VLM."""

    # Class name mappings
    FIGURE_CLASSES = {"Figure"}
    TABLE_CLASSES = {"Table"}
    CAPTION_CLASSES = {"Figure-Caption", "Table-Caption", "Figure-caption", "Table-caption"}

    def __init__(
        self,
        vlm_client: BaseVLMClient,
        dpi: int = 200,
        max_workers: int = 5,
        rate_limit: int = 10,
    ):
        """
        Initialize the annotator.

        Args:
            vlm_client: VLM client for analysis
            dpi: DPI used for image conversion (for coordinate reference)
            max_workers: Maximum concurrent VLM requests per document
            rate_limit: Maximum concurrent API calls (for rate limiting)
        """
        self.vlm_client = vlm_client
        self.dpi = dpi
        self.max_workers = max_workers
        self._semaphore = threading.Semaphore(rate_limit)
        # Use unified renderer with numbered label strategy (for detection mode)
        self.renderer = BoundingBoxRenderer(
            label_strategy=NumberedLabelStrategy(),
            line_thickness=3,
            font_scale=0.8,
        )

    # =========================================================================
    # Direct annotation mode (recommended)
    # =========================================================================

    def annotate_from_pdf(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
        visualize: bool = True,
    ) -> DirectAnnotationResult:
        """
        Directly annotate figure-caption correspondences from a PDF.

        This method converts the PDF to images and uses VLM to analyze each page
        directly, without relying on any pre-detection results.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save annotation results (default: data/benchmark/<pdf_name>/)
            visualize: Whether to generate visualization images (default: True)

        Returns:
            DirectAnnotationResult with all annotations
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Determine output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path("data/benchmark") / pdf_path.stem
        output_path.mkdir(parents=True, exist_ok=True)

        # Create pages directory for converted images
        pages_dir = output_path / "pages"
        pages_dir.mkdir(exist_ok=True)

        # Open PDF and convert to images
        pdf_doc = fitz.open(str(pdf_path))
        total_pages = len(pdf_doc)

        result = DirectAnnotationResult(
            pdf_name=pdf_path.name,
            total_pages=total_pages,
            annotator=self.vlm_client.client_name,
            mode="direct",
            created_at=datetime.now().isoformat(),
        )

        # Prepare page tasks: convert pages to images first
        page_tasks: List[Tuple[int, str]] = []
        for page_num in range(total_pages):
            page = pdf_doc[page_num]

            # Convert page to image
            mat = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
            pix = page.get_pixmap(matrix=mat)
            image_path = pages_dir / f"page_{page_num + 1:04d}.png"
            pix.save(str(image_path))

            page_tasks.append((page_num + 1, str(image_path)))

        pdf_doc.close()

        # Process pages concurrently with progress bar
        with tqdm(total=len(page_tasks), desc="Annotating pages", unit="page") as pbar:
            if self.max_workers > 1 and len(page_tasks) > 1:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(self._annotate_page_direct, page_num, image_path): page_num
                        for page_num, image_path in page_tasks
                    }

                    for future in as_completed(futures):
                        page_num = futures[future]
                        try:
                            page_annotation = future.result()
                            result.pages.append(page_annotation)
                        except Exception as e:
                            result.pages.append(
                                DirectPageAnnotation(
                                    page_number=page_num,
                                    vlm_response={"error": str(e)},
                                )
                            )
                        pbar.update(1)
            else:
                # Sequential processing
                for page_num, image_path in page_tasks:
                    try:
                        page_annotation = self._annotate_page_direct(page_num, image_path)
                        result.pages.append(page_annotation)
                    except Exception as e:
                        result.pages.append(
                            DirectPageAnnotation(
                                page_number=page_num,
                                vlm_response={"error": str(e)},
                            )
                        )
                    pbar.update(1)

        # Sort pages by page number
        result.pages.sort(key=lambda p: p.page_number)

        # Generate visualization if requested
        if visualize:
            self._generate_visualizations(output_path, pages_dir, result)

        # Save annotation result
        self._save_direct_result(output_path, result)

        return result

    def _annotate_page_direct(
        self,
        page_number: int,
        image_path: str,
    ) -> DirectPageAnnotation:
        """
        Annotate a single page using direct VLM analysis.

        Args:
            page_number: Page number (1-indexed)
            image_path: Path to the page image

        Returns:
            DirectPageAnnotation with VLM analysis results
        """
        # Call VLM for direct analysis with rate limiting
        with self._semaphore:
            vlm_response = self.vlm_client.analyze_page_direct(image_path)

        if not vlm_response.success:
            return DirectPageAnnotation(
                page_number=page_number,
                vlm_response=vlm_response.to_dict(),
            )

        # Convert VLM response to annotation format
        elements = [e.to_dict() for e in vlm_response.elements]

        matches = []
        for m in vlm_response.matches:
            matches.append(
                {
                    "figure_id": m.figure_id,
                    "figure_type": m.figure_type,
                    "caption_id": m.caption_id,
                    "confidence": m.confidence,
                    "reasoning": m.reasoning,
                }
            )

        return DirectPageAnnotation(
            page_number=page_number,
            elements=elements,
            matches=matches,
            unmatched_figures=vlm_response.unmatched_figures,
            unmatched_tables=vlm_response.unmatched_tables,
            unmatched_captions=vlm_response.unmatched_captions,
        )

    def _save_direct_result(
        self, output_dir: Path, result: DirectAnnotationResult
    ) -> str:
        """Save direct annotation result to JSON file."""
        output_file = output_dir / "caption_annotations.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        return str(output_file)

    def _generate_visualizations(
        self,
        output_dir: Path,
        pages_dir: Path,
        result: DirectAnnotationResult,
    ) -> None:
        """
        Generate visualization images for VLM annotations.

        Args:
            output_dir: Output directory for annotated images
            pages_dir: Directory containing original page images
            result: Annotation result with elements containing bbox
        """
        import cv2

        annotated_dir = output_dir / "annotated"
        annotated_dir.mkdir(exist_ok=True)

        # Create renderer with numbered label strategy
        vis_renderer = BoundingBoxRenderer(
            label_strategy=NumberedLabelStrategy(),
            line_thickness=3,
            font_scale=0.8,
        )

        pages_with_bbox = 0
        pages_without_bbox = 0

        for page_ann in result.pages:
            # Find the corresponding page image
            image_path = pages_dir / f"page_{page_ann.page_number:04d}.png"
            if not image_path.exists():
                continue

            # Check if any element has bbox
            elements_with_bbox = [e for e in page_ann.elements if e.get("bbox")]
            if not elements_with_bbox:
                pages_without_bbox += 1
                continue

            pages_with_bbox += 1

            # Load image to get dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            image_height, image_width = image.shape[:2]

            # Group elements by type and convert normalized coords to pixels
            figures = []
            tables = []
            captions = []

            for element in page_ann.elements:
                bbox_normalized = element.get("bbox")
                if not bbox_normalized:
                    continue

                # Convert normalized (0-1000) to pixel coordinates
                bbox_pixel = self._normalize_to_pixel(
                    bbox_normalized, image_width, image_height
                )

                elem_data = {
                    "id": element.get("id"),
                    "bbox": bbox_pixel,
                    "item_type": element.get("type"),
                }

                elem_type = element.get("type")
                if elem_type == "figure":
                    figures.append(elem_data)
                elif elem_type == "table":
                    tables.append(elem_data)
                elif elem_type == "caption":
                    captions.append(elem_data)

            # Generate annotated image
            output_path = annotated_dir / f"page_{page_ann.page_number:04d}_annotated.png"
            vis_renderer.render_annotated_image(
                str(image_path),
                figures,
                tables,
                captions,
                str(output_path),
            )

        # Print summary
        if pages_without_bbox > 0:
            print(
                f"\nVisualization: {pages_with_bbox} pages with bbox, "
                f"{pages_without_bbox} pages without bbox (model may not support grounding)"
            )

    def _normalize_to_pixel(
        self,
        bbox_normalized: Dict[str, int],
        image_width: int,
        image_height: int,
    ) -> Dict[str, int]:
        """
        Convert normalized coordinates (0-1000) to pixel coordinates.

        Args:
            bbox_normalized: Bounding box with normalized coords {"x1", "y1", "x2", "y2"}
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Bounding box with pixel coordinates
        """
        return {
            "x1": int(bbox_normalized["x1"] * image_width / 1000),
            "y1": int(bbox_normalized["y1"] * image_height / 1000),
            "x2": int(bbox_normalized["x2"] * image_width / 1000),
            "y2": int(bbox_normalized["y2"] * image_height / 1000),
        }

    # =========================================================================
    # Detection-based annotation mode (legacy)
    # =========================================================================

    def annotate_from_detection(
        self,
        detection_result_path: str,
        pages_dir: str,
        output_dir: str,
        pdf_path: Optional[str] = None,
    ) -> AnnotationResult:
        """
        Annotate figure-caption correspondences from detection results.

        Args:
            detection_result_path: Path to detection result.json
            pages_dir: Directory containing page images
            output_dir: Directory to save annotation results
            pdf_path: Optional path to original PDF (for text extraction)

        Returns:
            AnnotationResult with all annotations
        """
        # Load detection results
        with open(detection_result_path, "r") as f:
            detection_data = json.load(f)

        pdf_name = detection_data.get("pdf_name", "unknown")
        total_pages = detection_data.get("total_pages", 0)

        # Create output directory
        output_path = Path(output_dir)
        annotated_dir = output_path / "vlm_annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)

        # Open PDF if provided (for text extraction)
        pdf_doc = None
        if pdf_path and Path(pdf_path).exists():
            pdf_doc = fitz.open(pdf_path)

        result = AnnotationResult(
            pdf_name=pdf_name,
            total_pages=total_pages,
            annotator=self.vlm_client.client_name,
            created_at=datetime.now().isoformat(),
        )

        # Prepare page tasks (extract caption text first since pdf_doc is not thread-safe)
        pages_path = Path(pages_dir)
        page_tasks: List[Tuple[int, List[Dict], str, Dict[int, str]]] = []

        for page_data in detection_data.get("pages", []):
            page_number = page_data.get("page_number", 1)
            detections = page_data.get("detections", [])

            # Find page image
            page_image = self._find_page_image(pages_path, page_number)
            if not page_image:
                print(f"  Warning: Page image not found for page {page_number}")
                continue

            # Pre-extract caption texts (not thread-safe operation)
            caption_texts: Dict[int, str] = {}
            if pdf_doc and page_number <= len(pdf_doc):
                for i, det in enumerate(detections):
                    if det.get("class_name", "") in self.CAPTION_CLASSES:
                        caption_texts[i] = self._extract_text_from_bbox(
                            pdf_doc[page_number - 1], det["bbox"]
                        )

            page_tasks.append((page_number, detections, str(page_image), caption_texts))

        if pdf_doc:
            pdf_doc.close()

        # Process pages concurrently with progress bar
        with tqdm(total=len(page_tasks), desc="Annotating pages", unit="page") as pbar:
            if self.max_workers > 1 and len(page_tasks) > 1:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._annotate_page_concurrent,
                            page_number,
                            detections,
                            page_image,
                            annotated_dir,
                            caption_texts,
                        ): page_number
                        for page_number, detections, page_image, caption_texts in page_tasks
                    }

                    for future in as_completed(futures):
                        page_number = futures[future]
                        try:
                            page_annotation = future.result()
                            result.pages.append(page_annotation)
                        except Exception as e:
                            pass  # Error already logged
                        pbar.update(1)
            else:
                # Sequential processing
                for page_number, detections, page_image, caption_texts in page_tasks:
                    page_annotation = self._annotate_page_concurrent(
                        page_number, detections, page_image, annotated_dir, caption_texts
                    )
                    result.pages.append(page_annotation)
                    pbar.update(1)

        # Sort pages by page number
        result.pages.sort(key=lambda p: p.page_number)

        # Save annotation result
        self._save_result(output_path, result)

        return result

    def _find_page_image(self, pages_dir: Path, page_number: int) -> Optional[Path]:
        """Find the page image file for a given page number."""
        patterns = [
            f"page_{page_number:04d}.png",
            f"page_{page_number:03d}.png",
            f"page_{page_number:02d}.png",
            f"page_{page_number}.png",
        ]

        for pattern in patterns:
            page_path = pages_dir / pattern
            if page_path.exists():
                return page_path

        # Try glob
        for match in pages_dir.glob(f"*{page_number}*.png"):
            return match

        return None

    def _annotate_page_concurrent(
        self,
        page_number: int,
        detections: List[Dict[str, Any]],
        page_image: str,
        annotated_dir: Path,
        caption_texts: Dict[int, str],
    ) -> PageAnnotation:
        """
        Annotate a single page (thread-safe version with pre-extracted caption texts).

        Args:
            page_number: Page number
            detections: List of detections on this page
            page_image: Path to page image
            annotated_dir: Directory for annotated images
            caption_texts: Pre-extracted caption texts keyed by detection index
        """
        # Filter detections by type
        figures = []
        tables = []
        captions = []
        caption_indices = []  # Track original detection indices for caption text lookup

        for i, det in enumerate(detections):
            class_name = det.get("class_name", "")
            if class_name in self.FIGURE_CLASSES:
                figures.append(det)
            elif class_name in self.TABLE_CLASSES:
                tables.append(det)
            elif class_name in self.CAPTION_CLASSES:
                captions.append(det)
                caption_indices.append(i)

        # Assign IDs and use pre-extracted caption text
        figures_with_id = [
            {"id": i + 1, "bbox": f["bbox"], "item_type": "figure"} for i, f in enumerate(figures)
        ]
        tables_with_id = [
            {"id": i + 1, "bbox": t["bbox"], "item_type": "table"} for i, t in enumerate(tables)
        ]
        captions_with_id = []

        for i, cap in enumerate(captions):
            det_idx = caption_indices[i]
            cap_data = {
                "id": i + 1,
                "bbox": cap["bbox"],
                "text": caption_texts.get(det_idx, ""),
                "item_type": "caption",
            }
            captions_with_id.append(cap_data)

        # Create annotation result for empty pages
        if not figures and not tables:
            unmatched_caps = [f"cap_{page_number:02d}_{c['id']:02d}" for c in captions_with_id]
            return PageAnnotation(
                page_number=page_number,
                unmatched_captions=unmatched_caps,
            )

        # Render annotated image using unified renderer
        annotated_image_path = annotated_dir / f"page_{page_number:04d}_annotated.png"
        self.renderer.render_annotated_image(
            page_image,
            figures_with_id,
            tables_with_id,
            captions_with_id,
            str(annotated_image_path),
        )

        # Call VLM for analysis with rate limiting
        with self._semaphore:
            vlm_response = self.vlm_client.analyze_page(
                str(annotated_image_path),
                figures_with_id,
                tables_with_id,
                captions_with_id,
            )

        # Process VLM response
        return self._process_vlm_response(
            page_number=page_number,
            vlm_response=vlm_response,
            figures=figures_with_id,
            tables=tables_with_id,
            captions=captions_with_id,
            annotated_image_path=str(annotated_image_path),
        )

    def _annotate_page(
        self,
        page_number: int,
        detections: List[Dict[str, Any]],
        page_image: str,
        annotated_dir: Path,
        pdf_doc: Optional[fitz.Document] = None,
    ) -> PageAnnotation:
        """Annotate a single page (legacy method for backwards compatibility)."""
        # Filter detections by type
        figures = []
        tables = []
        captions = []

        for det in detections:
            class_name = det.get("class_name", "")
            if class_name in self.FIGURE_CLASSES:
                figures.append(det)
            elif class_name in self.TABLE_CLASSES:
                tables.append(det)
            elif class_name in self.CAPTION_CLASSES:
                captions.append(det)

        # Assign IDs and extract caption text
        figures_with_id = [
            {"id": i + 1, "bbox": f["bbox"], "item_type": "figure"} for i, f in enumerate(figures)
        ]
        tables_with_id = [
            {"id": i + 1, "bbox": t["bbox"], "item_type": "table"} for i, t in enumerate(tables)
        ]
        captions_with_id = []

        for i, cap in enumerate(captions):
            cap_data = {"id": i + 1, "bbox": cap["bbox"], "text": "", "item_type": "caption"}

            # Extract caption text if PDF is available
            if pdf_doc and page_number <= len(pdf_doc):
                cap_data["text"] = self._extract_text_from_bbox(
                    pdf_doc[page_number - 1],
                    cap["bbox"],
                )

            captions_with_id.append(cap_data)

        # Create annotation result for empty pages
        if not figures and not tables:
            unmatched_caps = [f"cap_{page_number:02d}_{c['id']:02d}" for c in captions_with_id]
            return PageAnnotation(
                page_number=page_number,
                unmatched_captions=unmatched_caps,
            )

        # Render annotated image using unified renderer
        annotated_image_path = annotated_dir / f"page_{page_number:04d}_annotated.png"
        self.renderer.render_annotated_image(
            page_image,
            figures_with_id,
            tables_with_id,
            captions_with_id,
            str(annotated_image_path),
        )

        # Call VLM for analysis with rate limiting
        with self._semaphore:
            vlm_response = self.vlm_client.analyze_page(
                str(annotated_image_path),
                figures_with_id,
                tables_with_id,
                captions_with_id,
            )

        # Process VLM response
        return self._process_vlm_response(
            page_number=page_number,
            vlm_response=vlm_response,
            figures=figures_with_id,
            tables=tables_with_id,
            captions=captions_with_id,
            annotated_image_path=str(annotated_image_path),
        )

    def _extract_text_from_bbox(
        self,
        page: fitz.Page,
        bbox: Dict[str, float],
    ) -> str:
        """Extract text from a bounding box region."""
        scale = 72.0 / self.dpi
        pdf_rect = fitz.Rect(
            bbox["x1"] * scale,
            bbox["y1"] * scale,
            bbox["x2"] * scale,
            bbox["y2"] * scale,
        )
        text = page.get_text("text", clip=pdf_rect)
        return text.strip()

    def _process_vlm_response(
        self,
        page_number: int,
        vlm_response: VLMResponse,
        figures: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        captions: List[Dict[str, Any]],
        annotated_image_path: str,
    ) -> PageAnnotation:
        """Process VLM response into PageAnnotation."""
        matches = []
        matched_figure_ids = set()
        matched_table_ids = set()
        matched_caption_ids = set()

        if vlm_response.success:
            for match in vlm_response.matches:
                fig_id = match.figure_id
                fig_type = match.figure_type
                cap_id = match.caption_id

                # Get corresponding detection data
                if fig_type == "figure" and fig_id <= len(figures):
                    fig_data = figures[fig_id - 1]
                    matched_figure_ids.add(fig_id)
                elif fig_type == "table" and fig_id <= len(tables):
                    fig_data = tables[fig_id - 1]
                    matched_table_ids.add(fig_id)
                else:
                    continue

                cap_data = None
                if cap_id and cap_id <= len(captions):
                    cap_data = captions[cap_id - 1]
                    matched_caption_ids.add(cap_id)

                # Create match record
                fig_prefix = "fig" if fig_type == "figure" else "table"
                fig_id_str = f"{fig_prefix}_{page_number:02d}_{fig_id:02d}"
                cap_id_str = f"cap_{page_number:02d}_{cap_id:02d}" if cap_id else None
                match_record = {
                    "figure_id": fig_id_str,
                    "figure_type": fig_type,
                    "figure_bbox": fig_data["bbox"],
                    "caption_id": cap_id_str,
                    "caption_bbox": cap_data["bbox"] if cap_data else None,
                    "caption_text": cap_data.get("text") if cap_data else None,
                    "confidence": match.confidence,
                    "reasoning": match.reasoning,
                }
                matches.append(match_record)

        # Find unmatched items
        unmatched_figures = [
            f"fig_{page_number:02d}_{f['id']:02d}"
            for f in figures
            if f["id"] not in matched_figure_ids
        ]
        unmatched_tables = [
            f"table_{page_number:02d}_{t['id']:02d}"
            for t in tables
            if t["id"] not in matched_table_ids
        ]
        unmatched_captions = [
            f"cap_{page_number:02d}_{c['id']:02d}"
            for c in captions
            if c["id"] not in matched_caption_ids
        ]

        # Add VLM-reported unmatched captions
        for cap_id in vlm_response.unmatched_captions:
            cap_str = f"cap_{page_number:02d}_{cap_id:02d}"
            if cap_str not in unmatched_captions:
                unmatched_captions.append(cap_str)

        return PageAnnotation(
            page_number=page_number,
            matches=matches,
            unmatched_figures=unmatched_figures,
            unmatched_tables=unmatched_tables,
            unmatched_captions=unmatched_captions,
            vlm_response=vlm_response.to_dict() if not vlm_response.success else None,
            annotated_image_path=annotated_image_path,
        )

    def _save_result(self, output_dir: Path, result: AnnotationResult) -> str:
        """Save annotation result to JSON file."""
        output_file = output_dir / "caption_annotations.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        return str(output_file)


def create_vlm_client(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs,
) -> BaseVLMClient:
    """
    Factory function to create a VLM client.

    Uses LiteLLM for unified multi-provider support. Configuration priority:
    parameters > VLM_* env vars > provider-native env vars > defaults.

    Args:
        model: Model name (or set VLM_MODEL env var, default: gpt-4o)
            - OpenAI: "gpt-4o", "gpt-4-turbo"
            - Anthropic: "claude-sonnet-4-20250514"
            - Ollama: "ollama/llava:13b"
            - Third-party: "openai/model-name" with api_base
        api_key: API key (or set VLM_API_KEY env var)
        api_base: API base URL for third-party providers (or set VLM_API_BASE env var)
        **kwargs: Additional client configuration (max_tokens, temperature, etc.)

    Returns:
        Configured VLM client

    Examples:
        # Use environment variables (recommended)
        client = create_vlm_client()  # reads VLM_MODEL, VLM_API_KEY, VLM_API_BASE

        # OpenAI
        client = create_vlm_client(model="gpt-4o")

        # Anthropic
        client = create_vlm_client(model="claude-sonnet-4-20250514")

        # Ollama (local)
        client = create_vlm_client(model="ollama/llava:13b")

        # SiliconFlow (third-party)
        client = create_vlm_client(
            model="openai/Qwen/Qwen2-VL-72B-Instruct",
            api_base="https://api.siliconflow.cn/v1",
            api_key="your-key",
        )
    """
    from .litellm_client import LiteLLMClient

    return LiteLLMClient(model=model, api_key=api_key, api_base=api_base, **kwargs)
