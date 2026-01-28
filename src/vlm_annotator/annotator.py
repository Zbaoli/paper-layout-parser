"""
Caption Annotator

Core logic for VLM-assisted figure-caption annotation.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

from .base import BaseVLMClient, VLMResponse
from .image_renderer import AnnotationRenderer


@dataclass
class PageAnnotation:
    """Annotation result for a single page."""

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
    """Complete annotation result for a document."""

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
    ):
        """
        Initialize the annotator.

        Args:
            vlm_client: VLM client for analysis
            dpi: DPI used for image conversion (for coordinate reference)
            max_workers: Maximum concurrent VLM requests per document
        """
        self.vlm_client = vlm_client
        self.dpi = dpi
        self.max_workers = max_workers
        self.renderer = AnnotationRenderer()

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

        # Process pages concurrently
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
                        print(f"  Error processing page {page_number}: {e}")
        else:
            # Sequential processing
            for page_number, detections, page_image, caption_texts in page_tasks:
                page_annotation = self._annotate_page_concurrent(
                    page_number, detections, page_image, annotated_dir, caption_texts
                )
                result.pages.append(page_annotation)

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
        figures_with_id = [{"id": i + 1, "bbox": f["bbox"]} for i, f in enumerate(figures)]
        tables_with_id = [{"id": i + 1, "bbox": t["bbox"]} for i, t in enumerate(tables)]
        captions_with_id = []

        for i, cap in enumerate(captions):
            det_idx = caption_indices[i]
            cap_data = {
                "id": i + 1,
                "bbox": cap["bbox"],
                "text": caption_texts.get(det_idx, ""),
            }
            captions_with_id.append(cap_data)

        # Create annotation result for empty pages
        if not figures and not tables:
            unmatched_caps = [f"cap_{page_number:02d}_{c['id']:02d}" for c in captions_with_id]
            return PageAnnotation(
                page_number=page_number,
                unmatched_captions=unmatched_caps,
            )

        # Render annotated image
        annotated_image_path = annotated_dir / f"page_{page_number:04d}_annotated.png"
        self.renderer.render_annotated_image(
            page_image,
            figures_with_id,
            tables_with_id,
            captions_with_id,
            str(annotated_image_path),
        )

        # Call VLM for analysis
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
        figures_with_id = [{"id": i + 1, "bbox": f["bbox"]} for i, f in enumerate(figures)]
        tables_with_id = [{"id": i + 1, "bbox": t["bbox"]} for i, t in enumerate(tables)]
        captions_with_id = []

        for i, cap in enumerate(captions):
            cap_data = {"id": i + 1, "bbox": cap["bbox"], "text": ""}

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

        # Render annotated image
        annotated_image_path = annotated_dir / f"page_{page_number:04d}_annotated.png"
        self.renderer.render_annotated_image(
            page_image,
            figures_with_id,
            tables_with_id,
            captions_with_id,
            str(annotated_image_path),
        )

        # Call VLM for analysis
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
    backend: str = "ollama",
    model: Optional[str] = None,
    **kwargs,
) -> BaseVLMClient:
    """
    Factory function to create a VLM client.

    Args:
        backend: VLM backend ("ollama", "openai", or "anthropic")
        model: Model name (optional, uses default for backend)
        **kwargs: Additional client configuration

    Returns:
        Configured VLM client
    """
    if backend == "ollama":
        from .ollama_client import OllamaClient

        return OllamaClient(model=model, **kwargs)
    elif backend == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(model=model, **kwargs)
    elif backend == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown VLM backend: {backend}")
