"""
Annotation Dataset

Manages ground truth annotation datasets for caption matching evaluation.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GroundTruthMatch:
    """A single ground truth figure-caption match."""

    figure_id: str  # e.g., "fig_01_01"
    figure_type: str  # "figure" or "table"
    figure_bbox: Dict[str, float]
    caption_id: Optional[str]  # e.g., "cap_01_01" or None
    caption_bbox: Optional[Dict[str, float]]
    caption_text: Optional[str]
    confidence: float
    page_number: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "figure_type": self.figure_type,
            "figure_bbox": self.figure_bbox,
            "caption_id": self.caption_id,
            "caption_bbox": self.caption_bbox,
            "caption_text": self.caption_text,
            "confidence": self.confidence,
            "page_number": self.page_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], page_number: int) -> "GroundTruthMatch":
        return cls(
            figure_id=data["figure_id"],
            figure_type=data.get("figure_type", "figure"),
            figure_bbox=data["figure_bbox"],
            caption_id=data.get("caption_id"),
            caption_bbox=data.get("caption_bbox"),
            caption_text=data.get("caption_text"),
            confidence=data.get("confidence", 1.0),
            page_number=page_number,
        )


@dataclass
class AnnotationDataset:
    """Dataset of ground truth annotations."""

    pdf_name: str
    annotator: str
    matches: List[GroundTruthMatch] = field(default_factory=list)
    unmatched_figures: List[str] = field(default_factory=list)
    unmatched_tables: List[str] = field(default_factory=list)
    unmatched_captions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        figures_with_caps = len(
            [m for m in self.matches if m.caption_id and m.figure_type == "figure"]
        )
        tables_with_caps = len(
            [m for m in self.matches if m.caption_id and m.figure_type == "table"]
        )
        return {
            "pdf_name": self.pdf_name,
            "annotator": self.annotator,
            "statistics": {
                "total_matches": len(self.matches),
                "figures_with_captions": figures_with_caps,
                "tables_with_captions": tables_with_caps,
                "unmatched_figures": len(self.unmatched_figures),
                "unmatched_tables": len(self.unmatched_tables),
                "unmatched_captions": len(self.unmatched_captions),
            },
            "matches": [m.to_dict() for m in self.matches],
            "unmatched_figures": self.unmatched_figures,
            "unmatched_tables": self.unmatched_tables,
            "unmatched_captions": self.unmatched_captions,
        }

    @classmethod
    def from_annotation_file(cls, file_path: str) -> "AnnotationDataset":
        """
        Load dataset from VLM annotation file.

        Args:
            file_path: Path to caption_annotations.json

        Returns:
            AnnotationDataset instance
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        dataset = cls(
            pdf_name=data.get("pdf_name", "unknown"),
            annotator=data.get("annotator", "unknown"),
        )

        # Process pages
        for page_data in data.get("pages", []):
            page_number = page_data.get("page_number", 0)

            # Add matches
            for match_data in page_data.get("matches", []):
                match = GroundTruthMatch.from_dict(match_data, page_number)
                dataset.matches.append(match)

            # Add unmatched items
            dataset.unmatched_figures.extend(page_data.get("unmatched_figures", []))
            dataset.unmatched_tables.extend(page_data.get("unmatched_tables", []))
            dataset.unmatched_captions.extend(page_data.get("unmatched_captions", []))

        return dataset

    def get_matches_by_page(self, page_number: int) -> List[GroundTruthMatch]:
        """Get all matches for a specific page."""
        return [m for m in self.matches if m.page_number == page_number]

    def get_figure_caption_map(self) -> Dict[str, Optional[str]]:
        """
        Get a mapping from figure IDs to caption IDs.

        Returns:
            Dict mapping figure_id to caption_id (or None)
        """
        return {m.figure_id: m.caption_id for m in self.matches}

    def get_high_confidence_matches(self, threshold: float = 0.8) -> List[GroundTruthMatch]:
        """Get matches with confidence above threshold."""
        return [m for m in self.matches if m.confidence >= threshold]

    def save(self, output_path: str) -> None:
        """Save dataset to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def merge_datasets(datasets: List[AnnotationDataset]) -> AnnotationDataset:
    """
    Merge multiple datasets into one.

    Args:
        datasets: List of datasets to merge

    Returns:
        Merged dataset
    """
    if not datasets:
        raise ValueError("No datasets to merge")

    merged = AnnotationDataset(
        pdf_name="merged",
        annotator=datasets[0].annotator,
    )

    for ds in datasets:
        merged.matches.extend(ds.matches)
        merged.unmatched_figures.extend(ds.unmatched_figures)
        merged.unmatched_tables.extend(ds.unmatched_tables)
        merged.unmatched_captions.extend(ds.unmatched_captions)

    return merged
