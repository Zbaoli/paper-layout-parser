"""
Matching Module Types

Data types and enums for caption matching.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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
