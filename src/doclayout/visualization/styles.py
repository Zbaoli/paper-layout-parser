"""
Visualization Styles

Color definitions and label strategies for bounding box rendering.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

# Default colors for each class (BGR format for OpenCV)
DEFAULT_COLORS = {
    # DocLayNet classes
    "Caption": (0, 165, 255),  # Orange
    "Footnote": (128, 128, 128),  # Gray
    "Formula": (255, 0, 255),  # Magenta
    "List-item": (255, 255, 0),  # Cyan
    "Page-footer": (192, 192, 192),  # Silver
    "Page-header": (192, 192, 192),  # Silver
    "Picture": (0, 255, 0),  # Green
    "Section-header": (0, 0, 255),  # Red
    "Table": (255, 0, 0),  # Blue
    "Text": (0, 255, 255),  # Yellow
    "Title": (128, 0, 128),  # Purple
    # DocLayout-YOLO DocStructBench classes
    "Plain-Text": (0, 180, 0),  # Dark Green
    "Abandon": (64, 64, 64),  # Dark Gray
    "Figure": (0, 255, 0),  # Green
    "Figure-Caption": (0, 200, 255),  # Light Orange
    "Table-Caption": (255, 100, 0),  # Light Blue
    "Table-Footnote": (128, 128, 128),  # Gray
    "Isolate-Formula": (255, 0, 255),  # Magenta
    "Formula-Caption": (200, 0, 200),  # Dark Magenta
}

# Colors for annotation types (used by NumberedLabelStrategy)
ANNOTATION_COLORS = {
    "figure": (0, 255, 0),  # Green
    "table": (255, 0, 0),  # Blue
    "caption": (0, 165, 255),  # Orange
}

# Prefixes for numbered labels
ANNOTATION_PREFIXES = {
    "figure": "F",
    "table": "T",
    "caption": "C",
}


class ColorPalette:
    """Manages color assignments for visualization."""

    def __init__(
        self,
        class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        default_color: Tuple[int, int, int] = (0, 255, 0),
    ):
        """
        Initialize color palette.

        Args:
            class_colors: Custom color mappings (BGR format)
            default_color: Default color for unknown classes
        """
        self.colors = class_colors or DEFAULT_COLORS.copy()
        self.default_color = default_color

    def get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for a class name."""
        return self.colors.get(class_name, self.default_color)

    def add_color(self, class_name: str, color: Tuple[int, int, int]) -> None:
        """Add or update a color mapping."""
        self.colors[class_name] = color


class LabelStrategy(ABC):
    """Abstract base class for label generation strategies."""

    @abstractmethod
    def get_label(self, bbox: Dict[str, Any], index: int = 0) -> str:
        """
        Generate a label for a bounding box.

        Args:
            bbox: Bounding box data with at least class_name and confidence
            index: Optional index for numbered labels

        Returns:
            Label string to display
        """
        pass

    @abstractmethod
    def get_color(self, bbox: Dict[str, Any]) -> Tuple[int, int, int]:
        """
        Get the color for a bounding box.

        Args:
            bbox: Bounding box data

        Returns:
            BGR color tuple
        """
        pass


class ClassNameLabelStrategy(LabelStrategy):
    """
    Label strategy that shows class name and confidence.

    Used by the original Visualizer for general detection visualization.
    """

    def __init__(
        self,
        show_confidence: bool = True,
        color_palette: Optional[ColorPalette] = None,
    ):
        """
        Initialize the strategy.

        Args:
            show_confidence: Whether to show confidence scores
            color_palette: Color palette to use
        """
        self.show_confidence = show_confidence
        self.palette = color_palette or ColorPalette()

    def get_label(self, bbox: Dict[str, Any], index: int = 0) -> str:
        """Generate label with class name and optional confidence."""
        class_name = bbox.get("class_name", "unknown")
        if self.show_confidence:
            confidence = bbox.get("confidence", 0.0)
            return f"{class_name}: {confidence:.2f}"
        return class_name

    def get_color(self, bbox: Dict[str, Any]) -> Tuple[int, int, int]:
        """Get color based on class name."""
        class_name = bbox.get("class_name", "unknown")
        return self.palette.get_color(class_name)


class NumberedLabelStrategy(LabelStrategy):
    """
    Label strategy that shows numbered identifiers (F1, T2, C3).

    Used for VLM annotation where items need numbered references.
    """

    def __init__(
        self,
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        prefixes: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the strategy.

        Args:
            colors: Colors for each item type (figure, table, caption)
            prefixes: Label prefixes for each type
        """
        self.colors = colors or ANNOTATION_COLORS.copy()
        self.prefixes = prefixes or ANNOTATION_PREFIXES.copy()

    def get_label(self, bbox: Dict[str, Any], index: int = 0) -> str:
        """Generate numbered label like F1, T2, C3."""
        item_type = bbox.get("item_type", "figure")
        prefix = self.prefixes.get(item_type, "?")
        # Index can be from the bbox data or passed in
        num = bbox.get("id", index + 1)
        return f"{prefix}{num}"

    def get_color(self, bbox: Dict[str, Any]) -> Tuple[int, int, int]:
        """Get color based on item type."""
        item_type = bbox.get("item_type", "figure")
        return self.colors.get(item_type, (0, 255, 0))
