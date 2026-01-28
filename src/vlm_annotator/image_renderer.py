"""
Annotation Image Renderer

Renders numbered bounding boxes on page images for VLM analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class AnnotationRenderer:
    """Renders numbered bounding boxes on page images."""

    # Colors in BGR format for OpenCV
    COLORS = {
        "figure": (0, 255, 0),  # Green
        "table": (255, 0, 0),  # Blue
        "caption": (0, 165, 255),  # Orange
    }

    # Prefixes for each type
    PREFIXES = {
        "figure": "F",
        "table": "T",
        "caption": "C",
    }

    def __init__(
        self,
        line_thickness: int = 3,
        font_scale: float = 0.8,
        label_padding: int = 5,
    ):
        """
        Initialize the renderer.

        Args:
            line_thickness: Thickness of bounding box lines
            font_scale: Scale factor for text labels
            label_padding: Padding around label text
        """
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.label_padding = label_padding
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render_annotated_image(
        self,
        image_path: str,
        figures: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        captions: List[Dict[str, Any]],
        output_path: str,
    ) -> str:
        """
        Render an annotated image with numbered bounding boxes.

        Args:
            image_path: Path to the original page image
            figures: List of figure detections with bbox
            tables: List of table detections with bbox
            captions: List of caption detections with bbox
            output_path: Path to save the annotated image

        Returns:
            Path to the saved annotated image
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Draw figures
        for i, fig in enumerate(figures, 1):
            self._draw_labeled_box(
                image,
                fig["bbox"],
                f"F{i}",
                self.COLORS["figure"],
            )

        # Draw tables
        for i, tbl in enumerate(tables, 1):
            self._draw_labeled_box(
                image,
                tbl["bbox"],
                f"T{i}",
                self.COLORS["table"],
            )

        # Draw captions
        for i, cap in enumerate(captions, 1):
            self._draw_labeled_box(
                image,
                cap["bbox"],
                f"C{i}",
                self.COLORS["caption"],
            )

        # Save image
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)

        return str(output_path)

    def _draw_labeled_box(
        self,
        image: np.ndarray,
        bbox: Dict[str, float],
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        """
        Draw a labeled bounding box on the image.

        Args:
            image: Image to draw on (modified in place)
            bbox: Bounding box with x1, y1, x2, y2
            label: Text label for the box
            color: BGR color tuple
        """
        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.line_thickness)

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label, self.font, self.font_scale, 2)

        # Draw label background
        label_x1 = x1
        label_y1 = y1 - text_height - 2 * self.label_padding
        label_x2 = x1 + text_width + 2 * self.label_padding
        label_y2 = y1

        # Ensure label is within image bounds
        if label_y1 < 0:
            # Place label inside the box if it would go above image
            label_y1 = y1
            label_y2 = y1 + text_height + 2 * self.label_padding

        # Draw filled rectangle for label background
        cv2.rectangle(
            image,
            (label_x1, label_y1),
            (label_x2, label_y2),
            color,
            -1,  # Filled
        )

        # Draw label text
        text_x = label_x1 + self.label_padding
        text_y = label_y2 - self.label_padding

        # White text on colored background
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            self.font,
            self.font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def create_legend(self, output_path: str, width: int = 400, height: int = 150) -> str:
        """
        Create a legend image explaining the color coding.

        Args:
            output_path: Path to save the legend image
            width: Legend image width
            height: Legend image height

        Returns:
            Path to the saved legend image
        """
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Title
        cv2.putText(
            image,
            "Annotation Legend",
            (20, 30),
            self.font,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Legend items
        items = [
            ("Figure (F#)", self.COLORS["figure"]),
            ("Table (T#)", self.COLORS["table"]),
            ("Caption (C#)", self.COLORS["caption"]),
        ]

        y_offset = 60
        for label, color in items:
            # Draw color box
            cv2.rectangle(image, (20, y_offset), (50, y_offset + 25), color, -1)
            cv2.rectangle(image, (20, y_offset), (50, y_offset + 25), (0, 0, 0), 1)

            # Draw label
            cv2.putText(
                image,
                label,
                (60, y_offset + 18),
                self.font,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            y_offset += 35

        # Save image
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)

        return str(output_path)
