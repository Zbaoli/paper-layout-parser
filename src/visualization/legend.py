"""
Legend Renderer

Creates legend images showing color mappings.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .styles import DEFAULT_COLORS, ANNOTATION_COLORS, ANNOTATION_PREFIXES


class LegendRenderer:
    """Creates legend images for visualization."""

    def __init__(
        self,
        font_scale: float = 0.5,
        line_height: int = 35,
        box_size: int = 20,
        margin: int = 20,
    ):
        """
        Initialize the legend renderer.

        Args:
            font_scale: Scale factor for text
            line_height: Height of each legend entry
            box_size: Size of color boxes
            margin: Margin from edges
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.line_height = line_height
        self.box_size = box_size
        self.margin = margin

    def create_class_legend(
        self,
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        width: int = 400,
        height: int = 400,
    ) -> np.ndarray:
        """
        Create a legend image showing class colors.

        Args:
            colors: Color mapping to display (defaults to DEFAULT_COLORS)
            width: Width of the legend image
            height: Height of the legend image

        Returns:
            Legend image (BGR format)
        """
        colors = colors or DEFAULT_COLORS
        legend = np.ones((height, width, 3), dtype=np.uint8) * 255

        y_offset = self.margin + 10
        x_offset = self.margin

        for class_name, color in colors.items():
            if y_offset + self.box_size > height - 10:
                break

            # Draw color box
            cv2.rectangle(
                legend,
                (x_offset, y_offset),
                (x_offset + self.box_size, y_offset + self.box_size),
                color,
                -1,
            )

            # Draw class name
            cv2.putText(
                legend,
                class_name,
                (x_offset + self.box_size + 10, y_offset + self.box_size - 5),
                self.font,
                self.font_scale,
                (0, 0, 0),
                1,
            )

            y_offset += self.line_height

        return legend

    def create_annotation_legend(
        self,
        width: int = 400,
        height: int = 150,
    ) -> np.ndarray:
        """
        Create a legend for annotation colors (F/T/C).

        Args:
            width: Legend image width
            height: Legend image height

        Returns:
            Legend image (BGR format)
        """
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Title
        cv2.putText(
            image,
            "Annotation Legend",
            (self.margin, 30),
            self.font,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Legend items
        items = [
            (f"Figure ({ANNOTATION_PREFIXES['figure']}#)", ANNOTATION_COLORS["figure"]),
            (f"Table ({ANNOTATION_PREFIXES['table']}#)", ANNOTATION_COLORS["table"]),
            (f"Caption ({ANNOTATION_PREFIXES['caption']}#)", ANNOTATION_COLORS["caption"]),
        ]

        y_offset = 60
        for label, color in items:
            # Draw color box
            cv2.rectangle(
                image, (self.margin, y_offset), (self.margin + 30, y_offset + 25), color, -1
            )
            cv2.rectangle(
                image, (self.margin, y_offset), (self.margin + 30, y_offset + 25), (0, 0, 0), 1
            )

            # Draw label
            cv2.putText(
                image,
                label,
                (self.margin + 40, y_offset + 18),
                self.font,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            y_offset += 35

        return image

    def save_legend(
        self,
        output_path: str,
        legend_type: str = "class",
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        width: int = 400,
        height: Optional[int] = None,
    ) -> str:
        """
        Create and save a legend image.

        Args:
            output_path: Path to save the legend
            legend_type: "class" for class colors, "annotation" for F/T/C
            colors: Custom colors (for class legend)
            width: Legend width
            height: Legend height (auto-calculated if None)

        Returns:
            Path to saved legend
        """
        if legend_type == "annotation":
            legend = self.create_annotation_legend(width=width, height=height or 150)
        else:
            legend = self.create_class_legend(colors=colors, width=width, height=height or 400)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), legend)

        return str(output_path)
