"""
Bounding Box Renderer

Unified renderer for drawing bounding boxes on images.
Supports different labeling strategies via strategy pattern.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .styles import (
    ColorPalette,
    LabelStrategy,
    ClassNameLabelStrategy,
    NumberedLabelStrategy,
)


class BoundingBoxRenderer:
    """
    Unified renderer for drawing bounding boxes on images.

    Supports multiple labeling strategies:
    - ClassNameLabelStrategy: Shows class name + confidence (original Visualizer)
    - NumberedLabelStrategy: Shows F1, T2, C3 etc. (for VLM annotation)
    """

    def __init__(
        self,
        label_strategy: Optional[LabelStrategy] = None,
        line_thickness: int = 2,
        font_scale: float = 0.6,
        label_padding: int = 5,
    ):
        """
        Initialize the renderer.

        Args:
            label_strategy: Strategy for generating labels and colors
            line_thickness: Thickness of bounding box lines
            font_scale: Scale factor for text labels
            label_padding: Padding around label text
        """
        self.label_strategy = label_strategy or ClassNameLabelStrategy()
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.label_padding = label_padding
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_box(
        self,
        image: np.ndarray,
        bbox: Dict[str, Any],
        index: int = 0,
    ) -> np.ndarray:
        """
        Draw a single bounding box with label on an image.

        Args:
            image: Input image (BGR format)
            bbox: Bounding box dict with at least {"x1", "y1", "x2", "y2"}
                  and optionally {"class_name", "confidence", "item_type", "id"}
            index: Index for numbered labels

        Returns:
            Image with bounding box drawn
        """
        # Get box coordinates
        box_data = bbox.get("bbox", bbox)  # Support nested or flat structure
        x1 = int(box_data.get("x1", box_data.get("bbox", {}).get("x1", 0)))
        y1 = int(box_data.get("y1", box_data.get("bbox", {}).get("y1", 0)))
        x2 = int(box_data.get("x2", box_data.get("bbox", {}).get("x2", 0)))
        y2 = int(box_data.get("y2", box_data.get("bbox", {}).get("y2", 0)))

        # Get color and label from strategy
        color = self.label_strategy.get_color(bbox)
        label = self.label_strategy.get_label(bbox, index)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.line_thickness)

        # Calculate label position and size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.line_thickness
        )

        # Draw label background
        label_y1 = y1 - text_height - 2 * self.label_padding
        label_y2 = y1

        # Handle case where label would go above image
        if label_y1 < 0:
            label_y1 = y1
            label_y2 = y1 + text_height + 2 * self.label_padding

        label_x1 = x1
        label_x2 = x1 + text_width + 2 * self.label_padding

        # Draw filled rectangle for label background
        cv2.rectangle(
            image,
            (label_x1, label_y1),
            (label_x2, label_y2),
            color,
            -1,  # Filled
        )

        # Draw label text (white on colored background)
        text_x = label_x1 + self.label_padding
        text_y = label_y2 - self.label_padding

        cv2.putText(
            image,
            label,
            (text_x, text_y),
            self.font,
            self.font_scale,
            (255, 255, 255),  # White text
            self.line_thickness,
            cv2.LINE_AA,
        )

        return image

    def render(
        self,
        image: np.ndarray,
        boxes: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw all bounding boxes on an image.

        Args:
            image: Input image (BGR format)
            boxes: List of bounding box dictionaries

        Returns:
            Image with all boxes drawn
        """
        result = image.copy()
        for i, box in enumerate(boxes):
            result = self.draw_box(result, box, i)
        return result

    def render_image(
        self,
        image_path: str,
        boxes: List[Dict[str, Any]],
        output_path: str,
    ) -> str:
        """
        Load image, render boxes, and save result.

        Args:
            image_path: Path to input image
            boxes: List of bounding box dictionaries
            output_path: Path to save output image

        Returns:
            Path to saved image
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Render boxes
        result = self.render(image, boxes)

        # Save result
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)

        return output_path

    def render_annotated_image(
        self,
        image_path: str,
        figures: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        captions: List[Dict[str, Any]],
        output_path: str,
    ) -> str:
        """
        Render annotated image with figures, tables, and captions.

        This method provides compatibility with the original AnnotationRenderer
        by accepting separate lists for each item type.

        Args:
            image_path: Path to original page image
            figures: List of figure detections with bbox
            tables: List of table detections with bbox
            captions: List of caption detections with bbox
            output_path: Path to save the annotated image

        Returns:
            Path to saved image
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Prepare boxes with item_type annotation
        all_boxes = []

        for i, fig in enumerate(figures):
            box = fig.copy()
            box["item_type"] = "figure"
            box["id"] = fig.get("id", i + 1)
            all_boxes.append(box)

        for i, tbl in enumerate(tables):
            box = tbl.copy()
            box["item_type"] = "table"
            box["id"] = tbl.get("id", i + 1)
            all_boxes.append(box)

        for i, cap in enumerate(captions):
            box = cap.copy()
            box["item_type"] = "caption"
            box["id"] = cap.get("id", i + 1)
            all_boxes.append(box)

        # Render all boxes
        result = self.render(image, all_boxes)

        # Save result
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)

        return str(output_path)


def create_visualizer(
    style: str = "class_name",
    show_confidence: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.6,
    colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> BoundingBoxRenderer:
    """
    Factory function to create a BoundingBoxRenderer with appropriate strategy.

    Args:
        style: Label style - "class_name" or "numbered"
        show_confidence: Whether to show confidence (for class_name style)
        line_thickness: Thickness of bounding box lines
        font_scale: Scale factor for text labels
        colors: Custom color mappings

    Returns:
        Configured BoundingBoxRenderer
    """
    if style == "numbered":
        strategy = NumberedLabelStrategy(colors=colors)
    else:
        palette = ColorPalette(colors) if colors else None
        strategy = ClassNameLabelStrategy(
            show_confidence=show_confidence,
            color_palette=palette,
        )

    return BoundingBoxRenderer(
        label_strategy=strategy,
        line_thickness=line_thickness,
        font_scale=font_scale,
    )
