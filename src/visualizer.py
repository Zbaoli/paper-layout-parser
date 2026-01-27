"""
Visualizer Module

Draws detection results on images with bounding boxes and labels.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

from .layout_detector import Detection


# Default colors for each class (BGR format for OpenCV)
DEFAULT_COLORS = {
    # DocLayNet classes
    "Caption": (0, 165, 255),       # Orange
    "Footnote": (128, 128, 128),    # Gray
    "Formula": (255, 0, 255),       # Magenta
    "List-item": (255, 255, 0),     # Cyan
    "Page-footer": (192, 192, 192), # Silver
    "Page-header": (192, 192, 192), # Silver
    "Picture": (0, 255, 0),         # Green
    "Section-header": (0, 0, 255),  # Red
    "Table": (255, 0, 0),           # Blue
    "Text": (0, 255, 255),          # Yellow
    "Title": (128, 0, 128),         # Purple
    # DocLayout-YOLO DocStructBench classes
    "Plain-Text": (0, 180, 0),      # Dark Green
    "Abandon": (64, 64, 64),        # Dark Gray
    "Figure": (0, 255, 0),          # Green
    "Figure-Caption": (0, 200, 255), # Light Orange
    "Table-Caption": (255, 100, 0), # Light Blue
    "Table-Footnote": (128, 128, 128), # Gray
    "Isolate-Formula": (255, 0, 255),  # Magenta
    "Formula-Caption": (200, 0, 200),  # Dark Magenta
}


class Visualizer:
    """Visualizes detection results on images."""

    def __init__(
        self,
        line_thickness: int = 2,
        font_scale: float = 0.6,
        show_confidence: bool = True,
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ):
        """
        Initialize the visualizer.

        Args:
            line_thickness: Thickness of bounding box lines
            font_scale: Scale factor for text labels
            show_confidence: Whether to show confidence scores
            colors: Optional custom colors for each class (BGR format)
        """
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.show_confidence = show_confidence
        self.colors = colors or DEFAULT_COLORS
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def get_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        Get the color for a class name.

        Args:
            class_name: Name of the detection class

        Returns:
            BGR color tuple
        """
        return self.colors.get(class_name, (0, 255, 0))  # Default to green

    def draw_detection(
        self,
        image: np.ndarray,
        detection: Detection,
    ) -> np.ndarray:
        """
        Draw a single detection on an image.

        Args:
            image: Input image (BGR format)
            detection: Detection to draw

        Returns:
            Image with detection drawn
        """
        color = self.get_color(detection.class_name)
        bbox = detection.bbox

        # Draw bounding box
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.line_thickness)

        # Prepare label text
        if self.show_confidence:
            label = f"{detection.class_name}: {detection.confidence:.2f}"
        else:
            label = detection.class_name

        # Calculate label size and position
        (label_width, label_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.line_thickness
        )

        # Draw label background
        label_y1 = max(y1 - label_height - 10, 0)
        label_y2 = y1
        cv2.rectangle(
            image,
            (x1, label_y1),
            (x1 + label_width + 5, label_y2),
            color,
            -1,  # Filled
        )

        # Draw label text
        text_color = (255, 255, 255)  # White text
        cv2.putText(
            image,
            label,
            (x1 + 2, y1 - 5),
            self.font,
            self.font_scale,
            text_color,
            self.line_thickness,
        )

        return image

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:
        """
        Draw all detections on an image.

        Args:
            image: Input image (BGR format)
            detections: List of detections to draw

        Returns:
            Image with all detections drawn
        """
        result = image.copy()
        for detection in detections:
            result = self.draw_detection(result, detection)
        return result

    def visualize_image(
        self,
        image_path: str,
        detections: List[Detection],
        output_path: str,
    ) -> str:
        """
        Visualize detections on an image and save the result.

        Args:
            image_path: Path to the input image
            detections: List of detections to draw
            output_path: Path to save the visualization

        Returns:
            Path to the saved visualization
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Draw detections
        result = self.draw_detections(image, detections)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save result
        cv2.imwrite(output_path, result)

        return output_path

    def visualize_document(
        self,
        page_results: List[Dict],
        output_dir: str,
    ) -> List[str]:
        """
        Visualize all pages of a document.

        Args:
            page_results: List of page result dictionaries
            output_dir: Directory to save visualization images

        Returns:
            List of paths to saved visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []

        for page_result in page_results:
            image_path = page_result.get("image_path")
            if not image_path or not os.path.exists(image_path):
                continue

            # Reconstruct Detection objects from dictionaries
            detections = []
            for det_dict in page_result.get("detections", []):
                detection = Detection(
                    class_id=det_dict.get("class_id", 0),
                    class_name=det_dict.get("class_name", "unknown"),
                    confidence=det_dict.get("confidence", 0.0),
                    bbox=det_dict.get("bbox", {"x1": 0, "y1": 0, "x2": 0, "y2": 0}),
                )
                detections.append(detection)

            # Generate output path
            page_num = page_result.get("page_number", 0)
            output_filename = f"page_{page_num:04d}.png"
            output_path = str(output_dir / output_filename)

            # Visualize and save
            try:
                self.visualize_image(image_path, detections, output_path)
                output_paths.append(output_path)
            except Exception as e:
                print(f"Warning: Failed to visualize page {page_num}: {e}")

        return output_paths

    def create_legend(
        self,
        width: int = 400,
        height: int = 400,
    ) -> np.ndarray:
        """
        Create a legend image showing all class colors.

        Args:
            width: Width of the legend image
            height: Height of the legend image

        Returns:
            Legend image (BGR format)
        """
        legend = np.ones((height, width, 3), dtype=np.uint8) * 255

        y_offset = 30
        x_offset = 20
        box_size = 20
        line_height = 35

        for class_name, color in self.colors.items():
            if y_offset + box_size > height - 10:
                break

            # Draw color box
            cv2.rectangle(
                legend,
                (x_offset, y_offset),
                (x_offset + box_size, y_offset + box_size),
                color,
                -1,
            )

            # Draw class name
            cv2.putText(
                legend,
                class_name,
                (x_offset + box_size + 10, y_offset + box_size - 5),
                self.font,
                0.5,
                (0, 0, 0),
                1,
            )

            y_offset += line_height

        return legend

    def save_legend(self, output_path: str) -> str:
        """
        Save the legend image.

        Args:
            output_path: Path to save the legend image

        Returns:
            Path to the saved legend
        """
        legend = self.create_legend()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), legend)
        return str(output_path)
