"""
Layout Detector Module

Provides document layout detection using DocLayout-YOLO and YOLOv8 models.
Supports MPS acceleration for Mac M-series chips.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np


# Class names for DocLayout-YOLO DocStructBench model (10 classes)
DOCLAYOUT_CLASS_NAMES = {
    0: "Title",
    1: "Text",
    2: "Abandon",
    3: "Figure",
    4: "Figure-caption",
    5: "Table",
    6: "Table-caption",
    7: "Table-footnote",
    8: "Formula",
    9: "Formula-caption",
}

# Class names for DocLayNet (11 classes)
DOCLAYNET_CLASS_NAMES = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
}


class Detection:
    """Represents a single detection result."""

    def __init__(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox: Dict[str, float],
    ):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # {"x1": float, "y1": float, "x2": float, "y2": float}

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary format."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x1": round(self.bbox["x1"], 2),
                "y1": round(self.bbox["y1"], 2),
                "x2": round(self.bbox["x2"], 2),
                "y2": round(self.bbox["y2"], 2),
            },
        }


class BaseLayoutDetector(ABC):
    """Abstract base class for layout detectors."""

    def __init__(
        self,
        model_path: str,
        device: str = "mps",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = {}  # Will be set by subclasses

    @abstractmethod
    def load_model(self):
        """Load the detection model."""
        pass

    @abstractmethod
    def detect(self, image_path: str) -> List[Detection]:
        """
        Detect layout elements in an image.

        Args:
            image_path: Path to the input image

        Returns:
            List of Detection objects
        """
        pass

    def detect_batch(self, image_paths: List[str]) -> Dict[str, List[Detection]]:
        """
        Detect layout elements in multiple images.

        Args:
            image_paths: List of paths to input images

        Returns:
            Dictionary mapping image paths to their detections
        """
        results = {}
        for image_path in image_paths:
            results[image_path] = self.detect(image_path)
        return results


class DocLayoutDetector(BaseLayoutDetector):
    """Layout detector using DocLayout-YOLO model."""

    def __init__(
        self,
        model_path: str = "juliozhao/DocLayout-YOLO-DocStructBench",
        device: str = "mps",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        super().__init__(model_path, device, confidence_threshold, iou_threshold)
        self.load_model()

    def load_model(self):
        """Load the DocLayout-YOLO model from HuggingFace."""
        try:
            from doclayout_yolo import YOLOv10
            from huggingface_hub import hf_hub_download

            # Download model from HuggingFace if it's a repo ID
            if "/" in self.model_path and not os.path.exists(self.model_path):
                model_file = hf_hub_download(
                    repo_id=self.model_path,
                    filename="doclayout_yolo_docstructbench_imgsz1024.pt",
                )
            else:
                model_file = self.model_path

            self.model = YOLOv10(model_file)

            # Use the model's own class names (capitalize first letter)
            self.class_names = {
                idx: name.replace("_", "-").title().replace(" ", "-")
                for idx, name in self.model.names.items()
            }
            print(f"DocLayout-YOLO model loaded from: {self.model_path}")
            print(f"  Classes: {list(self.class_names.values())}")

        except ImportError as e:
            raise ImportError(
                "doclayout-yolo package not installed. "
                "Please install it with: pip install doclayout-yolo"
            ) from e

    def detect(self, image_path: str) -> List[Detection]:
        """Detect layout elements using DocLayout-YOLO."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Run inference
        results = self.model.predict(
            image_path,
            imgsz=1024,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get box coordinates
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    detection = Detection(
                        class_id=cls_id,
                        class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                        confidence=conf,
                        bbox={
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3]),
                        },
                    )
                    detections.append(detection)

        return detections


class YOLOv8LayoutDetector(BaseLayoutDetector):
    """Layout detector using ultralytics YOLOv8 model."""

    # HuggingFace repos for YOLOv8 DocLayNet models
    HF_REPOS = {
        "hantian": {
            "repo_id": "hantian/yolo-doclaynet",
            "files": {
                "n": "yolov8n-doclaynet.pt",
                "s": "yolov8s-doclaynet.pt",
                "m": "yolov8m-doclaynet.pt",
            },
            "default": "n",
        },
        "DILHTWD": {
            "repo_id": "DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet",
            "files": {
                "default": "yolov8x-doclaynet.pt",
            },
            "default": "default",
        },
    }

    def __init__(
        self,
        model_path: str = "hantian/yolo-doclaynet",
        device: str = "mps",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        model_variant: str = "n",  # n, s, m for hantian repo
    ):
        self.model_variant = model_variant
        super().__init__(model_path, device, confidence_threshold, iou_threshold)
        self.load_model()

    def _download_from_huggingface(self, repo_id: str, filename: str) -> str:
        """Download model from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download

            print(f"Downloading {filename} from {repo_id}...")
            model_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )
            print(f"Model downloaded to: {model_file}")
            return model_file
        except Exception as e:
            raise RuntimeError(f"Failed to download model from HuggingFace: {e}") from e

    def load_model(self):
        """Load the YOLOv8 model, downloading from HuggingFace if necessary."""
        try:
            from ultralytics import YOLO

            model_file = None

            # Check if model_path is a HuggingFace repo ID (contains /)
            if "/" in self.model_path and not os.path.exists(self.model_path):
                # Parse repo ID to determine which HF repo to use
                repo_parts = self.model_path.split("/")
                repo_owner = repo_parts[0]

                if repo_owner in self.HF_REPOS:
                    repo_info = self.HF_REPOS[repo_owner]
                    repo_id = repo_info["repo_id"]
                    variant = self.model_variant if self.model_variant in repo_info["files"] else repo_info["default"]
                    filename = repo_info["files"][variant]
                    model_file = self._download_from_huggingface(repo_id, filename)
                else:
                    # Try to download directly using the provided path as repo_id
                    # Assume default model filename pattern
                    model_file = self._download_from_huggingface(
                        self.model_path,
                        f"yolov8{self.model_variant}-doclaynet.pt"
                    )
            elif os.path.exists(self.model_path):
                # Use local model file
                model_file = self.model_path
            else:
                # Model file not found locally, try to download from default HF repo
                print(f"Model not found at {self.model_path}, downloading from HuggingFace...")
                repo_info = self.HF_REPOS["hantian"]
                variant = self.model_variant if self.model_variant in repo_info["files"] else repo_info["default"]
                filename = repo_info["files"][variant]
                model_file = self._download_from_huggingface(repo_info["repo_id"], filename)

            self.model = YOLO(model_file)

            # Use the model's own class names if available, otherwise use DocLayNet defaults
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = {
                    idx: name.replace("_", "-").title()
                    for idx, name in self.model.names.items()
                }
            else:
                self.class_names = DOCLAYNET_CLASS_NAMES.copy()

            print(f"YOLOv8 model loaded: {model_file}")
            print(f"  Classes: {list(self.class_names.values())}")

        except ImportError as e:
            raise ImportError(
                "ultralytics package not installed. "
                "Please install it with: pip install ultralytics"
            ) from e

    def detect(self, image_path: str) -> List[Detection]:
        """Detect layout elements using YOLOv8."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Run inference
        results = self.model.predict(
            image_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get box coordinates
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    detection = Detection(
                        class_id=cls_id,
                        class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                        confidence=conf,
                        bbox={
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3]),
                        },
                    )
                    detections.append(detection)

        return detections


def create_detector(
    model_type: str = "doclayout",
    model_path: Optional[str] = None,
    device: str = "mps",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    model_variant: str = "n",
) -> BaseLayoutDetector:
    """
    Factory function to create a layout detector.

    Args:
        model_type: Type of model ("doclayout" or "yolov8")
        model_path: Path to the model file (optional, can be HuggingFace repo ID)
        device: Device to run inference on ("mps", "cpu", "cuda")
        confidence_threshold: Minimum confidence for detections
        iou_threshold: IOU threshold for NMS
        model_variant: For YOLOv8, the model variant ("n", "s", "m" for hantian repo)

    Returns:
        A layout detector instance
    """
    if model_type == "doclayout":
        default_path = "juliozhao/DocLayout-YOLO-DocStructBench"
        return DocLayoutDetector(
            model_path=model_path or default_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
    elif model_type == "yolov8":
        default_path = "hantian/yolo-doclaynet"
        return YOLOv8LayoutDetector(
            model_path=model_path or default_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            model_variant=model_variant,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'doclayout' or 'yolov8'")
