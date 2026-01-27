"""
Benchmark Evaluation Module

Evaluates figure/table extraction performance using the DocLayNet dataset.
Provides metrics for detection accuracy and caption matching.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# HuggingFace mirror endpoints
HF_MIRRORS = {
    "hf-mirror": "https://hf-mirror.com",
    "openxlab": "https://huggingface.openxlab.org.cn",
}


def setup_hf_mirror(mirror: Optional[str]) -> None:
    """Set up HuggingFace mirror endpoint if specified."""
    if mirror and mirror in HF_MIRRORS:
        os.environ["HF_ENDPOINT"] = HF_MIRRORS[mirror]
        print(f"Using HuggingFace mirror: {mirror} ({HF_MIRRORS[mirror]})")


@dataclass
class BenchmarkMetrics:
    """Stores evaluation metrics."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark evaluation result."""

    dataset: str
    split: str
    model: str
    samples_evaluated: int
    figure_detection: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)
    table_detection: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)
    caption_matching: Dict[str, float] = field(default_factory=dict)
    end_to_end: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)
    per_class_breakdown: Dict[str, Any] = field(default_factory=dict)
    error_analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "model": self.model,
            "samples_evaluated": self.samples_evaluated,
            "metrics": {
                "figure_detection": self.figure_detection.to_dict(),
                "table_detection": self.table_detection.to_dict(),
                "caption_matching": self.caption_matching,
                "end_to_end": self.end_to_end.to_dict(),
            },
            "per_class_breakdown": self.per_class_breakdown,
            "error_analysis": self.error_analysis,
            "evaluated_at": datetime.now().isoformat(),
        }


def compute_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1, box2: Bounding boxes with keys "x1", "y1", "x2", "y2"

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_precision_recall_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> BenchmarkMetrics:
    """Compute precision, recall, and F1 score."""
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return BenchmarkMetrics(precision=precision, recall=recall, f1=f1)


class DocLayNetSmallLoader:
    """Loads and processes DocLayNet-small dataset (per-image JSON format)."""

    # DocLayNet class names
    CLASS_NAMES = [
        "Caption",
        "Footnote",
        "Formula",
        "List-item",
        "Page-footer",
        "Page-header",
        "Picture",
        "Section-header",
        "Table",
        "Text",
        "Title",
    ]

    def __init__(self, dataset_path: str, split: str = "test"):
        """
        Initialize the DocLayNet-small loader.

        Args:
            dataset_path: Path to the DocLayNet-small dataset directory
            split: Dataset split ("train", "val", or "test")
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.images = []
        self.annotations_cache = {}

    def load(self) -> bool:
        """
        Load the dataset.

        Returns:
            True if loaded successfully, False otherwise
        """
        # Find the split directory
        possible_paths = [
            self.dataset_path / "small_dataset" / self.split,
            self.dataset_path / self.split,
        ]

        split_dir = None
        for path in possible_paths:
            if path.exists():
                split_dir = path
                break

        if split_dir is None:
            print(f"Split directory not found. Tried:")
            for path in possible_paths:
                print(f"  - {path}")
            return False

        self.split_dir = split_dir
        images_dir = split_dir / "images"
        annotations_dir = split_dir / "annotations"

        if not images_dir.exists() or not annotations_dir.exists():
            print(f"Images or annotations directory not found in {split_dir}")
            return False

        # List all images
        image_files = sorted(images_dir.glob("*.png"))
        print(f"Found {len(image_files)} images in {split_dir}")

        for img_file in image_files:
            img_hash = img_file.stem
            ann_file = annotations_dir / f"{img_hash}.json"

            if ann_file.exists():
                self.images.append(
                    {
                        "id": img_hash,
                        "file_name": img_file.name,
                        "path": img_file,
                        "annotation_path": ann_file,
                    }
                )

        print(f"Loaded {len(self.images)} image-annotation pairs")
        return len(self.images) > 0

    def get_images(self) -> List[Dict[str, Any]]:
        """Get list of images in the dataset."""
        return self.images

    def get_annotations_for_image(self, image_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all annotations for a specific image."""
        img_id = image_info["id"]

        if img_id in self.annotations_cache:
            return self.annotations_cache[img_id]

        ann_path = image_info.get("annotation_path")
        if not ann_path or not Path(ann_path).exists():
            return []

        with open(ann_path, "r") as f:
            data = json.load(f)

        # Convert format to COCO-like structure
        annotations = []
        form_data = data.get("form", [])

        # Group by id_box to get unique bounding boxes
        seen_boxes = set()
        for item in form_data:
            box = item.get("box", [])
            category = item.get("category", "")
            id_box = item.get("id_box", 0)

            # Skip duplicates (same id_box)
            box_key = (id_box, tuple(box[:2]) if len(box) >= 2 else ())
            if box_key in seen_boxes:
                continue
            seen_boxes.add(box_key)

            if len(box) >= 4 and category:
                # Format: [x, y, width, height]
                annotations.append(
                    {
                        "bbox": box,  # Already in [x, y, w, h] format
                        "category_name": category,
                    }
                )

        self.annotations_cache[img_id] = annotations
        return annotations

    def get_image_path(self, image_info: Dict[str, Any]) -> Path:
        """Get the full path to an image file."""
        return image_info.get("path", Path(image_info["file_name"]))

    def coco_to_bbox(self, coco_bbox: List[float]) -> Dict[str, float]:
        """Convert [x, y, width, height] to our format."""
        x, y, w, h = coco_bbox
        return {
            "x1": x,
            "y1": y,
            "x2": x + w,
            "y2": y + h,
        }

    def get_category_name(self, category_id: int) -> str:
        """Get category name from ID (not used in this format)."""
        return self.CLASS_NAMES[category_id - 1] if 1 <= category_id <= len(self.CLASS_NAMES) else ""


class DocLayNetLoader:
    """Loads and processes DocLayNet dataset (COCO format)."""

    # DocLayNet class IDs (1-indexed in COCO format)
    CLASS_IDS = {
        "Caption": 1,
        "Footnote": 2,
        "Formula": 3,
        "List-item": 4,
        "Page-footer": 5,
        "Page-header": 6,
        "Picture": 7,
        "Section-header": 8,
        "Table": 9,
        "Text": 10,
        "Title": 11,
    }

    def __init__(self, dataset_path: str, split: str = "test"):
        """
        Initialize the DocLayNet loader.

        Args:
            dataset_path: Path to the DocLayNet dataset directory
            split: Dataset split ("train", "val", or "test")
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.annotations = None
        self.image_id_to_annotations = {}
        self.categories = {}

    def load(self) -> bool:
        """
        Load the dataset annotations.

        Returns:
            True if loaded successfully, False otherwise
        """
        # Try multiple possible locations for annotations
        possible_paths = [
            self.dataset_path / "annotations" / f"{self.split}.json",
            self.dataset_path / "COCO" / f"{self.split}.json",
            self.dataset_path / f"{self.split}.json",
        ]

        annotations_file = None
        for path in possible_paths:
            if path.exists():
                annotations_file = path
                break

        if annotations_file is None:
            print(f"Annotations file not found. Tried:")
            for path in possible_paths:
                print(f"  - {path}")
            print("\nPlease download the dataset first with:")
            print(f"  uv run python -m src.benchmark download --split {self.split}")
            return False

        print(f"Loading annotations from: {annotations_file}")
        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)

        # Build category mapping
        for cat in self.annotations.get("categories", []):
            self.categories[cat["id"]] = cat["name"]

        # Index annotations by image ID
        for ann in self.annotations.get("annotations", []):
            image_id = ann["image_id"]
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

        print(f"Loaded {len(self.annotations.get('images', []))} images")
        print(f"Categories: {list(self.categories.values())}")

        return True

    def get_images(self) -> List[Dict[str, Any]]:
        """Get list of images in the dataset."""
        if self.annotations is None:
            return []
        return self.annotations.get("images", [])

    def get_annotations_for_image(self, image_id: int) -> List[Dict[str, Any]]:
        """Get all annotations for a specific image."""
        return self.image_id_to_annotations.get(image_id, [])

    def get_image_path(self, image_info: Dict[str, Any]) -> Path:
        """Get the full path to an image file."""
        file_name = image_info["file_name"]

        # Try multiple possible locations
        possible_paths = [
            self.dataset_path / "images" / file_name,
            self.dataset_path / "PNG" / self.split / file_name,
            self.dataset_path / file_name,
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # Return the first option as default (will be checked by caller)
        return possible_paths[0]

    def coco_to_bbox(self, coco_bbox: List[float]) -> Dict[str, float]:
        """
        Convert COCO format [x, y, width, height] to our format.

        Args:
            coco_bbox: [x, y, width, height]

        Returns:
            Dict with x1, y1, x2, y2
        """
        x, y, w, h = coco_bbox
        return {
            "x1": x,
            "y1": y,
            "x2": x + w,
            "y2": y + h,
        }

    def get_category_name(self, category_id: int) -> str:
        """Get category name from ID."""
        return self.categories.get(category_id, f"unknown_{category_id}")


class BenchmarkEvaluator:
    """Evaluates extraction performance against ground truth."""

    FIGURE_CLASSES = {"Picture"}
    TABLE_CLASSES = {"Table"}
    CAPTION_CLASSES = {"Caption"}

    def __init__(
        self,
        dataset_path: str,
        split: str = "test",
        iou_threshold: float = 0.5,
        dataset_format: Optional[str] = None,
    ):
        """
        Initialize the benchmark evaluator.

        Args:
            dataset_path: Path to the DocLayNet dataset
            split: Dataset split to evaluate on
            iou_threshold: IoU threshold for matching predictions to ground truth
            dataset_format: "small" for DocLayNet-small, "coco" for full, None for auto-detect
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.iou_threshold = iou_threshold
        self.dataset_format = dataset_format

        # Auto-detect format or use specified
        if dataset_format is None:
            self.dataset_format = self._detect_format()

        if self.dataset_format == "small":
            self.loader = DocLayNetSmallLoader(dataset_path, split)
        else:
            self.loader = DocLayNetLoader(dataset_path, split)

    def _detect_format(self) -> str:
        """Detect the dataset format based on directory structure."""
        small_path = self.dataset_path / "small_dataset" / self.split
        if small_path.exists() and (small_path / "annotations").exists():
            return "small"
        return "coco"

    def _match_predictions_to_gt(
        self,
        predictions: List[Dict[str, float]],
        ground_truth: List[Dict[str, float]],
    ) -> Tuple[int, int, int]:
        """
        Match predictions to ground truth using IoU.

        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        if not ground_truth:
            return 0, len(predictions), 0

        if not predictions:
            return 0, 0, len(ground_truth)

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(predictions), len(ground_truth)))
        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truth):
                iou_matrix[i, j] = compute_iou(pred, gt)

        # Greedy matching
        matched_pred = set()
        matched_gt = set()

        # Sort by IoU descending
        indices = np.argsort(iou_matrix.flatten())[::-1]

        for idx in indices:
            i = idx // len(ground_truth)
            j = idx % len(ground_truth)

            if i in matched_pred or j in matched_gt:
                continue

            if iou_matrix[i, j] >= self.iou_threshold:
                matched_pred.add(i)
                matched_gt.add(j)

        true_positives = len(matched_pred)
        false_positives = len(predictions) - len(matched_pred)
        false_negatives = len(ground_truth) - len(matched_gt)

        return true_positives, false_positives, false_negatives

    def _evaluate_detection(
        self,
        pred_detections: List[Dict[str, Any]],
        gt_annotations: List[Dict[str, Any]],
        target_classes: set,
    ) -> Tuple[int, int, int]:
        """
        Evaluate detection for a specific class type.

        Args:
            pred_detections: Predicted detections
            gt_annotations: Ground truth annotations
            target_classes: Set of class names to evaluate

        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        # Filter predictions - handle both class formats
        pred_boxes = []
        for d in pred_detections:
            class_name = d.get("class_name", "")
            # Map DocLayout-YOLO class names to DocLayNet equivalents
            if class_name in target_classes:
                pred_boxes.append(d["bbox"])
            elif class_name == "Figure" and "Picture" in target_classes:
                pred_boxes.append(d["bbox"])

        # Filter ground truth by category name
        gt_boxes = []
        for ann in gt_annotations:
            # Handle both formats: small format has category_name, COCO has category_id
            cat_name = ann.get("category_name") or self.loader.get_category_name(
                ann.get("category_id", 0)
            )
            if cat_name in target_classes:
                gt_boxes.append(self.loader.coco_to_bbox(ann["bbox"]))

        return self._match_predictions_to_gt(pred_boxes, gt_boxes)

    def evaluate_sample(
        self,
        pred_detections: List[Dict[str, Any]],
        gt_annotations: List[Dict[str, Any]],
    ) -> Dict[str, Tuple[int, int, int]]:
        """
        Evaluate predictions for a single sample.

        Args:
            pred_detections: List of predicted detections
            gt_annotations: List of ground truth annotations

        Returns:
            Dict mapping class type to (TP, FP, FN) tuple
        """
        results = {}

        # Evaluate figures (Picture in DocLayNet, Figure in DocLayout-YOLO)
        results["figure"] = self._evaluate_detection(
            pred_detections,
            gt_annotations,
            self.FIGURE_CLASSES,
        )

        # Evaluate tables
        results["table"] = self._evaluate_detection(
            pred_detections,
            gt_annotations,
            self.TABLE_CLASSES,
        )

        return results

    def evaluate(
        self,
        extractor,
        detector,
        max_samples: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Run full benchmark evaluation.

        Args:
            extractor: FigureTableExtractor instance
            detector: Layout detector instance
            max_samples: Maximum number of samples to evaluate (None for all)

        Returns:
            BenchmarkResult with all metrics
        """
        if not self.loader.load():
            raise RuntimeError("Failed to load dataset")

        images = self.loader.get_images()
        if max_samples:
            images = images[:max_samples]

        # Aggregate metrics
        total_tp = defaultdict(int)
        total_fp = defaultdict(int)
        total_fn = defaultdict(int)

        print(f"Evaluating {len(images)} samples...")

        for i, image_info in enumerate(images):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(images)} samples")

            image_path = self.loader.get_image_path(image_info)
            if not image_path.exists():
                continue

            # Get ground truth - pass image_info for small format, id for COCO format
            if self.dataset_format == "small":
                gt_annotations = self.loader.get_annotations_for_image(image_info)
            else:
                gt_annotations = self.loader.get_annotations_for_image(image_info["id"])

            # Run detection
            try:
                detections = detector.detect(str(image_path))
                pred_detections = [d.to_dict() for d in detections]
            except Exception as e:
                print(f"  Error processing {image_path.name}: {e}")
                continue

            # Evaluate
            sample_results = self.evaluate_sample(pred_detections, gt_annotations)

            for class_type, (tp, fp, fn) in sample_results.items():
                total_tp[class_type] += tp
                total_fp[class_type] += fp
                total_fn[class_type] += fn

        # Compute final metrics
        figure_metrics = compute_precision_recall_f1(
            total_tp["figure"], total_fp["figure"], total_fn["figure"]
        )
        table_metrics = compute_precision_recall_f1(
            total_tp["table"], total_fp["table"], total_fn["table"]
        )

        # Combined end-to-end metrics
        total_tp_all = total_tp["figure"] + total_tp["table"]
        total_fp_all = total_fp["figure"] + total_fp["table"]
        total_fn_all = total_fn["figure"] + total_fn["table"]
        end_to_end_metrics = compute_precision_recall_f1(
            total_tp_all, total_fp_all, total_fn_all
        )

        return BenchmarkResult(
            dataset="DocLayNet",
            split=self.split,
            model=type(detector).__name__,
            samples_evaluated=len(images),
            figure_detection=figure_metrics,
            table_detection=table_metrics,
            caption_matching={"accuracy": 0.0, "recall": 0.0},  # TODO: implement
            end_to_end=end_to_end_metrics,
            per_class_breakdown={
                "figure": {
                    "true_positives": total_tp["figure"],
                    "false_positives": total_fp["figure"],
                    "false_negatives": total_fn["figure"],
                },
                "table": {
                    "true_positives": total_tp["table"],
                    "false_positives": total_fp["table"],
                    "false_negatives": total_fn["table"],
                },
            },
        )


def download_doclaynet_small(output_dir: str) -> bool:
    """
    Download DocLayNet-small dataset from HuggingFace.

    Args:
        output_dir: Directory to save the dataset

    Returns:
        True if successful
    """
    import shutil
    import zipfile

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub package required.")
        print("Install with: pip install huggingface_hub")
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if already exists
    test_dir = output_path / "small_dataset" / "test"
    if test_dir.exists() and (test_dir / "images").exists():
        num_images = len(list((test_dir / "images").glob("*.png")))
        if num_images > 0:
            print(f"DocLayNet-small already exists at {output_path}")
            print(f"  Found {num_images} test images")
            return True

    print(f"Downloading DocLayNet-small dataset from HuggingFace...")
    print(f"  Destination: {output_path}")

    try:
        # Download the zip file
        zip_path = hf_hub_download(
            repo_id="pierreguillou/DocLayNet-small",
            filename="data/dataset_small.zip",
            repo_type="dataset",
            local_dir=output_path,
        )
        print(f"  Downloaded to: {zip_path}")

        # Extract the zip file
        print("  Extracting zip file...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_path)

        print("Download complete!")
        for split in ["train", "val", "test"]:
            split_dir = output_path / "small_dataset" / split / "images"
            if split_dir.exists():
                num_images = len(list(split_dir.glob("*.png")))
                print(f"  {split}: {num_images} images")

        return True

    except Exception as e:
        print(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_doclaynet(output_dir: str, split: str = "test") -> bool:
    """
    Download DocLayNet dataset from IBM Cloud Storage.

    Args:
        output_dir: Directory to save the dataset
        split: Dataset split to download

    Returns:
        True if successful
    """
    import shutil
    import subprocess
    import tempfile
    import zipfile

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # IBM Cloud Storage URL for DocLayNet
    dataset_url = "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip"

    print(f"Downloading DocLayNet dataset...")
    print(f"  Source: {dataset_url}")
    print(f"  Destination: {output_path}")
    print("  Note: This is a ~28 GiB download and may take a while.")

    zip_path = output_path / "DocLayNet_core.zip"

    # Check if already downloaded
    annotations_file = output_path / "annotations" / f"{split}.json"
    images_dir = output_path / "images"

    if annotations_file.exists() and images_dir.exists() and any(images_dir.iterdir()):
        print(f"  Dataset already exists at {output_path}")
        print(f"  To re-download, remove the directory first.")
        return True

    try:
        # Download using curl (more reliable for large files)
        if not zip_path.exists():
            print("  Downloading zip file...")
            result = subprocess.run(
                ["curl", "-L", "-o", str(zip_path), "-C", "-", dataset_url],
                capture_output=False,
            )
            if result.returncode != 0:
                print(f"  Download failed. Try manual download:")
                print(f"    curl -L -o {zip_path} {dataset_url}")
                return False

        # Extract the zip file
        print("  Extracting zip file...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract to temp dir first to handle nested structure
            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extractall(tmpdir)

                # Find the extracted content (may be nested)
                extracted_root = Path(tmpdir)
                contents = list(extracted_root.iterdir())
                if len(contents) == 1 and contents[0].is_dir():
                    extracted_root = contents[0]

                # Copy COCO annotations
                coco_dir = extracted_root / "COCO"
                if coco_dir.exists():
                    annotations_dir = output_path / "annotations"
                    annotations_dir.mkdir(parents=True, exist_ok=True)

                    src_coco = coco_dir / f"{split}.json"
                    if src_coco.exists():
                        print(f"  Copying {split}.json annotations...")
                        shutil.copy(src_coco, annotations_dir / f"{split}.json")
                    else:
                        print(f"  Warning: {split}.json not found in COCO directory")
                        # List available splits
                        available = [f.stem for f in coco_dir.glob("*.json")]
                        print(f"  Available splits: {available}")

                # Copy PNG images for the split
                png_dir = extracted_root / "PNG" / split
                if png_dir.exists():
                    images_dir = output_path / "images"
                    images_dir.mkdir(parents=True, exist_ok=True)

                    png_files = list(png_dir.glob("*.png"))
                    print(f"  Copying {len(png_files)} images...")

                    for i, img_file in enumerate(png_files):
                        if (i + 1) % 500 == 0:
                            print(f"    Copied {i + 1}/{len(png_files)} images")
                        shutil.copy(img_file, images_dir / img_file.name)
                else:
                    print(f"  Warning: PNG/{split} directory not found")

        # Clean up zip file to save space
        print("  Cleaning up zip file...")
        zip_path.unlink()

        print("Download complete!")
        print(f"  Annotations: {output_path / 'annotations'}")
        print(f"  Images: {output_path / 'images'}")
        return True

    except Exception as e:
        print(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_evaluation(
    dataset_path: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    output_path: Optional[str] = None,
) -> BenchmarkResult:
    """
    Run benchmark evaluation.

    Args:
        dataset_path: Path to DocLayNet dataset
        split: Dataset split
        max_samples: Maximum samples to evaluate
        output_path: Path to save results

    Returns:
        BenchmarkResult
    """
    from .layout_detector import create_detector
    from .figure_table_extractor import FigureTableExtractor

    # Check if dataset exists
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        print("\nPlease download the dataset first:")
        print(f"  uv run python -m src.benchmark download --output-dir {dataset_path} --split {split}")
        sys.exit(1)

    # Create detector
    print("Loading DocLayout-YOLO model...")
    detector = create_detector()

    # Create extractor (not used directly in current evaluation)
    extractor = FigureTableExtractor()

    # Run evaluation
    evaluator = BenchmarkEvaluator(dataset_path, split)
    result = evaluator.evaluate(extractor, detector, max_samples)

    # Save results
    if output_path:
        output_file = Path(output_path)
    else:
        output_file = Path("data/benchmark/results/benchmark_report.json")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return result


def generate_comparison_report(input_files: List[str], output_path: str) -> None:
    """
    Generate a comparison report from multiple benchmark results.

    Args:
        input_files: List of paths to benchmark result JSON files
        output_path: Path to save the comparison markdown report
    """
    results = []
    for input_file in input_files:
        file_path = Path(input_file)
        if not file_path.exists():
            print(f"Warning: Report not found: {file_path}, skipping...")
            continue

        with open(file_path, "r") as f:
            report = json.load(f)
            results.append(report)

    if not results:
        print("No valid report files found.")
        sys.exit(1)

    # Generate markdown report
    lines = [
        "# Model Evaluation Comparison Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Dataset: {results[0].get('dataset', 'N/A')}",
        f"- Split: {results[0].get('split', 'N/A')}",
        f"- Samples: {results[0].get('samples_evaluated', 'N/A')}",
        "",
        "## Figure Detection (Picture)",
        "",
        "| Model | Precision | Recall | F1 |",
        "|-------|-----------|--------|-----|",
    ]

    for r in results:
        model = r.get("model", "Unknown")
        metrics = r.get("metrics", {}).get("figure_detection", {})
        lines.append(
            f"| {model} | {metrics.get('precision', 0):.4f} | "
            f"{metrics.get('recall', 0):.4f} | {metrics.get('f1', 0):.4f} |"
        )

    lines.extend([
        "",
        "## Table Detection",
        "",
        "| Model | Precision | Recall | F1 |",
        "|-------|-----------|--------|-----|",
    ])

    for r in results:
        model = r.get("model", "Unknown")
        metrics = r.get("metrics", {}).get("table_detection", {})
        lines.append(
            f"| {model} | {metrics.get('precision', 0):.4f} | "
            f"{metrics.get('recall', 0):.4f} | {metrics.get('f1', 0):.4f} |"
        )

    lines.extend([
        "",
        "## End-to-End (Combined)",
        "",
        "| Model | Precision | Recall | F1 |",
        "|-------|-----------|--------|-----|",
    ])

    for r in results:
        model = r.get("model", "Unknown")
        metrics = r.get("metrics", {}).get("end_to_end", {})
        lines.append(
            f"| {model} | {metrics.get('precision', 0):.4f} | "
            f"{metrics.get('recall', 0):.4f} | {metrics.get('f1', 0):.4f} |"
        )

    lines.extend([
        "",
        "## Detailed Breakdown",
        "",
    ])

    for r in results:
        model = r.get("model", "Unknown")
        breakdown = r.get("per_class_breakdown", {})
        lines.extend([
            f"### {model}",
            "",
            "| Class | TP | FP | FN |",
            "|-------|-----|-----|-----|",
        ])
        for class_name, counts in breakdown.items():
            lines.append(
                f"| {class_name} | {counts.get('true_positives', 0)} | "
                f"{counts.get('false_positives', 0)} | {counts.get('false_negatives', 0)} |"
            )
        lines.append("")

    # Write report
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Comparison report saved to: {output_file}")


def main():
    """CLI entry point for benchmark module."""
    parser = argparse.ArgumentParser(
        description="DocLayNet Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download command
    download_parser = subparsers.add_parser(
        "download", help="Download DocLayNet dataset"
    )
    download_parser.add_argument(
        "--dataset",
        type=str,
        default="doclaynet-small",
        choices=["doclaynet", "doclaynet-small"],
        help="Dataset to download (small ~50MB, full ~28GB)",
    )
    download_parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to download (only for full dataset)",
    )
    download_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for dataset (default: data/benchmark/<dataset>)",
    )
    download_parser.add_argument(
        "--mirror",
        type=str,
        default=None,
        choices=["hf-mirror", "openxlab"],
        help="Use HuggingFace mirror for faster download in China",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate", help="Run benchmark evaluation"
    )
    eval_parser.add_argument(
        "--dataset",
        type=str,
        default="data/benchmark/doclaynet",
        help="Path to dataset directory",
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate",
    )
    eval_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (for quick testing)",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON",
    )
    eval_parser.add_argument(
        "--mirror",
        type=str,
        default=None,
        choices=["hf-mirror", "openxlab"],
        help="Use HuggingFace mirror for model download",
    )

    # Report command
    report_parser = subparsers.add_parser(
        "report", help="Display benchmark report"
    )
    report_parser.add_argument(
        "--input",
        type=str,
        default="data/benchmark/results/benchmark_report.json",
        help="Path to benchmark results JSON",
    )

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Generate comparison report from multiple benchmark results"
    )
    compare_parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to benchmark result JSON files to compare",
    )
    compare_parser.add_argument(
        "--output",
        type=str,
        default="data/benchmark/results/comparison.md",
        help="Output path for comparison markdown report",
    )

    args = parser.parse_args()

    if args.command == "download":
        setup_hf_mirror(args.mirror)
        if args.dataset == "doclaynet-small":
            output_dir = args.output_dir or "data/benchmark/doclaynet-small"
            success = download_doclaynet_small(output_dir)
        else:
            output_dir = args.output_dir or "data/benchmark/doclaynet"
            success = download_doclaynet(output_dir, args.split)
        sys.exit(0 if success else 1)

    elif args.command == "evaluate":
        setup_hf_mirror(args.mirror)
        result = run_evaluation(
            dataset_path=args.dataset,
            split=args.split,
            max_samples=args.max_samples,
            output_path=args.output,
        )

        # Print summary
        print("\n" + "=" * 50)
        print("Benchmark Results Summary")
        print("=" * 50)
        print(f"Dataset: {result.dataset}")
        print(f"Model: {result.model}")
        print(f"Samples: {result.samples_evaluated}")
        print(f"\nFigure Detection:")
        print(f"  Precision: {result.figure_detection.precision:.4f}")
        print(f"  Recall: {result.figure_detection.recall:.4f}")
        print(f"  F1: {result.figure_detection.f1:.4f}")
        print(f"\nTable Detection:")
        print(f"  Precision: {result.table_detection.precision:.4f}")
        print(f"  Recall: {result.table_detection.recall:.4f}")
        print(f"  F1: {result.table_detection.f1:.4f}")
        print(f"\nEnd-to-End:")
        print(f"  Precision: {result.end_to_end.precision:.4f}")
        print(f"  Recall: {result.end_to_end.recall:.4f}")
        print(f"  F1: {result.end_to_end.f1:.4f}")

    elif args.command == "report":
        report_path = Path(args.input)
        if not report_path.exists():
            print(f"Report not found: {report_path}")
            sys.exit(1)

        with open(report_path, "r") as f:
            report = json.load(f)

        print(json.dumps(report, indent=2))

    elif args.command == "compare":
        generate_comparison_report(args.inputs, args.output)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
