"""
Caption Matching Evaluator

Evaluates the CaptionMatcher algorithm against VLM-generated ground truth.
Uses bbox-based matching instead of ID-based matching for accurate evaluation.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dataset import AnnotationDataset, GroundTruthMatch


@dataclass
class MatchComparison:
    """Comparison between predicted and ground truth match."""

    figure_id: str
    figure_type: str
    page_number: int
    predicted_caption: Optional[str]
    ground_truth_caption: Optional[str]
    is_correct: bool
    error_type: Optional[str] = None  # "false_positive", "false_negative", "wrong_match"
    figure_iou: float = 0.0  # IoU between predicted and ground truth figure bbox
    caption_iou: float = 0.0  # IoU between predicted and ground truth caption bbox

    def to_dict(self) -> Dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "figure_type": self.figure_type,
            "page_number": self.page_number,
            "predicted_caption": self.predicted_caption,
            "ground_truth_caption": self.ground_truth_caption,
            "is_correct": self.is_correct,
            "error_type": self.error_type,
            "figure_iou": round(self.figure_iou, 4),
            "caption_iou": round(self.caption_iou, 4),
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    pdf_name: str
    ground_truth_annotator: str
    total_figures: int
    total_tables: int

    # Metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # Detailed counts
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    correct_no_caption: int = 0  # Correctly predicted no caption

    # Per-type metrics
    figure_metrics: Dict[str, float] = field(default_factory=dict)
    table_metrics: Dict[str, float] = field(default_factory=dict)

    # Detailed comparisons
    comparisons: List[MatchComparison] = field(default_factory=list)

    # Error analysis
    error_analysis: Dict[str, Any] = field(default_factory=dict)

    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pdf_name": self.pdf_name,
            "ground_truth_annotator": self.ground_truth_annotator,
            "created_at": self.created_at,
            "summary": {
                "total_figures": self.total_figures,
                "total_tables": self.total_tables,
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1": round(self.f1, 4),
            },
            "detailed_counts": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "correct_no_caption": self.correct_no_caption,
            },
            "per_type_metrics": {
                "figure": self.figure_metrics,
                "table": self.table_metrics,
            },
            "error_analysis": self.error_analysis,
            "detailed_comparisons": [c.to_dict() for c in self.comparisons],
        }


class CaptionMatchingEvaluator:
    """Evaluates caption matching predictions against ground truth using bbox matching."""

    # IoU thresholds for matching
    FIGURE_IOU_THRESHOLD = 0.5  # Minimum IoU to consider figure bbox matched
    CAPTION_IOU_THRESHOLD = 0.5  # Minimum IoU to consider caption bbox matched

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the evaluator.

        Args:
            confidence_threshold: Minimum confidence for ground truth matches
        """
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _calculate_iou(bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bboxes.

        Args:
            bbox1: First bbox with keys x1, y1, x2, y2
            bbox2: Second bbox with keys x1, y1, x2, y2

        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection
        x1 = max(bbox1["x1"], bbox2["x1"])
        y1 = max(bbox1["y1"], bbox2["y1"])
        x2 = min(bbox1["x2"], bbox2["x2"])
        y2 = min(bbox1["y2"], bbox2["y2"])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union
        area1 = (bbox1["x2"] - bbox1["x1"]) * (bbox1["y2"] - bbox1["y1"])
        area2 = (bbox2["x2"] - bbox2["x1"]) * (bbox2["y2"] - bbox2["y1"])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _find_matching_prediction(
        self,
        gt_match: GroundTruthMatch,
        predictions: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Find the prediction that best matches a ground truth figure by bbox IoU.

        Args:
            gt_match: Ground truth match containing figure_bbox
            predictions: List of prediction dicts with item_bbox

        Returns:
            Tuple of (best matching prediction or None, IoU score)
        """
        best_pred = None
        best_iou = 0.0

        for pred in predictions:
            # Filter by page number
            if pred.get("page_number") != gt_match.page_number:
                continue

            pred_bbox = pred.get("item_bbox")
            if not pred_bbox:
                continue

            iou = self._calculate_iou(gt_match.figure_bbox, pred_bbox)
            if iou > best_iou:
                best_iou = iou
                best_pred = pred

        if best_iou >= self.FIGURE_IOU_THRESHOLD:
            return best_pred, best_iou

        return None, best_iou

    def _check_caption_match(
        self,
        gt_caption_bbox: Optional[Dict[str, float]],
        pred_caption_bbox: Optional[Dict[str, float]],
    ) -> Tuple[bool, float]:
        """
        Check if predicted caption bbox matches ground truth caption bbox.

        Args:
            gt_caption_bbox: Ground truth caption bbox (or None if no caption expected)
            pred_caption_bbox: Predicted caption bbox (or None if no caption predicted)

        Returns:
            Tuple of (is_match, IoU score)
        """
        # Both None = correct match (no caption expected, none predicted)
        if gt_caption_bbox is None and pred_caption_bbox is None:
            return True, 1.0

        # One is None, other is not = mismatch
        if gt_caption_bbox is None or pred_caption_bbox is None:
            return False, 0.0

        # Both have values - compare by IoU
        iou = self._calculate_iou(gt_caption_bbox, pred_caption_bbox)
        return iou >= self.CAPTION_IOU_THRESHOLD, iou

    def evaluate(
        self,
        ground_truth: AnnotationDataset,
        predictions: Dict[str, Any],
    ) -> EvaluationResult:
        """
        Evaluate predictions against ground truth using bbox matching.

        Args:
            ground_truth: VLM-annotated ground truth dataset
            predictions: Detection result with extraction metadata

        Returns:
            EvaluationResult with metrics
        """
        result = EvaluationResult(
            pdf_name=ground_truth.pdf_name,
            ground_truth_annotator=ground_truth.annotator,
            total_figures=0,
            total_tables=0,
            created_at=datetime.now().isoformat(),
        )

        # Get ground truth matches above confidence threshold
        gt_matches = ground_truth.get_high_confidence_matches(self.confidence_threshold)

        # Get all predictions (figures and tables)
        all_predictions = predictions.get("figures", []) + predictions.get("tables", [])

        # Count figures and tables
        figure_count = sum(1 for m in gt_matches if m.figure_type == "figure")
        table_count = sum(1 for m in gt_matches if m.figure_type == "table")
        result.total_figures = figure_count
        result.total_tables = table_count

        # Compare each ground truth item
        figure_comparisons = []
        table_comparisons = []

        for gt_match in gt_matches:
            # Find matching prediction by figure bbox
            matched_pred, figure_iou = self._find_matching_prediction(gt_match, all_predictions)

            # Determine predicted caption
            pred_caption_bbox = None
            pred_caption_id = None
            if matched_pred:
                pred_caption_bbox = matched_pred.get("caption_bbox")
                if pred_caption_bbox:
                    pred_caption_id = f"matched (IoU-based)"

            # Check if caption matches
            is_correct, caption_iou = self._check_caption_match(
                gt_match.caption_bbox,
                pred_caption_bbox,
            )

            # Determine error type
            error_type = None
            if not is_correct:
                if gt_match.caption_bbox is None and pred_caption_bbox is not None:
                    error_type = "false_positive"
                elif gt_match.caption_bbox is not None and pred_caption_bbox is None:
                    error_type = "false_negative"
                else:
                    error_type = "wrong_match"

            comparison = MatchComparison(
                figure_id=gt_match.figure_id,
                figure_type=gt_match.figure_type,
                page_number=gt_match.page_number,
                predicted_caption=pred_caption_id,
                ground_truth_caption=gt_match.caption_id,
                is_correct=is_correct,
                error_type=error_type,
                figure_iou=figure_iou,
                caption_iou=caption_iou,
            )

            result.comparisons.append(comparison)

            if gt_match.figure_type == "figure":
                figure_comparisons.append(comparison)
            else:
                table_comparisons.append(comparison)

            # Update counts
            if is_correct:
                if gt_match.caption_bbox is not None:
                    result.true_positives += 1
                else:
                    result.correct_no_caption += 1
            else:
                if error_type == "false_positive":
                    result.false_positives += 1
                elif error_type == "false_negative":
                    result.false_negatives += 1
                else:  # wrong_match counts as both FP and FN
                    result.false_positives += 1
                    result.false_negatives += 1

        # Calculate overall metrics
        result.precision, result.recall, result.f1 = self._calculate_metrics(
            result.true_positives,
            result.false_positives,
            result.false_negatives,
        )

        # Calculate per-type metrics
        result.figure_metrics = self._calculate_type_metrics(figure_comparisons)
        result.table_metrics = self._calculate_type_metrics(table_comparisons)

        # Error analysis
        result.error_analysis = self._analyze_errors(result.comparisons)

        return result

    def _calculate_metrics(
        self,
        tp: int,
        fp: int,
        fn: int,
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    def _calculate_type_metrics(
        self,
        comparisons: List[MatchComparison],
    ) -> Dict[str, float]:
        """Calculate metrics for a specific type (figure or table)."""
        if not comparisons:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

        tp = sum(1 for c in comparisons if c.is_correct and c.ground_truth_caption is not None)
        fp = sum(1 for c in comparisons if c.error_type == "false_positive")
        fn = sum(1 for c in comparisons if c.error_type == "false_negative")
        correct = sum(1 for c in comparisons if c.is_correct)

        precision, recall, f1 = self._calculate_metrics(tp, fp, fn)
        accuracy = correct / len(comparisons)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "total": len(comparisons),
        }

    def _analyze_errors(
        self,
        comparisons: List[MatchComparison],
    ) -> Dict[str, Any]:
        """Analyze error patterns."""
        errors = [c for c in comparisons if not c.is_correct]

        if not errors:
            return {"total_errors": 0}

        error_types = {}
        for e in errors:
            error_type = e.error_type or "unknown"
            error_types[error_type] = error_types.get(error_type, 0) + 1

        # Group errors by page
        errors_by_page = {}
        for e in errors:
            page = e.page_number
            if page not in errors_by_page:
                errors_by_page[page] = []
            errors_by_page[page].append(
                {
                    "figure_id": e.figure_id,
                    "error_type": e.error_type,
                    "predicted": e.predicted_caption,
                    "expected": e.ground_truth_caption,
                }
            )

        return {
            "total_errors": len(errors),
            "error_type_distribution": error_types,
            "errors_by_page": errors_by_page,
        }

    def save_result(self, result: EvaluationResult, output_path: str) -> str:
        """Save evaluation result to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        return str(output_file)


def evaluate_from_files(
    ground_truth_path: str,
    detection_path: str,
    output_path: Optional[str] = None,
    confidence_threshold: float = 0.7,
) -> EvaluationResult:
    """
    Convenience function to evaluate from file paths.

    Args:
        ground_truth_path: Path to VLM annotation file (caption_annotations.json)
        detection_path: Path to detection result file (result.json or extraction_metadata.json)
        output_path: Optional path to save evaluation report
        confidence_threshold: Minimum confidence for ground truth matches

    Returns:
        EvaluationResult
    """
    # Load ground truth
    gt_dataset = AnnotationDataset.from_annotation_file(ground_truth_path)

    # Load predictions
    with open(detection_path, "r") as f:
        predictions = json.load(f)

    # Run evaluation
    evaluator = CaptionMatchingEvaluator(confidence_threshold=confidence_threshold)
    result = evaluator.evaluate(gt_dataset, predictions)

    # Save if output path provided
    if output_path:
        evaluator.save_result(result, output_path)

    return result
