"""
Caption Matching Evaluator

Evaluates the CaptionMatcher algorithm against VLM-generated ground truth.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dataset import AnnotationDataset


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "figure_type": self.figure_type,
            "page_number": self.page_number,
            "predicted_caption": self.predicted_caption,
            "ground_truth_caption": self.ground_truth_caption,
            "is_correct": self.is_correct,
            "error_type": self.error_type,
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
    """Evaluates caption matching predictions against ground truth."""

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the evaluator.

        Args:
            confidence_threshold: Minimum confidence for ground truth matches
        """
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        ground_truth: AnnotationDataset,
        predictions: Dict[str, Any],
    ) -> EvaluationResult:
        """
        Evaluate predictions against ground truth.

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

        # Build prediction mapping from extraction metadata
        pred_map = self._build_prediction_map(predictions)

        # Count figures and tables
        figure_count = sum(1 for m in gt_matches if m.figure_type == "figure")
        table_count = sum(1 for m in gt_matches if m.figure_type == "table")
        result.total_figures = figure_count
        result.total_tables = table_count

        # Compare each ground truth item
        figure_comparisons = []
        table_comparisons = []

        for gt_match in gt_matches:
            fig_id = gt_match.figure_id
            gt_caption = gt_match.caption_id
            pred_caption = pred_map.get(fig_id)

            # Determine correctness
            is_correct = self._compare_captions(pred_caption, gt_caption)

            # Determine error type
            error_type = None
            if not is_correct:
                if gt_caption is None and pred_caption is not None:
                    error_type = "false_positive"
                elif gt_caption is not None and pred_caption is None:
                    error_type = "false_negative"
                else:
                    error_type = "wrong_match"

            comparison = MatchComparison(
                figure_id=fig_id,
                figure_type=gt_match.figure_type,
                page_number=gt_match.page_number,
                predicted_caption=pred_caption,
                ground_truth_caption=gt_caption,
                is_correct=is_correct,
                error_type=error_type,
            )

            result.comparisons.append(comparison)

            if gt_match.figure_type == "figure":
                figure_comparisons.append(comparison)
            else:
                table_comparisons.append(comparison)

            # Update counts
            if is_correct:
                if gt_caption is not None:
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

    def _build_prediction_map(self, predictions: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """
        Build mapping from figure IDs to predicted caption IDs.

        Args:
            predictions: Detection result with extraction metadata

        Returns:
            Dict mapping figure_id to caption_id
        """
        pred_map = {}

        # Try to load from extraction metadata if available
        figures = predictions.get("figures", [])
        tables = predictions.get("tables", [])

        for fig in figures:
            fig_id = fig.get("item_id", "")
            # Convert item_id format (fig_01_01 -> fig_01_01)
            caption_id = None
            if fig.get("caption_bbox"):
                # Derive caption ID from figure ID
                parts = fig_id.split("_")
                if len(parts) >= 3:
                    page_num = parts[1]
                    # Caption ID follows a similar pattern
                    caption_id = f"cap_{page_num}_{parts[2]}"

            pred_map[fig_id] = caption_id

        for tbl in tables:
            tbl_id = tbl.get("item_id", "")
            caption_id = None
            if tbl.get("caption_bbox"):
                parts = tbl_id.split("_")
                if len(parts) >= 3:
                    page_num = parts[1]
                    caption_id = f"cap_{page_num}_{parts[2]}"

            pred_map[tbl_id] = caption_id

        return pred_map

    def _compare_captions(
        self,
        predicted: Optional[str],
        ground_truth: Optional[str],
    ) -> bool:
        """Compare predicted and ground truth caption IDs."""
        # Both None = correct (no caption expected, none predicted)
        if predicted is None and ground_truth is None:
            return True

        # One is None, other is not = incorrect
        if predicted is None or ground_truth is None:
            return False

        # Both have values - compare
        return predicted == ground_truth

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
