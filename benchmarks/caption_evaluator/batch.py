"""
Caption Matching Benchmark - Batch Evaluator

Provides batch evaluation capabilities for caption matching algorithm
using VLM-generated ground truth datasets.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dataset import AnnotationDataset
from .evaluator import CaptionMatchingEvaluator, EvaluationResult
from .manifest import CaptionBenchmarkDataset, DocumentEntry
from .metrics import calculate_precision_recall_f1


@dataclass
class DocumentResult:
    """Evaluation result for a single document."""

    name: str
    evaluation: EvaluationResult
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "error": self.error,
        }


@dataclass
class BenchmarkSummary:
    """Summary of benchmark evaluation results."""

    dataset_name: str
    dataset_version: str
    total_documents: int
    successful_evaluations: int

    # Aggregated metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # Per-type aggregated metrics
    figure_metrics: Dict[str, float] = field(default_factory=dict)
    table_metrics: Dict[str, float] = field(default_factory=dict)

    # Detailed counts
    total_true_positives: int = 0
    total_false_positives: int = 0
    total_false_negatives: int = 0

    # Per-document results
    document_results: List[DocumentResult] = field(default_factory=list)

    # Metadata
    evaluator_config: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": {
                "name": self.dataset_name,
                "version": self.dataset_version,
            },
            "summary": {
                "total_documents": self.total_documents,
                "successful_evaluations": self.successful_evaluations,
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1": round(self.f1, 4),
            },
            "per_type_metrics": {
                "figure": self.figure_metrics,
                "table": self.table_metrics,
            },
            "detailed_counts": {
                "true_positives": self.total_true_positives,
                "false_positives": self.total_false_positives,
                "false_negatives": self.total_false_negatives,
            },
            "evaluator_config": self.evaluator_config,
            "created_at": self.created_at,
            "document_results": [r.to_dict() for r in self.document_results],
        }


class BatchEvaluator:
    """Evaluates caption matching across a benchmark dataset."""

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the batch evaluator.

        Args:
            confidence_threshold: Minimum confidence for ground truth matches
        """
        self.confidence_threshold = confidence_threshold
        self.evaluator = CaptionMatchingEvaluator(confidence_threshold=confidence_threshold)

    def evaluate_dataset(
        self,
        dataset: CaptionBenchmarkDataset,
        base_path: str,
        predictions_dir: Optional[str] = None,
    ) -> BenchmarkSummary:
        """
        Evaluate all documents in the dataset.

        Args:
            dataset: Benchmark dataset to evaluate
            base_path: Base path for resolving relative paths in dataset
            predictions_dir: Directory containing prediction files (extraction_metadata.json)
                           If None, uses paths from dataset

        Returns:
            BenchmarkSummary with aggregated results
        """
        summary = BenchmarkSummary(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            total_documents=len(dataset.documents),
            successful_evaluations=0,
            evaluator_config={"confidence_threshold": self.confidence_threshold},
            created_at=datetime.now().isoformat(),
        )

        # Aggregated counts for overall metrics
        total_tp = 0
        total_fp = 0
        total_fn = 0

        # Per-type counts
        figure_tp, figure_fp, figure_fn = 0, 0, 0
        table_tp, table_fp, table_fn = 0, 0, 0

        for doc in dataset.documents:
            try:
                # Load ground truth
                ann_path = dataset.get_annotation_path(base_path, doc)
                gt_dataset = AnnotationDataset.from_annotation_file(str(ann_path))

                # Load predictions
                pred_path = self._find_predictions(doc, base_path, predictions_dir)
                if not pred_path:
                    raise FileNotFoundError(f"Predictions not found for {doc.name}")

                with open(pred_path, "r", encoding="utf-8") as f:
                    predictions = json.load(f)

                # Run evaluation
                result = self.evaluator.evaluate(gt_dataset, predictions)

                # Aggregate results
                total_tp += result.true_positives
                total_fp += result.false_positives
                total_fn += result.false_negatives

                # Per-type aggregation
                fig_metrics = result.figure_metrics
                tbl_metrics = result.table_metrics

                if fig_metrics.get("total", 0) > 0:
                    # Approximate TP/FP/FN from per-type metrics
                    fig_total = fig_metrics.get("total", 0)
                    fig_accuracy = fig_metrics.get("accuracy", 0)
                    figure_tp += int(fig_total * fig_accuracy)
                    figure_fn += int(fig_total * (1 - fig_accuracy))

                if tbl_metrics.get("total", 0) > 0:
                    tbl_total = tbl_metrics.get("total", 0)
                    tbl_accuracy = tbl_metrics.get("accuracy", 0)
                    table_tp += int(tbl_total * tbl_accuracy)
                    table_fn += int(tbl_total * (1 - tbl_accuracy))

                summary.document_results.append(DocumentResult(name=doc.name, evaluation=result))
                summary.successful_evaluations += 1

            except Exception as e:
                summary.document_results.append(
                    DocumentResult(name=doc.name, evaluation=None, error=str(e))
                )

        # Calculate overall metrics
        summary.total_true_positives = total_tp
        summary.total_false_positives = total_fp
        summary.total_false_negatives = total_fn

        summary.precision, summary.recall, summary.f1 = calculate_precision_recall_f1(
            total_tp, total_fp, total_fn
        )

        # Calculate per-type metrics
        fig_p, fig_r, fig_f1 = calculate_precision_recall_f1(figure_tp, figure_fp, figure_fn)
        summary.figure_metrics = {
            "precision": round(fig_p, 4),
            "recall": round(fig_r, 4),
            "f1": round(fig_f1, 4),
        }

        tbl_p, tbl_r, tbl_f1 = calculate_precision_recall_f1(table_tp, table_fp, table_fn)
        summary.table_metrics = {
            "precision": round(tbl_p, 4),
            "recall": round(tbl_r, 4),
            "f1": round(tbl_f1, 4),
        }

        return summary

    def _find_predictions(
        self,
        doc: DocumentEntry,
        base_path: str,
        predictions_dir: Optional[str],
    ) -> Optional[Path]:
        """Find predictions file for a document."""
        # Try dataset-specified path first
        if doc.extraction_path:
            ext_path = Path(base_path) / doc.extraction_path
            if ext_path.exists():
                return ext_path

        # Try predictions directory
        if predictions_dir:
            pred_path = (
                Path(predictions_dir) / doc.name / "extractions" / "extraction_metadata.json"
            )
            if pred_path.exists():
                return pred_path

            # Also try result.json
            result_path = Path(predictions_dir) / doc.name / "result.json"
            if result_path.exists():
                return result_path

        return None
