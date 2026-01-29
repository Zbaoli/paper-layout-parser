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


@dataclass
class DocumentEntry:
    """Entry for a document in the benchmark dataset."""

    name: str
    annotation_path: str
    extraction_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "annotation_path": self.annotation_path,
        }
        if self.extraction_path:
            result["extraction_path"] = self.extraction_path
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentEntry":
        return cls(
            name=data["name"],
            annotation_path=data["annotation_path"],
            extraction_path=data.get("extraction_path"),
        )


@dataclass
class CaptionBenchmarkDataset:
    """Dataset manifest for caption matching benchmark."""

    name: str
    version: str
    annotator: str
    documents: List[DocumentEntry] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        # Calculate statistics
        total_figures = 0
        total_tables = 0

        return {
            "name": self.name,
            "version": self.version,
            "annotator": self.annotator,
            "created_at": self.created_at,
            "statistics": {
                "total_documents": len(self.documents),
                "total_figures": total_figures,
                "total_tables": total_tables,
            },
            "documents": [d.to_dict() for d in self.documents],
        }

    @classmethod
    def load(cls, path: str) -> "CaptionBenchmarkDataset":
        """
        Load dataset from dataset.json file.

        Args:
            path: Path to the benchmark directory containing dataset.json

        Returns:
            CaptionBenchmarkDataset instance
        """
        dataset_path = Path(path)
        dataset_file = dataset_path / "dataset.json"

        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "1.0.0"),
            annotator=data.get("annotator", "unknown"),
            created_at=data.get("created_at", ""),
        )

        for doc_data in data.get("documents", []):
            dataset.documents.append(DocumentEntry.from_dict(doc_data))

        return dataset

    def save(self, path: str) -> None:
        """
        Save dataset to dataset.json file.

        Args:
            path: Path to the benchmark directory
        """
        dataset_path = Path(path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        dataset_file = dataset_path / "dataset.json"

        with open(dataset_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def get_annotation_path(self, base_path: str, doc: DocumentEntry) -> Path:
        """Get absolute path to annotation file."""
        return Path(base_path) / doc.annotation_path

    def get_extraction_path(self, base_path: str, doc: DocumentEntry) -> Optional[Path]:
        """Get absolute path to extraction file."""
        if not doc.extraction_path:
            return None
        return Path(base_path) / doc.extraction_path


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


class DatasetBuilder:
    """Builds benchmark dataset from existing annotation files."""

    def __init__(self, name: str = "caption-matching-v1", version: str = "1.0.0"):
        """
        Initialize the dataset builder.

        Args:
            name: Dataset name
            version: Dataset version
        """
        self.name = name
        self.version = version

    def build_from_annotations(
        self,
        annotation_paths: List[str],
        output_dir: str,
        copy_files: bool = True,
    ) -> CaptionBenchmarkDataset:
        """
        Build dataset from annotation files.

        Args:
            annotation_paths: List of paths to caption_annotations.json files
            output_dir: Output directory for the benchmark dataset
            copy_files: Whether to copy annotation files to output directory

        Returns:
            CaptionBenchmarkDataset instance
        """
        import shutil

        output_path = Path(output_dir)
        annotations_dir = output_path / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)

        documents = []
        annotator = "unknown"

        for ann_path in annotation_paths:
            ann_file = Path(ann_path)
            if not ann_file.exists():
                print(f"Warning: Annotation file not found: {ann_file}")
                continue

            # Load annotation to get PDF name
            with open(ann_file, "r", encoding="utf-8") as f:
                ann_data = json.load(f)

            pdf_name = ann_data.get("pdf_name", ann_file.parent.name)
            if annotator == "unknown":
                annotator = ann_data.get("annotator", "unknown")

            # Create document directory
            doc_dir = annotations_dir / pdf_name
            doc_dir.mkdir(parents=True, exist_ok=True)

            # Copy or reference annotation file
            if copy_files:
                dest_ann = doc_dir / "caption_annotations.json"
                shutil.copy(ann_file, dest_ann)
                ann_rel_path = f"annotations/{pdf_name}/caption_annotations.json"
            else:
                ann_rel_path = str(ann_file)

            # Try to find extraction_metadata.json
            extraction_path = ann_file.parent / "extractions" / "extraction_metadata.json"
            ext_rel_path = None
            if extraction_path.exists():
                if copy_files:
                    dest_ext = doc_dir / "extraction_metadata.json"
                    shutil.copy(extraction_path, dest_ext)
                    ext_rel_path = f"annotations/{pdf_name}/extraction_metadata.json"
                else:
                    ext_rel_path = str(extraction_path)

            documents.append(
                DocumentEntry(
                    name=pdf_name,
                    annotation_path=ann_rel_path,
                    extraction_path=ext_rel_path,
                )
            )

        dataset = CaptionBenchmarkDataset(
            name=self.name,
            version=self.version,
            annotator=annotator,
            documents=documents,
            created_at=datetime.now().isoformat(),
        )

        # Save dataset manifest
        dataset.save(str(output_path))

        return dataset

    def build_from_output_dir(
        self,
        output_dir: str,
        benchmark_dir: str,
    ) -> CaptionBenchmarkDataset:
        """
        Build dataset from data/output directory structure.

        Args:
            output_dir: Path to data/output directory
            benchmark_dir: Output directory for benchmark dataset

        Returns:
            CaptionBenchmarkDataset instance
        """
        output_path = Path(output_dir)

        # Find all caption_annotations.json files
        annotation_paths = []
        for ann_file in output_path.glob("*/caption_annotations.json"):
            annotation_paths.append(str(ann_file))

        if not annotation_paths:
            raise ValueError(f"No caption_annotations.json files found in {output_path}")

        return self.build_from_annotations(
            annotation_paths=annotation_paths,
            output_dir=benchmark_dir,
            copy_files=True,
        )


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

        summary.precision, summary.recall, summary.f1 = self._calculate_metrics(
            total_tp, total_fp, total_fn
        )

        # Calculate per-type metrics
        fig_p, fig_r, fig_f1 = self._calculate_metrics(figure_tp, figure_fp, figure_fn)
        summary.figure_metrics = {
            "precision": round(fig_p, 4),
            "recall": round(fig_r, 4),
            "f1": round(fig_f1, 4),
        }

        tbl_p, tbl_r, tbl_f1 = self._calculate_metrics(table_tp, table_fp, table_fn)
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

    def _calculate_metrics(self, tp: int, fp: int, fn: int) -> tuple:
        """Calculate precision, recall, F1."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1


class CaptionMatchingBenchmark:
    """Main class for managing caption matching benchmarks."""

    def __init__(
        self,
        benchmark_dir: str = "benchmark/caption-matching",
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize the benchmark manager.

        Args:
            benchmark_dir: Path to benchmark directory
            confidence_threshold: Minimum confidence for ground truth matches
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.confidence_threshold = confidence_threshold
        self.batch_evaluator = BatchEvaluator(confidence_threshold=confidence_threshold)

    def build_dataset(
        self,
        annotation_paths: List[str],
        name: str = "caption-matching-v1",
        version: str = "1.0.0",
    ) -> CaptionBenchmarkDataset:
        """
        Build a benchmark dataset from annotation files.

        Args:
            annotation_paths: List of paths to caption_annotations.json files
            name: Dataset name
            version: Dataset version

        Returns:
            CaptionBenchmarkDataset
        """
        builder = DatasetBuilder(name=name, version=version)
        return builder.build_from_annotations(
            annotation_paths=annotation_paths,
            output_dir=str(self.benchmark_dir),
            copy_files=True,
        )

    def load_dataset(self) -> CaptionBenchmarkDataset:
        """Load existing dataset from benchmark directory."""
        return CaptionBenchmarkDataset.load(str(self.benchmark_dir))

    def evaluate(
        self,
        predictions_dir: Optional[str] = None,
    ) -> BenchmarkSummary:
        """
        Run evaluation on the benchmark dataset.

        Args:
            predictions_dir: Optional directory containing prediction files

        Returns:
            BenchmarkSummary with results
        """
        dataset = self.load_dataset()
        return self.batch_evaluator.evaluate_dataset(
            dataset=dataset,
            base_path=str(self.benchmark_dir),
            predictions_dir=predictions_dir,
        )

    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validate dataset integrity.

        Returns:
            Validation report
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {},
        }

        try:
            dataset = self.load_dataset()
        except Exception as e:
            report["valid"] = False
            report["errors"].append(f"Failed to load dataset: {e}")
            return report

        report["statistics"]["total_documents"] = len(dataset.documents)

        # Check each document
        missing_annotations = []
        missing_extractions = []

        for doc in dataset.documents:
            ann_path = dataset.get_annotation_path(str(self.benchmark_dir), doc)
            if not ann_path.exists():
                missing_annotations.append(doc.name)

            ext_path = dataset.get_extraction_path(str(self.benchmark_dir), doc)
            if ext_path and not ext_path.exists():
                missing_extractions.append(doc.name)

        if missing_annotations:
            report["valid"] = False
            report["errors"].append(f"Missing annotations: {missing_annotations}")

        if missing_extractions:
            report["warnings"].append(f"Missing extractions: {missing_extractions}")

        report["statistics"]["valid_documents"] = len(dataset.documents) - len(missing_annotations)

        return report
