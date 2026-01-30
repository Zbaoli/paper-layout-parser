"""
Benchmark Module

Caption matching benchmark evaluation tools including:
- VLM-assisted annotation
- Benchmark dataset building
- Evaluation against ground truth
- Report generation
"""

from .batch import BatchEvaluator, BenchmarkSummary, DocumentResult
from .builder import DatasetBuilder
from .dataset import AnnotationDataset, GroundTruthMatch, merge_datasets
from .evaluator import CaptionMatchingEvaluator, EvaluationResult, MatchComparison
from .manifest import CaptionBenchmarkDataset, DocumentEntry
from .metrics import calculate_precision_recall_f1
from .reporter import BenchmarkReporter, load_summary_from_json

__all__ = [
    # Dataset
    "AnnotationDataset",
    "GroundTruthMatch",
    "merge_datasets",
    # Evaluator
    "CaptionMatchingEvaluator",
    "EvaluationResult",
    "MatchComparison",
    # Metrics
    "calculate_precision_recall_f1",
    # Manifest
    "CaptionBenchmarkDataset",
    "DocumentEntry",
    # Builder
    "DatasetBuilder",
    # Batch Evaluator
    "BatchEvaluator",
    "BenchmarkSummary",
    "DocumentResult",
    # Reporter
    "BenchmarkReporter",
    "load_summary_from_json",
]
