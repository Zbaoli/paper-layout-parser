"""
Benchmark Module

Caption matching benchmark evaluation tools including:
- VLM-assisted annotation
- Benchmark dataset building
- Evaluation against ground truth
- Report generation
"""

from .dataset import AnnotationDataset, GroundTruthMatch, merge_datasets
from .evaluator import CaptionMatchingEvaluator, EvaluationResult, MatchComparison
from .batch_evaluator import (
    BatchEvaluator,
    BenchmarkSummary,
    CaptionBenchmarkDataset,
    CaptionMatchingBenchmark,
    DatasetBuilder,
    DocumentEntry,
    DocumentResult,
)
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
    # Batch Evaluator
    "BatchEvaluator",
    "BenchmarkSummary",
    "CaptionBenchmarkDataset",
    "CaptionMatchingBenchmark",
    "DatasetBuilder",
    "DocumentEntry",
    "DocumentResult",
    # Reporter
    "BenchmarkReporter",
    "load_summary_from_json",
]
