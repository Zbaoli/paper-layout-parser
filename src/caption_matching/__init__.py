"""
Caption Matching Evaluation Module

Provides tools for evaluating caption matching algorithm accuracy
against VLM-generated ground truth.
"""

from .benchmark import (
    BatchEvaluator,
    BenchmarkSummary,
    CaptionBenchmarkDataset,
    CaptionMatchingBenchmark,
    DatasetBuilder,
    DocumentEntry,
    DocumentResult,
)
from .dataset import AnnotationDataset, GroundTruthMatch
from .evaluator import CaptionMatchingEvaluator, EvaluationResult
from .reporter import BenchmarkReporter, load_summary_from_json

__all__ = [
    # Dataset
    "AnnotationDataset",
    "GroundTruthMatch",
    # Evaluator
    "CaptionMatchingEvaluator",
    "EvaluationResult",
    # Benchmark
    "CaptionBenchmarkDataset",
    "CaptionMatchingBenchmark",
    "BatchEvaluator",
    "BenchmarkSummary",
    "DatasetBuilder",
    "DocumentEntry",
    "DocumentResult",
    # Reporter
    "BenchmarkReporter",
    "load_summary_from_json",
]
