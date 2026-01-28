"""
Caption Matching Evaluation Module

Provides tools for evaluating caption matching algorithm accuracy
against VLM-generated ground truth.
"""

from .dataset import AnnotationDataset, GroundTruthMatch
from .evaluator import CaptionMatchingEvaluator, EvaluationResult

__all__ = [
    "AnnotationDataset",
    "GroundTruthMatch",
    "CaptionMatchingEvaluator",
    "EvaluationResult",
]
