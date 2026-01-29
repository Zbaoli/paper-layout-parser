"""
Caption Matching Benchmark - Shared Metrics

Provides shared metric calculation functions used by evaluators.
"""

from typing import Tuple


def calculate_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.

    Args:
        tp: True positives count
        fp: False positives count
        fn: False negatives count

    Returns:
        Tuple of (precision, recall, f1)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
