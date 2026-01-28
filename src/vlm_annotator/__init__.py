"""
VLM Annotator Module

Uses Vision Language Models to annotate figure-caption correspondences
for evaluating the CaptionMatcher algorithm.
"""

from .annotator import CaptionAnnotator
from .base import BaseVLMClient, VLMMatch, VLMResponse
from .image_renderer import AnnotationRenderer

__all__ = [
    "BaseVLMClient",
    "VLMResponse",
    "VLMMatch",
    "CaptionAnnotator",
    "AnnotationRenderer",
]
