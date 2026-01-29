"""
VLM Annotator Module

Uses Vision Language Models to annotate figure-caption correspondences
for evaluating the CaptionMatcher algorithm.
"""

from .annotator import CaptionAnnotator, create_vlm_client
from .base import BaseVLMClient, VLMMatch, VLMResponse

# For backward compatibility, import AnnotationRenderer from visualization module
from ..visualization import BoundingBoxRenderer as AnnotationRenderer

__all__ = [
    "BaseVLMClient",
    "VLMResponse",
    "VLMMatch",
    "CaptionAnnotator",
    "create_vlm_client",
    "AnnotationRenderer",  # Backward compatibility alias
]
