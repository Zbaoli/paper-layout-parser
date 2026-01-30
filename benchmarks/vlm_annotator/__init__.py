"""
VLM Annotator Module

Uses Vision Language Models to annotate figure-caption correspondences
for evaluating the CaptionMatcher algorithm.

Unified VLM access via LiteLLM. Configure with environment variables:
    VLM_MODEL: Model name (e.g., "gpt-4o", "claude-sonnet-4-20250514", "ollama/llava:13b")
    VLM_API_KEY: API key for the provider
    VLM_API_BASE: Base URL for third-party providers
"""

# For backward compatibility, import AnnotationRenderer from doclayout
from doclayout.visualization import BoundingBoxRenderer as AnnotationRenderer

from .annotator import CaptionAnnotator, create_vlm_client
from .base import BaseVLMClient, VLMMatch, VLMResponse
from .litellm_client import LiteLLMClient

__all__ = [
    "BaseVLMClient",
    "LiteLLMClient",
    "VLMResponse",
    "VLMMatch",
    "CaptionAnnotator",
    "create_vlm_client",
    "AnnotationRenderer",  # Backward compatibility alias
]
