"""
VLM Annotator Module

Uses Vision Language Models to annotate figure-caption correspondences
for evaluating the CaptionMatcher algorithm.

Supports two modes:
1. Direct mode (recommended): VLM analyzes raw PDF pages directly
2. Detection mode (legacy): VLM analyzes pages with YOLO detection overlays

Unified VLM access via LiteLLM. Configure with environment variables:
    VLM_MODEL: Model name (e.g., "gpt-4o", "claude-sonnet-4-20250514", "ollama/llava:13b")
    VLM_API_KEY: API key for the provider
    VLM_API_BASE: Base URL for third-party providers
"""

# For backward compatibility, import AnnotationRenderer from doclayout
from doclayout.visualization import BoundingBoxRenderer as AnnotationRenderer

from .annotator import (
    CaptionAnnotator,
    DirectAnnotationResult,
    DirectPageAnnotation,
    create_vlm_client,
)
from .base import BaseVLMClient, VLMDirectResponse, VLMElement, VLMMatch, VLMResponse
from .litellm_client import LiteLLMClient

__all__ = [
    # VLM Clients
    "BaseVLMClient",
    "LiteLLMClient",
    # Response types
    "VLMResponse",
    "VLMDirectResponse",
    "VLMMatch",
    "VLMElement",
    # Annotator
    "CaptionAnnotator",
    "DirectAnnotationResult",
    "DirectPageAnnotation",
    "create_vlm_client",
    # Backward compatibility
    "AnnotationRenderer",
]
