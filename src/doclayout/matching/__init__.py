"""
Caption Matching Module

Provides algorithms for matching figures/tables with their captions
based on spatial proximity.
"""

from .types import SearchDirection, ExtractedItem, ExtractionResult
from .caption_matcher import CaptionMatcher

__all__ = [
    "CaptionMatcher",
    "SearchDirection",
    "ExtractedItem",
    "ExtractionResult",
]
