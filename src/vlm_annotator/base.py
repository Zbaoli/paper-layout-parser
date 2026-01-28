"""
VLM Client Abstract Base Class

Defines the interface for VLM clients used in figure-caption annotation.
"""

import base64
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class VLMMatch:
    """Represents a single figure/table to caption match identified by VLM."""

    figure_id: int  # F1, F2, ... -> 1, 2, ...
    figure_type: str  # "figure" or "table"
    caption_id: Optional[int]  # C1, C2, ... -> 1, 2, ... or None if no match
    confidence: float = 1.0
    reasoning: Optional[str] = None


@dataclass
class VLMResponse:
    """Response from VLM analysis."""

    success: bool
    matches: List[VLMMatch] = field(default_factory=list)
    unmatched_captions: List[int] = field(default_factory=list)  # Caption IDs with no match
    raw_response: str = ""
    model: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "matches": [
                {
                    "figure_id": m.figure_id,
                    "figure_type": m.figure_type,
                    "caption_id": m.caption_id,
                    "confidence": m.confidence,
                    "reasoning": m.reasoning,
                }
                for m in self.matches
            ],
            "unmatched_captions": self.unmatched_captions,
            "model": self.model,
            "error": self.error,
        }


class BaseVLMClient(ABC):
    """Abstract base class for VLM clients."""

    def __init__(self, model: str, **kwargs):
        """
        Initialize the VLM client.

        Args:
            model: Model identifier
            **kwargs: Additional client-specific configuration
        """
        self.model = model
        self.config = kwargs

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the VLM client is available and properly configured.

        Returns:
            True if the client can be used, False otherwise
        """
        pass

    @abstractmethod
    def analyze_page(
        self,
        image_path: str,
        figures: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        captions: List[Dict[str, Any]],
    ) -> VLMResponse:
        """
        Analyze a page image to determine figure/table-caption correspondences.

        Args:
            image_path: Path to the annotated page image with numbered boxes
            figures: List of figure detections with id and bbox
            tables: List of table detections with id and bbox
            captions: List of caption detections with id, bbox, and text

        Returns:
            VLMResponse with matches and analysis results
        """
        pass

    def _encode_image_base64(self, image_path: str) -> str:
        """
        Encode an image file as base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _get_image_media_type(self, image_path: str) -> str:
        """
        Get the media type for an image file.

        Args:
            image_path: Path to the image file

        Returns:
            Media type string (e.g., "image/png")
        """
        suffix = Path(image_path).suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return media_types.get(suffix, "image/png")

    @property
    def client_name(self) -> str:
        """Get a descriptive name for this client."""
        return f"{self.__class__.__name__}:{self.model}"

    def _fix_json_trailing_commas(self, json_str: str) -> str:
        """
        Fix trailing commas in JSON string (common LLM output issue).

        Args:
            json_str: JSON string potentially with trailing commas

        Returns:
            Fixed JSON string
        """
        # Remove trailing commas before ] or }
        json_str = re.sub(r",\s*]", "]", json_str)
        json_str = re.sub(r",\s*}", "}", json_str)
        return json_str
