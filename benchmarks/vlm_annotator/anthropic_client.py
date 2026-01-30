"""
Anthropic VLM Client

Uses Anthropic's Claude for high-quality figure-caption matching.
"""

import json
import os
from typing import Any, Dict, List, Optional

from .base import BaseVLMClient, VLMMatch, VLMResponse
from .prompts import SYSTEM_PROMPT, build_user_prompt


class AnthropicClient(BaseVLMClient):
    """VLM client using Anthropic Claude."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        **kwargs,
    ):
        """
        Initialize Anthropic client.

        Args:
            model: Anthropic model name (default: claude-sonnet-4-20250514)
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        model = model or os.getenv("ANTHROPIC_MODEL", self.DEFAULT_MODEL)
        super().__init__(model, **kwargs)

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                client_kwargs = {}
                if self.api_key:
                    client_kwargs["api_key"] = self.api_key

                self._client = anthropic.Anthropic(**client_kwargs)
            except ImportError:
                raise ImportError(
                    "anthropic package is required. Install with: uv sync --extra vlm"
                )
        return self._client

    def is_available(self) -> bool:
        """Check if Anthropic API key is configured."""
        if not self.api_key:
            print("Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable.")
            return False

        # Just check if key is present - don't make API call
        return True

    def analyze_page(
        self,
        image_path: str,
        figures: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        captions: List[Dict[str, Any]],
    ) -> VLMResponse:
        """Analyze page using Anthropic Claude."""
        if not figures and not tables:
            return VLMResponse(
                success=True,
                matches=[],
                unmatched_captions=[c["id"] for c in captions],
                model=self.client_name,
            )

        try:
            client = self._get_client()

            # Encode image
            image_base64 = self._encode_image_base64(image_path)
            media_type = self._get_image_media_type(image_path)

            # Build prompt
            user_prompt = build_user_prompt(figures, tables, captions)

            # Anthropic API request
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    }
                ],
            )

            raw_response = response.content[0].text
            return self._parse_response(raw_response, figures, tables, captions)

        except Exception as e:
            return VLMResponse(
                success=False,
                error=str(e),
                model=self.client_name,
            )

    def _parse_response(
        self,
        raw_response: str,
        figures: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        captions: List[Dict[str, Any]],
    ) -> VLMResponse:
        """Parse VLM response and extract matches."""
        try:
            # Try to extract JSON from the response
            json_str = raw_response.strip()

            # Handle markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            # Find JSON object
            start_idx = json_str.find("{")
            end_idx = json_str.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = json_str[start_idx:end_idx]

            # Fix trailing commas (common LLM issue)
            json_str = self._fix_json_trailing_commas(json_str)

            data = json.loads(json_str)

            matches = []
            for m in data.get("matches", []):
                fig_id = m.get("figure_id")
                fig_type = m.get("figure_type", "figure")
                cap_id = m.get("caption_id")

                if fig_id is not None:
                    matches.append(
                        VLMMatch(
                            figure_id=fig_id,
                            figure_type=fig_type,
                            caption_id=cap_id,
                            confidence=m.get("confidence", 1.0),
                            reasoning=m.get("reasoning"),
                        )
                    )

            unmatched = data.get("unmatched_captions", [])

            return VLMResponse(
                success=True,
                matches=matches,
                unmatched_captions=unmatched,
                raw_response=raw_response,
                model=self.client_name,
            )

        except json.JSONDecodeError as e:
            return VLMResponse(
                success=False,
                error=f"Failed to parse VLM response as JSON: {e}",
                raw_response=raw_response,
                model=self.client_name,
            )
