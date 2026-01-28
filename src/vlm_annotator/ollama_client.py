"""
Ollama VLM Client

Local VLM client using Ollama for free, offline inference.
"""

import json
import os
from typing import Any, Dict, List, Optional

from .base import BaseVLMClient, VLMMatch, VLMResponse
from .prompts import SYSTEM_PROMPT, build_user_prompt


class OllamaClient(BaseVLMClient):
    """VLM client using local Ollama server."""

    DEFAULT_MODEL = "llava:13b"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: int = 120,
        **kwargs,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Ollama model name (default: llava:13b)
            host: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds
        """
        model = model or os.getenv("OLLAMA_MODEL", self.DEFAULT_MODEL)
        super().__init__(model, **kwargs)

        self.host = host or os.getenv("OLLAMA_HOST", self.DEFAULT_HOST)
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None:
            try:
                import httpx

                self._client = httpx.Client(timeout=self.timeout)
            except ImportError:
                raise ImportError(
                    "httpx is required for Ollama client. " "Install with: uv sync --extra vlm"
                )
        return self._client

    def is_available(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            client = self._get_client()
            response = client.get(f"{self.host}/api/tags")
            if response.status_code != 200:
                return False

            # Check if the model is available
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]

            # Check for exact match or partial match
            model_base = self.model.split(":")[0]
            for available_model in models:
                if self.model == available_model or model_base in available_model:
                    return True

            print(f"Model '{self.model}' not found. Available models: {models}")
            return False
        except Exception as e:
            print(f"Ollama not available: {e}")
            return False

    def analyze_page(
        self,
        image_path: str,
        figures: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        captions: List[Dict[str, Any]],
    ) -> VLMResponse:
        """Analyze page using Ollama vision model."""
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

            # Build prompt
            user_prompt = build_user_prompt(figures, tables, captions)

            # Ollama API request
            payload = {
                "model": self.model,
                "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1000,
                },
            }

            response = client.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                return VLMResponse(
                    success=False,
                    error=f"Ollama API error: {response.status_code} - {response.text}",
                    model=self.client_name,
                )

            result = response.json()
            raw_response = result.get("response", "")

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
