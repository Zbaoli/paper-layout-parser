"""
LiteLLM VLM Client

Unified VLM client using LiteLLM for multi-provider support.
Supports OpenAI, Anthropic, Ollama, and any LiteLLM-compatible provider.
"""

import os
from typing import Any, Dict, List, Optional

from .base import BaseVLMClient, VLMResponse
from .prompts import SYSTEM_PROMPT, build_user_prompt


class LiteLLMClient(BaseVLMClient):
    """
    Unified VLM client using LiteLLM.

    Supports all LiteLLM-compatible providers through a single interface.
    Configuration priority: parameters > VLM_* env vars > provider-native env vars > defaults.

    Environment variables:
        VLM_MODEL: Model name (e.g., "gpt-4o", "claude-sonnet-4-20250514", "ollama/llava:13b")
        VLM_API_KEY: API key for the provider
        VLM_API_BASE: Base URL for third-party providers (e.g., SiliconFlow, DeepSeek)

    Model naming conventions:
        - OpenAI: "gpt-4o", "gpt-4-turbo"
        - Anthropic: "claude-sonnet-4-20250514", "claude-opus-4-20250514"
        - Ollama: "ollama/llava:13b", "ollama/llava:34b"
        - Third-party (OpenAI-compatible): "openai/model-name" with VLM_API_BASE set
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        **kwargs,
    ):
        """
        Initialize LiteLLM client.

        Args:
            model: Model name (or set VLM_MODEL env var)
            api_key: API key (or set VLM_API_KEY env var)
            api_base: API base URL for third-party providers (or set VLM_API_BASE env var)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional parameters passed to litellm.completion()
        """
        # Priority: parameter > VLM_* env var > default
        model = model or os.getenv("VLM_MODEL", self.DEFAULT_MODEL)
        super().__init__(model, **kwargs)

        self.api_key = api_key or os.getenv("VLM_API_KEY")
        self.api_base = api_base or os.getenv("VLM_API_BASE")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_kwargs = kwargs

    def is_available(self) -> bool:
        """Check if the VLM client is available and properly configured."""
        # For Ollama models, check if server is running
        if self.model.startswith("ollama/"):
            return self._check_ollama_available()

        # For cloud providers, check if API key is configured
        # LiteLLM will also check provider-native env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY)
        if self.api_key:
            return True

        # Check provider-native environment variables
        if self.model.startswith("claude") or self.model.startswith("anthropic/"):
            if os.getenv("ANTHROPIC_API_KEY"):
                return True
            print("Anthropic API key not configured. Set VLM_API_KEY or ANTHROPIC_API_KEY.")
            return False

        if self.model.startswith("gpt") or self.model.startswith("openai/"):
            if os.getenv("OPENAI_API_KEY"):
                return True
            print("OpenAI API key not configured. Set VLM_API_KEY or OPENAI_API_KEY.")
            return False

        # For other providers, assume available if api_base is set
        if self.api_base:
            return True

        print(f"API key not configured for model: {self.model}")
        print("Set VLM_API_KEY environment variable or pass api_key parameter.")
        return False

    def _check_ollama_available(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            import httpx

            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            with httpx.Client(timeout=10) as client:
                response = client.get(f"{host}/api/tags")
                if response.status_code != 200:
                    print(f"Ollama server not responding at {host}")
                    return False

                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]

                # Extract model name from "ollama/model:tag" format
                model_name = self.model.replace("ollama/", "")
                model_base = model_name.split(":")[0]

                for available_model in models:
                    if model_name == available_model or model_base in available_model:
                        return True

                print(f"Model '{model_name}' not found in Ollama. Available: {models}")
                print(f"Pull the model with: ollama pull {model_name}")
                return False

        except ImportError:
            print("httpx is required for Ollama availability check.")
            return False
        except Exception as e:
            print(f"Ollama not available: {e}")
            print("Make sure Ollama is running: ollama serve")
            return False

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
            image_path: Path to the annotated page image
            figures: List of figure detections with id and bbox
            tables: List of table detections with id and bbox
            captions: List of caption detections with id, bbox, and text

        Returns:
            VLMResponse with matches and analysis results
        """
        if not figures and not tables:
            return VLMResponse(
                success=True,
                matches=[],
                unmatched_captions=[c["id"] for c in captions],
                model=self.client_name,
            )

        try:
            import litellm

            # Encode image
            image_base64 = self._encode_image_base64(image_path)
            media_type = self._get_image_media_type(image_path)

            # Build prompt
            user_prompt = build_user_prompt(figures, tables, captions)

            # Build messages with vision content
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_base64}",
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]

            # Build completion kwargs
            completion_kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            # Add API key if provided
            if self.api_key:
                completion_kwargs["api_key"] = self.api_key

            # Add API base if provided
            if self.api_base:
                completion_kwargs["api_base"] = self.api_base

            # Add any extra kwargs
            completion_kwargs.update(self.extra_kwargs)

            # Call LiteLLM
            response = litellm.completion(**completion_kwargs)

            raw_response = response.choices[0].message.content
            return self._parse_response(raw_response, figures, tables, captions)

        except ImportError:
            return VLMResponse(
                success=False,
                error="litellm package is required. Install with: uv sync --extra vlm",
                model=self.client_name,
            )
        except Exception as e:
            return VLMResponse(
                success=False,
                error=str(e),
                model=self.client_name,
            )
