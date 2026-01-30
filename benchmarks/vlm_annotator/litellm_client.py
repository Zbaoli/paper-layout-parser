"""
LiteLLM VLM Client

Unified VLM client using LiteLLM for multi-provider support.
Supports OpenAI, Anthropic, Ollama, and any LiteLLM-compatible provider.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseVLMClient, VLMDirectResponse, VLMResponse
from .prompts import (
    DIRECT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_direct_user_prompt,
    build_user_prompt,
)

# Suppress LiteLLM's verbose logging and debug info
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("LiteLLM Proxy").setLevel(logging.CRITICAL)
logging.getLogger("LiteLLM Router").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

os.environ.setdefault("LITELLM_LOG", "ERROR")

try:
    import litellm

    litellm.suppress_debug_info = True
    litellm.set_verbose = False
except ImportError:
    pass

# Configure VLM annotator logger
logger = logging.getLogger("vlm_annotator")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Error messages for common API errors
ERROR_MESSAGES = {
    "RateLimitError": "Rate limit exceeded. Wait and retry, or reduce concurrency.",
    "AuthenticationError": "Invalid API key. Check VLM_API_KEY or provider-specific key.",
    "BadRequestError": "Invalid request: {detail}",
    "NotFoundError": "Model not found: {model}. Check model name.",
    "APIConnectionError": "Connection failed: {detail}. Check network or API base URL.",
    "Timeout": "Request timeout. Try increasing timeout or reducing image size.",
    "ServiceUnavailableError": "Service unavailable. The API server may be overloaded.",
    "InternalServerError": "Internal server error. Try again later.",
    "PermissionDeniedError": "Permission denied. Check API key permissions.",
    "UnprocessableEntityError": "Invalid request content: {detail}",
}


def _extract_error_info(exception: Exception, model: str = "") -> Dict[str, Any]:
    """
    Extract useful info from LiteLLM exception.

    Args:
        exception: The exception to extract info from
        model: The model name for context

    Returns:
        Dictionary with error type, status code, message, and friendly message
    """
    error_type = type(exception).__name__
    status_code = getattr(exception, "status_code", None)
    message = str(exception)

    # Clean up LiteLLM's verbose messages
    if "Give Feedback" in message:
        message = message.split("Give Feedback")[0].strip()
    if "Get Help" in message:
        message = message.split("Get Help")[0].strip()

    # Extract more specific message from litellm exceptions
    llm_provider = getattr(exception, "llm_provider", None)
    if llm_provider:
        message = f"[{llm_provider}] {message}"

    # Get friendly message
    friendly_template = ERROR_MESSAGES.get(error_type, "API error: {detail}")
    friendly = friendly_template.format(detail=message[:200], model=model)

    return {
        "type": error_type,
        "status_code": status_code,
        "message": message,
        "friendly": friendly,
    }


def _log_retry_attempt(retry_state):
    """Log retry attempt with useful context."""
    exc = retry_state.outcome.exception()
    attempt = retry_state.attempt_number
    max_attempts = 3

    # Get wait time for next retry
    wait_time = 0
    if retry_state.next_action and hasattr(retry_state.next_action, "sleep"):
        wait_time = retry_state.next_action.sleep

    error_info = _extract_error_info(exc)
    logger.warning(
        f"Retry {attempt}/{max_attempts}: {error_info['type']} - {error_info['friendly']} "
        f"(waiting {wait_time:.1f}s)"
    )


# Retry configuration: 3 attempts with exponential backoff (1s, 2s, 4s)
def _create_retry_decorator():
    """Create retry decorator for VLM API calls."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
        before_sleep=_log_retry_attempt,
    )


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
    KNOWN_PROVIDERS = (
        "openai/",
        "anthropic/",
        "ollama/",
        "azure/",
        "bedrock/",
        "vertex_ai/",
        "huggingface/",
        "together_ai/",
        "replicate/",
        "cohere/",
        "palm/",
    )

    @classmethod
    def _has_provider_prefix(cls, model: str) -> bool:
        """Check if model name already has a LiteLLM provider prefix."""
        # Native OpenAI/Anthropic models don't need prefix
        if model.startswith(("gpt-", "claude-", "o1-", "o3-")):
            return True
        # Check for known provider prefixes
        return model.startswith(cls.KNOWN_PROVIDERS)

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
        # Priority: parameter > VLM_* env var > OPENAI_* env var > default
        model = model or os.getenv("VLM_MODEL") or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)
        api_base = api_base or os.getenv("VLM_API_BASE") or os.getenv("OPENAI_API_BASE")

        # Auto-add openai/ prefix for third-party OpenAI-compatible APIs
        if api_base and not self._has_provider_prefix(model):
            model = f"openai/{model}"

        super().__init__(model, **kwargs)

        self.api_key = api_key or os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_kwargs = kwargs
        self._retry_decorator = _create_retry_decorator()

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

            # Call LiteLLM with retry
            @self._retry_decorator
            def _call_api():
                return litellm.completion(**completion_kwargs)

            response = _call_api()
            raw_response = response.choices[0].message.content
            return self._parse_response(raw_response, figures, tables, captions)

        except ImportError:
            return VLMResponse(
                success=False,
                error="litellm package is required. Install with: uv sync --extra vlm",
                model=self.client_name,
            )
        except Exception as e:
            error_info = _extract_error_info(e, self.model)
            logger.error(f"API call failed: {error_info['type']} - {error_info['friendly']}")
            return VLMResponse(
                success=False,
                error=f"[{error_info['type']}] {error_info['friendly']}",
                error_details={
                    "type": error_info["type"],
                    "status_code": error_info["status_code"],
                    "raw_message": error_info["message"],
                },
                model=self.client_name,
            )

    def analyze_page_direct(self, image_path: str) -> VLMDirectResponse:
        """
        Directly analyze a raw page image without pre-detection metadata.

        This method enables VLM to identify all figures, tables, and captions
        independently, without relying on YOLO detection results.

        Args:
            image_path: Path to the raw page image (not annotated)

        Returns:
            VLMDirectResponse with identified elements and matches
        """
        try:
            import litellm

            # Encode image
            image_base64 = self._encode_image_base64(image_path)
            media_type = self._get_image_media_type(image_path)

            # Build prompt for direct analysis
            user_prompt = build_direct_user_prompt()

            # Build messages with vision content
            messages = [
                {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
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

            # Call LiteLLM with retry
            @self._retry_decorator
            def _call_api():
                return litellm.completion(**completion_kwargs)

            response = _call_api()
            raw_response = response.choices[0].message.content
            return self._parse_direct_response(raw_response)

        except ImportError:
            return VLMDirectResponse(
                success=False,
                error="litellm package is required. Install with: uv sync --extra vlm",
                model=self.client_name,
            )
        except Exception as e:
            error_info = _extract_error_info(e, self.model)
            logger.error(f"API call failed: {error_info['type']} - {error_info['friendly']}")
            return VLMDirectResponse(
                success=False,
                error=f"[{error_info['type']}] {error_info['friendly']}",
                error_details={
                    "type": error_info["type"],
                    "status_code": error_info["status_code"],
                    "raw_message": error_info["message"],
                },
                model=self.client_name,
            )
