from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, Optional

import google.generativeai as genai

from config.settings import (
    GEMINI_API_KEY_ENV_VAR,
    GEMINI_MAX_RETRIES,
    GEMINI_SAFETY_SETTINGS,
)
from infra.llm_client import (
    BaseLlmClient,
    LlmBadRequestError,
    LlmEmptyResponseError,
    LlmRateLimitError,
    LlmRequest,
    LlmResponse,
    LlmSafetyError,
    LlmTimeoutError,
    LlmTransientError,
)

logger = logging.getLogger(__name__)


class GeminiLlmClient(BaseLlmClient):
    '"Implementation of LlmClient using Google\'s Gemini API."'

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_key_env_var: str = GEMINI_API_KEY_ENV_VAR,
        safety_settings: Optional[Dict[str, Any]] = GEMINI_SAFETY_SETTINGS,
        max_retries: int = GEMINI_MAX_RETRIES,
    ) -> None:
        super().__init__(max_retries=max_retries)

        resolved_key = api_key or os.getenv(api_key_env_var)
        if not resolved_key:
            raise RuntimeError(
                "Gemini API key is not configured. Set it explicitly or via "
                f"the {api_key_env_var} environment variable.",
            )

        genai.configure(api_key=resolved_key)
        self._safety_settings = safety_settings

    def _generate_impl(self, request: LlmRequest) -> LlmResponse:
        '# Implementation of the synchronous generation call.'
        model = genai.GenerativeModel(
            model_name=request.model_name,
            safety_settings=self._safety_settings,
            system_instruction=request.system_instruction,
        )

        try:
            response = model.generate_content(
                request.prompt,
                generation_config=self._build_generation_config(request),
            )
        except Exception as error:  # Catch and classify specific Gemini errors.
            raise self._classify_provider_error(error) from error

        self._raise_if_blocked(response)
        text = self._extract_text(response)
        if not text:
            raise LlmEmptyResponseError("Gemini returned an empty response.")

        return LlmResponse(
            text=text,
            raw=response,
            usage=None,
            finish_reason=None,
        )

    def _stream_impl(self, request: LlmRequest) -> Iterator[str]:
        '# Implementation of the synchronous stream call.'
        model = genai.GenerativeModel(
            model_name=request.model_name,
            safety_settings=self._safety_settings,
            system_instruction=request.system_instruction,
        )

        try:
            stream = model.generate_content(
                request.prompt,
                generation_config=self._build_generation_config(request),
                stream=True,
            )
        except Exception as error:  # Catch and classify errors during stream initialization.
            raise self._classify_provider_error(error) from error

        for chunk in stream:
            self._raise_if_blocked(chunk)
            text = self._extract_text(chunk)
            if text:
                yield text

    def _build_generation_config(self, request: LlmRequest) -> Dict[str, Any]:
        "# Map LlmRequest parameters to Gemini's generation config."
        config: Dict[str, Any] = {
            "max_output_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.top_p is not None:
            config["top_p"] = request.top_p
        if request.stop_sequences:
            config["stop_sequences"] = list(request.stop_sequences)
            
        # Add MIME type if specified (e.g. for JSON mode).
        if request.response_mime_type:
            config["response_mime_type"] = request.response_mime_type
            
        return config

    @staticmethod
    def _extract_text(response: Any) -> str:
        '"Extracts text content from a Gemini response object. Handles various response formats, including extracting text from content parts of candidates."'
        if hasattr(response, "text") and isinstance(response.text, str):
            return response.text.strip()

        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""

        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            fragments: list[str] = []
            for part in parts:
                value = getattr(part, "text", None)
                if isinstance(value, str) and value.strip():
                    fragments.append(value.strip())
            if fragments:
                return " ".join(fragments).strip()

        return ""

    @staticmethod
    def _raise_if_blocked(response: Any) -> None:
        '"Checks the response for safety blockages and raises LlmSafetyError."'
        feedback = getattr(response, "prompt_feedback", None)
        if feedback and getattr(feedback, "block_reason", None):
            raise LlmSafetyError(str(feedback.block_reason))

    def _classify_provider_error(self, error: Exception) -> Exception:
        '"Maps Google API errors to internal LlmError exceptions."'
        message = str(error)
        error_code = getattr(error, "code", None)

        if error_code in (429, "RESOURCE_EXHAUSTED"):
            return LlmRateLimitError(message)

        if error_code in (400, "INVALID_ARGUMENT"):
            return LlmBadRequestError(message)

        lowered = message.lower()
        if "safety" in lowered:
            return LlmSafetyError(message)

        if "timeout" in lowered:
            return LlmTimeoutError(message)

        return LlmTransientError(message)
