from __future__ import annotations

import abc
import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Protocol,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_BASE_SECONDS = 1.0
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_JITTER_SECONDS = 0.5


@dataclass(frozen=True)
class LlmRequest:
    '"Standardized request object for LLM generation."'

    model_name: str
    prompt: str
    max_tokens: int
    temperature: float
    system_instruction: Optional[str] = None
    stop_sequences: Optional[list[str]] = None
    top_p: Optional[float] = None
    
    # Expected MIME type for the response (e.g., application/json).
    response_mime_type: Optional[str] = None


@dataclass(frozen=True)
class LlmUsage:
    '"Token usage statistics for an LLM request."'

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LlmResponse:
    '"Standardized response object from an LLM."'

    text: str
    raw: Optional[Any] = None
    usage: Optional[LlmUsage] = None
    finish_reason: Optional[str] = None


class LlmError(RuntimeError):
    '"Base exception for all LLM-related errors."'


class LlmProviderError(LlmError):
    '"Error related to the specific LLM provider (API errors)."'


class LlmRateLimitError(LlmError):
    '"Error raised when rate limits or quotas are exceeded."'


class LlmBadRequestError(LlmError):
    '"Error raised when the request is malformed."'


class LlmSafetyError(LlmError):
    '"Error raised when content is blocked by safety filters."'


class LlmEmptyResponseError(LlmError):
    '"Error raised when the LLM returns an empty response."'


class LlmTransientError(LlmError):
    '"Error representing a temporary failure that may be retried."'


class LlmTimeoutError(LlmError):
    '"Error raised when the request times out."'


@runtime_checkable
class LlmClient(Protocol):
    '"Protocol defining the interface for LLM clients."'

    def generate(self, request: LlmRequest) -> LlmResponse:
        '"Generates a complete response for the given request."'

    def stream(self, request: LlmRequest) -> Iterator[str]:
        '"Streams the response chunks for the given request."'

    async def generate_async(self, request: LlmRequest) -> LlmResponse:
        '"Asynchronously generates a complete response."'

    async def stream_async(self, request: LlmRequest) -> AsyncIterator[str]:
        '"Asynchronously streams response chunks."'


class BaseLlmClient(LlmClient, abc.ABC):
    '"Abstract base class providing retry logic and async wrappers."'

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        jitter_seconds: float = DEFAULT_JITTER_SECONDS,
    ) -> None:
        self._max_retries = max_retries
        self._backoff_base_seconds = backoff_base_seconds
        self._backoff_factor = backoff_factor
        self._jitter_seconds = jitter_seconds

    def generate(self, request: LlmRequest) -> LlmResponse:
        '"Generates a response with automatic retries for transient errors."'
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._generate_impl(request)
                if not response.text.strip():
                    raise LlmEmptyResponseError("LLM returned empty text.")
                return response
            except Exception as error:  # specific error handling and backoff logic.
                wrapped = self._wrap_error(error)
                last_error = wrapped

                if not self._is_retryable_error(wrapped) or attempt >= self._max_retries:
                    raise wrapped

                delay = self._compute_backoff_delay(attempt)
                logger.warning(
                    "LLM generate failed (attempt %d/%d): %s; retrying in %.2fs",
                    attempt + 1,
                    self._max_retries + 1,
                    wrapped,
                    delay,
                )
                time.sleep(delay)

        if last_error is not None:
            raise last_error
        raise LlmError("LLM generate failed for unknown reasons.")

    def stream(self, request: LlmRequest) -> Iterator[str]:
        '"Streams response with error wrapping."'
        try:
            for chunk in self._stream_impl(request):
                if chunk:
                    yield chunk
        except Exception as error:  # Wrap errors that occur during iteration.
            raise self._wrap_error(error) from error

    async def generate_async(self, request: LlmRequest) -> LlmResponse:
        '"Async wrapper for generate (runs in thread)."'
        return await asyncio.to_thread(self.generate, request)

    async def stream_async(self, request: LlmRequest) -> AsyncIterator[str]:
        '"Async wrapper for stream."'
        def _collect() -> list[str]:
            '# Helper to collect stream in a thread.'
            return list(self.stream(request))

        chunks = await asyncio.to_thread(_collect)
        for chunk in chunks:
            yield chunk

    @abc.abstractmethod
    def _generate_impl(self, request: LlmRequest) -> LlmResponse:
        '"Provider-specific implementation of generation."'

    @abc.abstractmethod
    def _stream_impl(self, request: LlmRequest) -> Iterator[str]:
        '"Provider-specific implementation of streaming."'

    def _is_retryable_error(self, error: LlmError) -> bool:
        '"Determines if an error should trigger a retry."'
        return isinstance(error, (LlmTransientError, LlmTimeoutError))

    def _compute_backoff_delay(self, attempt: int) -> float:
        '"Calculates the sleep duration for exponential backoff, including jitter for load distribution."'
        exponent = float(attempt)
        base_delay = self._backoff_base_seconds * (self._backoff_factor ** exponent)
        jitter = random.uniform(-self._jitter_seconds, self._jitter_seconds)
        delay = base_delay + jitter
        return max(0.0, delay)

    def _wrap_error(self, error: Exception) -> LlmError:
        '"Wraps arbitrary exceptions into internal LlmError types, prioritizing specific classifications over generic ProviderError."'
        if isinstance(error, LlmError):
            return error

        if isinstance(error, TimeoutError):
            return LlmTimeoutError(str(error))

        return LlmProviderError(str(error))

    def estimate_tokens(self, request: LlmRequest) -> Optional[int]:
        '"Estimates the number of tokens in the request (optional)."'
        return None
