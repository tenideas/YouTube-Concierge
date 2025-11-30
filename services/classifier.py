from __future__ import annotations

import logging
import re
from typing import Optional

from config.prompts import (
    CLASSIFIER_PROMPT_TEMPLATE,
    CLASSIFIER_SYSTEM_INSTRUCTION,
)
from config.settings import (
    LLM_CLASSIFIER_MAX_TOKENS,
    LLM_CLASSIFIER_MODEL_NAME,
    LLM_CLASSIFIER_TEMPERATURE,
)
from domain import (
    ClassificationResult,
    ClassificationStatus,
    Transcript,
    VideoCategory,
)
from infra.llm_client import LlmClient, LlmRateLimitError, LlmRequest

logger = logging.getLogger(__name__)


class Classifier:
    '"Service responsible for classifying video content using an LLM."'

    def __init__(
        self,
        llm_client: LlmClient,
        model_name: str = LLM_CLASSIFIER_MODEL_NAME,
        max_tokens: int = LLM_CLASSIFIER_MAX_TOKENS,
        temperature: float = LLM_CLASSIFIER_TEMPERATURE,
        system_instruction: str = CLASSIFIER_SYSTEM_INSTRUCTION,
    ) -> None:
        self._llm = llm_client
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_instruction = system_instruction

    def classify(self, transcript: Transcript) -> ClassificationResult:
        '"Classifies the given transcript into a predefined category."'
        if transcript.is_empty:
            logger.warning(
                "Transcript for '%s' is empty; cannot classify.",
                transcript.video_id.value,
            )
            return ClassificationResult(
                status=ClassificationStatus.CLASSIFIER_ERROR,
                category=None,
                reason="Transcript text is empty.",
                raw_model_output=None,
            )

        # Construct the prompt with video metadata and a snippet of the transcript.
        prompt = CLASSIFIER_PROMPT_TEMPLATE.format(
            title=transcript.title,
            author=transcript.author,
            transcript_text=transcript.text.strip()[:10000], # Truncate transcript to avoid token limits.
        )
        try:
            response = self._llm.generate(
                LlmRequest(
                    model_name=self._model_name,
                    prompt=prompt,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    system_instruction=self._system_instruction,
                ),
            )
        # Handle rate limits specifically.
        except LlmRateLimitError:
            logger.error("Rate limit hit during classification for '%s'", transcript.video_id.value)
            raise
        except Exception:  # Catch generic errors and return a Classifier Error status.
            logger.exception(
                "Failed to classify transcript for '%s'",
                transcript.video_id.value,
            )
            return ClassificationResult(
                status=ClassificationStatus.CLASSIFIER_ERROR,
                category=None,
                reason="LLM request failed during classification.",
                raw_model_output=None,
            )

        text = response.text.strip()
        parsed_category = self._parse_category(text)

        if parsed_category is not None:
            logger.info(
                "Transcript for '%s' classified as '%s'",
                transcript.video_id.value,
                parsed_category.name,
            )
            return ClassificationResult(
                status=ClassificationStatus.OK,
                category=parsed_category,
                reason=None,
                raw_model_output=text,
            )

        logger.warning(
            "LLM output for '%s' did not match known categories: '%s'",
            transcript.video_id.value,
            text,
        )
        return ClassificationResult(
            status=ClassificationStatus.UNCERTAIN,
            category=None,
            reason="LLM output could not be parsed into a known category.",
            raw_model_output=text,
        )

    @staticmethod
    def _parse_category(raw: str) -> Optional[VideoCategory]:
        '"Parses the LLM output to find a matching VideoCategory."'
        # Clean up markdown formatting.
        cleaned = raw.strip().replace("**", "").replace('"', "").replace("'", "")
        
        # Normalize string for comparison.
        normalized = cleaned.replace(" ", "_").replace("-", "_").upper()
        
                       
        normalized = normalized.rstrip(".,;!")

        # checks for exact matches.
        for category in VideoCategory:
            if normalized == category.name:
                return category
                
                       
                       
        for category in VideoCategory:
            # Check for the category name as a whole word in the response.
            pattern = r"\b" + re.escape(category.name) + r"\b"
            if re.search(pattern, normalized):
                return category
                
        return None
