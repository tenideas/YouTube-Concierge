from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.prompts import (
    GENERIC_SUMMARY_INSTRUCTION,
    SUMMARY_PROMPT_TEMPLATE,
    UNCERTAIN_SUMMARY_INSTRUCTION,
    VIDEO_SUMMARY_INSTRUCTIONS,
)
from config.settings import (
    LLM_SUMMARY_MAX_TOKENS,
    LLM_SUMMARY_MODEL_NAME,
    LLM_SUMMARY_TEMPERATURE,
    SUMMARYT_OUTPUT_DIR,
)
from domain import (
    ClassificationResult,
    ClassificationStatus,
    SummaryResult,
    Transcript,
    VideoCategory,
    VideoId,
)
from infra.cache import JsonFileCache
from infra.llm_client import LlmClient, LlmRateLimitError, LlmRequest

logger = logging.getLogger(__name__)


class SummarizationService:
    '"Service for generating summaries of transcripts."'

    def __init__(
        self,
        llm_client: LlmClient,
        cache: Optional[JsonFileCache] = None,
        model_name: str = LLM_SUMMARY_MODEL_NAME,
        max_tokens: int = LLM_SUMMARY_MAX_TOKENS,
        temperature: float = LLM_SUMMARY_TEMPERATURE,
        system_instruction: str = (
            "You are an expert at writing concise, accurate summaries of "
            "YouTube video transcripts for busy viewers."
        ),
    ) -> None:
        self._llm = llm_client
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_instruction = system_instruction
        
        # Setup file-based cache.
        self._cache = cache or JsonFileCache(SUMMARYT_OUTPUT_DIR)

        self._log_missing_instructions_if_any()

    def summarize(
        self,
        transcript: Transcript,
        classification: Optional[ClassificationResult],
        target_language: Optional[str] = None,
    ) -> SummaryResult:
        if transcript.is_empty:
            return SummaryResult(
                status=ClassificationStatus.UNKNOWN_ERROR,
                summary=None,
                message="Transcript is empty; cannot summarize.",
            )

        summary_lang = target_language if target_language else transcript.language
        cache_key = self._generate_cache_key(transcript.video_id, transcript.language, summary_lang)

        # Check cache first.
        cached_data = self._cache.get(cache_key)
        if cached_data:
            logger.info("Returning cached summary for '%s'", transcript.video_id.value)
            return self._map_cache_to_result(cached_data)

        if classification is not None and classification.status.is_error:
            logger.error("Skipping summarization due to classification error.")
            return SummaryResult(
                status=classification.status,
                message=f"Summarization skipped (Classification: {classification.status.name})",
            )

        used_category = classification.category if classification else None
        instruction = self._select_instruction(classification, used_category)

        # Add language instruction if a specific output language is requested.
        language_instruction = ""
        if target_language:
            language_instruction = (
                f"IMPORTANT: The user has requested the output in '{target_language}'. "
                f"Regardless of the transcript's language, write the summary in {target_language}."
            )

        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            instruction=instruction,
            language_instruction=language_instruction,
            transcript_text=transcript.text.strip(),
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
        except LlmRateLimitError:
            logger.error("Rate limit hit during summarization.")
            raise
        except Exception as ex:
            logger.exception("Failed to summarize transcript.")
            return SummaryResult(
                status=ClassificationStatus.UNKNOWN_ERROR,
                message="LLM request failed during summarization.",
            )

        text = response.text.strip()
        if not text:
            return SummaryResult(
                status=ClassificationStatus.UNKNOWN_ERROR,
                message="LLM returned an empty summary.",
            )

        result = SummaryResult(
            status=ClassificationStatus.OK,
            summary=text,
            used_category=used_category,
            instruction=instruction,
        )

        # Cache the result before returning.
        self._save_to_cache(cache_key, result, transcript.language, summary_lang, transcript.video_id)

        return result

    def _generate_cache_key(self, video_id: VideoId, t_lang: str, s_lang: str) -> str:
        clean_t = t_lang.strip().replace("-", "_")
        clean_s = s_lang.strip().replace("-", "_")
        return f"{video_id.value}_{clean_t}_{clean_s}"

    def _map_cache_to_result(self, data: dict) -> SummaryResult:
        category_name = data.get("used_category")
        used_category = None
        if category_name:
            try:
                used_category = VideoCategory[category_name]
            except KeyError:
                pass
        
        return SummaryResult(
            status=ClassificationStatus.OK,
            summary=data.get("summary"),
            used_category=used_category,
            instruction=data.get("instruction"),
        )

    def _save_to_cache(
        self, 
        key: str, 
        result: SummaryResult, 
        t_lang: str, 
        s_lang: str, 
        video_id: VideoId
    ) -> None:
        if not result.summary:
            return
            
        payload = {
            "video_id": video_id.value,
            "transcript_language": t_lang,
            "summary_language": s_lang,
            "summary": result.summary,
            "used_category": result.used_category.name if result.used_category else None,
            "instruction": result.instruction,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._cache.set(key, payload)

    def _select_instruction(
        self,
        classification: Optional[ClassificationResult],
        category: Optional[VideoCategory],
    ) -> str:
        if classification is None:
            return GENERIC_SUMMARY_INSTRUCTION

        if classification.status is ClassificationStatus.UNCERTAIN:
            return UNCERTAIN_SUMMARY_INSTRUCTION

        if (
            classification.status.is_ok
            and category is not None
            and category in VIDEO_SUMMARY_INSTRUCTIONS
        ):
            return VIDEO_SUMMARY_INSTRUCTIONS[category]

        return GENERIC_SUMMARY_INSTRUCTION

    @staticmethod
    def _missing_instruction_categories() -> list[VideoCategory]:
        missing: list[VideoCategory] = []
        for category in VideoCategory:
            if category not in VIDEO_SUMMARY_INSTRUCTIONS:
                missing.append(category)
        return missing

    def _log_missing_instructions_if_any(self) -> None:
        missing = self._missing_instruction_categories()
        if missing:
            names = ", ".join(category.name for category in missing)
            logger.warning("No summary instructions defined for: %s", names)
