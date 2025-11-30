from __future__ import annotations

import logging
from typing import Optional

from config.settings import (
    LLM_QA_MAX_TOKENS,
    LLM_QA_MODEL_NAME,
    LLM_QA_TEMPERATURE,
)
from domain import Transcript
from infra.llm_client import LlmClient, LlmRateLimitError, LlmRequest

logger = logging.getLogger(__name__)


class QuestionAnsweringService:
    '"Service for answering questions based on video transcripts."'

    def __init__(
        self,
        llm_client: LlmClient,
        model_name: str = LLM_QA_MODEL_NAME,
        max_tokens: int = LLM_QA_MAX_TOKENS,
        temperature: float = LLM_QA_TEMPERATURE,
    ) -> None:
        self._llm = llm_client
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        
        # System instruction enforcing answers based ONLY on the transcript.
        self._system_instruction = (
            "You are a helpful assistant. Your task is to answer the user's "
            "question using ONLY the information provided in the transcript. "
            "If the answer is not in the transcript, say so clearly."
        )

    def answer_question(self, transcript: Transcript, question: str) -> str:
        '"Generates an answer to the question using the transcript."'
        if transcript.is_empty:
            logger.warning("Attempted to answer question on empty transcript.")
            return "Error: The transcript for this video is empty, so I cannot answer questions about it."

                       
                       
                       
        # Construct the prompt with transcript context and question.
        prompt = (
            f"Transcript Title: {transcript.title}\n"
            f"Author: {transcript.author}\n"
            f"---------------------\n"
            f"{transcript.text}\n"
            f"---------------------\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        try:
            response = self._llm.generate(
                LlmRequest(
                    model_name=self._model_name,
                    prompt=prompt,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    system_instruction=self._system_instruction,
                )
            )
            
            return response.text.strip()

        except LlmRateLimitError:
            logger.error("Rate limit hit during QA.")
            raise
        except Exception as e:
            logger.exception("Failed to generate answer for question.")
            return "Error: I encountered a problem while trying to answer your question."
