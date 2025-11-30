from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config import settings
from infra.llm_client import LlmClient, LlmRequest
from services.memory import MemoryService

logger = logging.getLogger(__name__)


class HistoryManager:
    '"Manages conversation history, including compaction of old turns."'

    def __init__(
        self,
        memory_service: MemoryService,
        llm_client: LlmClient,
        model_name: str = settings.LLM_HISTORY_MODEL_NAME,
        max_tokens: int = settings.LLM_HISTORY_MAX_TOKENS,
        temperature: float = settings.LLM_HISTORY_TEMPERATURE,
    ) -> None:
        self._memory = memory_service
        self._llm = llm_client
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature

        # System instruction for the history compaction/summarization task.
        self._compaction_system_instruction = (
            "You are a conversation consolidator. Your task is to condense the "
            "provided conversation history into a concise summary. Capture key "
            "facts, user intent, specific constraints, and the current state "
            "of the discussion. Do not lose specific details like URLs or names."
        )

    async def get_history_context(self, session_id: str, user_id: str) -> str:
        '"Retrieves the full formatted chat history for the prompt context."'
        history = await self._memory.get_chat_history(session_id, user_id)
        if not history:
            return ""

        lines = ["--- CONVERSATION HISTORY ---"]
        for item in history:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            
            if role == "system_summary":
                lines.append(f"[Previous Summary]: {content}")
            elif role == "user":
                lines.append(f"User: {content}")
            elif role == "model":
                lines.append(f"Agent: {content}")
        
        lines.append("----------------------------")
        return "\n".join(lines)

    async def append_and_compact(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        agent_response: str,
    ) -> None:
        '"Appends a new user/agent turn to history and triggers compaction if the history length exceeds the maximum setting."'
        history = await self._memory.get_chat_history(session_id, user_id)

        # Add the user input and agent response to the history list.
        history.append({"role": "user", "content": user_input})
        history.append({"role": "model", "content": agent_response})

        # Trigger compaction if the history is too long.
        if len(history) > settings.HISTORY_MAX_TURNS:
            try:
                history = await self._compact_history(history)
            except Exception:
                logger.exception(
                    "Failed to compact history for session '%s'. Saving raw history instead.",
                    session_id
                )

        await self._memory.save_chat_history(session_id, history, user_id)

    async def _compact_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        '"Summarizes the oldest part of the conversation history using the LLM and replaces the old turns with a single summary entry."'
        keep_count = settings.HISTORY_RECENT_TURNS_TO_KEEP
        
        # Do nothing if history is short enough.
        if len(history) <= keep_count:
            return history

        recent = history[-keep_count:]
        to_compact = history[:-keep_count]

        if not to_compact:
            return history

        # detailed formatting of the text to be summarized.
        text_lines = []
        for item in to_compact:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            if role == "system_summary":
                text_lines.append(f"Existing Context: {content}")
            elif role == "user":
                text_lines.append(f"User: {content}")
            elif role == "model":
                text_lines.append(f"Agent: {content}")
        
        text_block = "\n".join(text_lines)
        
        prompt = (
            "Summarize the following conversation history. If there is existing "
            "context, merge it into the new summary.\n\n"
            f"{text_block}\n\n"
            "Summary:"
        )

        logger.info("Triggering history compaction (compressing %d items)", len(to_compact))

        # Call LLM to summarize the 'to_compact' block.
        response = await self._llm.generate_async(
            LlmRequest(
                model_name=self._model_name,
                prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system_instruction=self._compaction_system_instruction
            )
        )

        summary_text = response.text.strip()
        
        # Reconstruct history: [Summary] + [Recent Turns].
        new_history = [
            {"role": "system_summary", "content": summary_text}
        ]
        new_history.extend(recent)
        
        return new_history
