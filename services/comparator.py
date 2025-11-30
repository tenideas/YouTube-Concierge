from typing import List
import asyncio
from infra.llm_client import LlmClient, LlmRequest
from services.memory import MemoryService
from domain import VideoId
from config.settings import (
    LLM_COMPARATOR_MODEL_NAME, 
    LLM_COMPARATOR_MAX_TOKENS, 
    LLM_COMPARATOR_TEMPERATURE,
    COMPARISON_CONTEXT_CHAR_LIMIT
)
from config.prompts import COMPARISON_PROMPT_TEMPLATE, COMPARISON_SYSTEM_INSTRUCTION

class ComparisonService:
    '"Service responsible for comparing multiple videos."'
    def __init__(
        self,
        llm_client: LlmClient,
        memory_service: MemoryService,
        model_name: str = LLM_COMPARATOR_MODEL_NAME,
        max_tokens: int = LLM_COMPARATOR_MAX_TOKENS,
        temperature: float = LLM_COMPARATOR_TEMPERATURE,
    ):
        self._llm = llm_client
        self._memory = memory_service
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def compare_videos(
        self, 
        session_id: str, 
        user_id: str, 
        video_ids: List[str], 
        question: str
    ) -> str:
        # Retrieve contexts for all requested video IDs.
        contexts = []
        for vid in video_ids:
            # Fetch from memory.
            ctx = await self._memory.get_video_context(session_id, VideoId(vid), user_id)
            if ctx:
                contexts.append(ctx)
        
        if not contexts:
            return "Error: Could not find any of the requested video IDs in memory."

        # Build a text context containing summaries or snippets of the videos.
        context_parts = []
        for ctx in contexts:
            title = ctx.transcript.title if ctx.transcript else "Unknown"
            # Use summary if available, otherwise use a transcript snippet.
            if ctx.summary and ctx.summary.summary:
                content = f"Summary: {ctx.summary.summary}"
            elif ctx.transcript:
                content = f"Transcript Snippet: {ctx.transcript.text[:COMPARISON_CONTEXT_CHAR_LIMIT]}..."
            else:
                content = "No content available."
                
            context_parts.append(f"VIDEO ID: {ctx.video_id.value}\nTITLE: {title}\nCONTENT:\n{content}")

        full_context = "\n\n".join(context_parts)

        # Format the prompt and send to LLM.
        prompt = COMPARISON_PROMPT_TEMPLATE.format(
            question=question,
            content_context=full_context
        )
        
        request = LlmRequest(
            model_name=self._model_name,
            prompt=prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system_instruction=COMPARISON_SYSTEM_INSTRUCTION
        )

        response = await self._llm.generate_async(request)
        return response.text
