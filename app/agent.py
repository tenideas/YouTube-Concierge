from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from app.tools import Tool
from config import settings
from domain import WorkflowState, VideoId, VideoContext
from infra.llm_client import LlmClient, LlmRequest, LlmRateLimitError
from infra.youtube_transcript_provider import TranscriptRateLimitError
from services.history import HistoryManager
from services.memory import MemoryService

logger = logging.getLogger(__name__)


class AgentPlan(TypedDict):
    """TypedDict representing the structure of the agent's execution plan."""
    plan: List[Dict[str, Any]]
    response: Optional[str]


class Agent:
    """The main orchestrator that implements a Plan-and-Execute workflow. 
    It generates a complete execution plan using the LLM and then executes 
    tools sequentially using a shared state."""

    def __init__(
        self,
        llm: LlmClient,
        tools: List[Tool],
        memory_service: MemoryService,
        history_manager: HistoryManager,
        model_name: str,
        max_tokens: int,
        temperature: float,
    ) -> None:
        self._llm = llm
        self._tools = {t.name: t for t in tools}
        self._memory = memory_service
        self._history = history_manager
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        
        self._system_instruction = self._build_system_instruction()

    def _build_system_instruction(self) -> str:
        tool_descriptions = []
        for name, tool in self._tools.items():
            tool_descriptions.append(f"- Name: \"{name}\"\n  Description: {tool.description}")

        tools_str = "\n".join(tool_descriptions)

        return (
            "You are a strategic planning engine. Your goal is to map a user's request "
            "into a linear sequence of tool executions.\n\n"
            "AVAILABLE TOOLS:\n"
            f"{tools_str}\n\n"
            "RULES:\n"
            "1. Output valid JSON only.\n"
            "2. The JSON must have two fields: 'plan' (list of tool calls) and 'response' (string).\n"
            "3. 'plan' is a list of objects, each with 'tool' (string) and 'parameters' (object).\n"
            "4. If the user's request requires action, put the steps in 'plan'.\n"
            "5. If the user's request is conversational (e.g., 'hi'), leave 'plan' empty and put your answer in 'response'.\n"
            "6. Context Awareness: If a video is currently active (shown in context), "
            "you do NOT need to pass the URL parameter to tools; they will read it from the state.\n"
            "7. Ambiguity Resolution: If the user's current request is ambiguous (e.g., 'now this', 'that too'), infer the required action from the IMMEDIATELY preceding successful action in the history, ignoring earlier dominant themes.\n"
            "8. Do NOT create infinite loops. Do not include the same tool call twice unless necessary.\n\n"
            "EXAMPLE OUTPUT:\n"
            "{\n"
            "  \"plan\": [\n"
            "    {\"tool\": \"get_transcript\", \"parameters\": {\"url\": \"...\"}},\n"
            "    {\"tool\": \"summarize_video\", \"parameters\": {}}\n"
            "  ],\n"
            "  \"response\": \"I will fetch the transcript and summarize it for you.\"\n"
            "}"
        )

    async def run(self, user_input: str, session_id: str, user_id: str) -> str:
        '"Executes the main agent loop for a user request, handling context, planning, and tool execution."'
        # Initialize the workflow state for this specific user interaction.
        state = WorkflowState(session_id=session_id, user_id=user_id)
        
        current_context = await self._memory.get_current_video_context(session_id, user_id)
        context_description = "NO ACTIVE VIDEO."
        
        if current_context:
            # Inject the active video context into the state if available.
            state.add_video_context(current_context)
            context_description = (
                f"CURRENT VIDEO CONTEXT:\n"
                f"ID: '{current_context.video_id.value}'\n"
                f"Title: '{current_context.transcript.title if current_context.transcript else 'Unknown'}'\n"
                f"URL: '{current_context.url}'"
            )

        # Retrieve the conversation history to provide context for the LLM.
        history_str = await self._history.get_history_context(session_id, user_id)

        # Construct the prompt combining history, current video context, and the user request.
        prompt = (
            f"{history_str}\n\n"
            f"{context_description}\n\n"
            f"USER REQUEST: {user_input}\n\n"
            "Generate your execution plan (JSON):"
        )

        # Invoke the LLM to generate a JSON-structured plan.
        try:
            logger.info("Generating plan for request: %s", user_input)
            plan_data = await asyncio.to_thread(self._generate_plan, prompt)
        except Exception as e:
            logger.exception("Failed to generate plan.")
            return f"Error: Failed to generate an execution plan. Details: {e}"

        plan = plan_data.get("plan", [])
        initial_response = plan_data.get("response")

        # Log the generated plan for debugging purposes.
        logger.debug("Plan Generated: %s", json.dumps(plan_data, indent=2))

        final_response_text = ""

        if not plan:
            # If the agent generated a direct response without tools, return it immediately.
            logger.debug(
                "No steps generated. Agent response: %s", 
                initial_response or "(No response provided)"
            )
            final_response_text = initial_response or "I'm not sure what to do."
            
            # Save the interaction to history before returning.
            await self._history.append_and_compact(session_id, user_id, user_input, final_response_text)
            return final_response_text

        # Iterate through the generated plan steps and execute tools.
        logger.info("Generated plan with %d steps. Executing...", len(plan))
        final_response_text = initial_response if initial_response else "Processing..."

        for i, step in enumerate(plan):
            tool_name = step.get("tool")
            params = step.get("parameters", {})

            if tool_name not in self._tools:
                return f"Error: Plan included unknown tool '{tool_name}'."

            tool = self._tools[tool_name]
            logger.info("Step %d/%d: Executing %s", i + 1, len(plan), tool_name)
            
            # Log state before execution for tracing context changes.
            logger.debug("Active Video IDs (Before Step): %s", state.active_video_ids)

            try:
                # Execute the tool in a separate thread to avoid blocking the async event loop.
                result = await asyncio.to_thread(tool.run, state=state, **params)
                
                # Log the outcome of the tool execution.
                logger.debug(
                    "Step Result: Success=%s, Message='%s', Data Keys=%s", 
                    result.success, 
                    result.message[:100] + "..." if result.message else "None", 
                    list(result.data.keys()) if result.data else "None"
                )

                # Update state success flags based on tool result.
                state.last_step_success = result.success
                if not result.success:
                    state.last_error = result.message
                    logger.warning("Step failed. Stopping execution. Reason: %s", result.message)
                    
                    # Construct a detailed error report if the tool failed.
                    error_report = (
                        f"Execution Stopped at Step {i + 1} ({tool_name}).\n"
                        f"Reason: {result.message}\n"
                        f"--- Plan Status ---\n"
                        f"Completed: {i}/{len(plan)} steps.\n"
                        f"Active Video: {state.active_video_ids[-1] if state.active_video_ids else 'None'}"
                    )
                    
                    # Save the failure to history and return the error message.
                    await self._history.append_and_compact(session_id, user_id, user_input, error_report)
                    return error_report

                # If the tool returned structured data (like a new transcript), persist it to memory.
                if result.data:
                    await self._update_memory_from_result(session_id, result.data, user_id)

                # Update the final response text with the tool's output message.
                final_response_text = result.message

            except (TranscriptRateLimitError, LlmRateLimitError):
                # Re-raise critical rate limit errors to halt the system.
                logger.critical("Rate limit hit during step execution. Propagating error.")
                raise
            except Exception as e:
                logger.exception("Unexpected error executing tool %s", tool_name)
                return f"System Error executing {tool_name}: {e}"

        # Save the successful interaction sequence to history.
        await self._history.append_and_compact(session_id, user_id, user_input, final_response_text)
        return final_response_text

    def _generate_plan(self, prompt: str) -> AgentPlan:
        '"Generates a plan using the LLM and parses the JSON response."'
        response = self._llm.generate(
            LlmRequest(
                model_name=self._model_name,
                prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system_instruction=self._system_instruction,
                response_mime_type="application/json",
            )
        )
        
        # Log the raw text from the LLM before parsing.
        logger.debug("Raw LLM Plan Output: %s", response.text)
        
        return self._clean_and_parse_json(response.text)

    def _clean_and_parse_json(self, text: str) -> AgentPlan:
        '"Cleans markdown formatting from the LLM output and parses it as JSON."'
        text = text.strip()
        
        # Use regex to find the JSON block inside markdown delimiters.
        # This handles cases where the LLM outputs conversational text along with the JSON.
        # Matches ```json ... ``` or just ``` ... ```
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        
        if json_match:
            text = json_match.group(1)
        
        # Fallback: If no code blocks are found, try to locate the first curly brace
        # and the last curly brace to extract the JSON object.
        elif "{" in text:
             first_brace = text.find("{")
             last_brace = text.rfind("}")
             if first_brace != -1 and last_brace != -1:
                 text = text[first_brace : last_brace + 1]

        try:
            data = json.loads(text)
            # Extract the plan and response fields, defaulting to empty list/None.
            return {
                "plan": data.get("plan", []),
                "response": data.get("response")
            }
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM JSON: %s", text)
            # Return a safe fallback structure if JSON parsing fails.
            return {
                "plan": [],
                "response": f"Error parsing plan. Raw output: {text}"
            }

    async def _update_memory_from_result(self, session_id: str, data: Dict[str, Any], user_id: str) -> None:
        '"Updates the video context memory based on data returned from tools. Handles updating existing context or creating a new one if a video ID and URL are provided."'
        video_id: VideoId = data.get("video_id")
        url: str = data.get("url")
        
        if not video_id:
            return

        # Check if we already have context for this video to update.
        existing_ctx = await self._memory.get_video_context(session_id, video_id, user_id=user_id)
        
        if existing_ctx:
            if "transcript" in data: existing_ctx.transcript = data["transcript"]
            if "classification" in data: existing_ctx.classification = data["classification"]
            if "summary" in data: existing_ctx.summary = data["summary"]
            existing_ctx.update_access_time()
            ctx_to_save = existing_ctx
        elif url:
            # Create a new video context if one doesn't exist.
            ctx_to_save = VideoContext(
                video_id=video_id,
                url=url,
                transcript=data.get("transcript"),
                classification=data.get("classification"),
                summary=data.get("summary")
            )
        else:
            # Warn if we have data but no valid video ID or URL to attach it to.
            logger.warning("Could not persist data for '%s': No existing context and no URL provided.", video_id.value)
            return

        await self._memory.save_video_context(session_id, ctx_to_save, user_id=user_id)
