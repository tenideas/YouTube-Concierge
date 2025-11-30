from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from config import settings
from domain import (
    ClassificationResult,
    ClassificationStatus,
    ToolResult,
    TranscriptSource,
    VideoId,
    VideoContext,
    WorkflowState,
)
from services.classifier import Classifier
from services.comparator import ComparisonService
from services.memory import MemoryService
from services.qa_service import QuestionAnsweringService
from services.summarizer import SummarizationService
from services.transcript_service import TranscriptService


class Tool(ABC):
    '"Abstract base class for all tools available to the agent."'

    @property
    @abstractmethod
    def name(self) -> str:
        '"The unique name of the tool used by the LLM for selection."'

    @property
    @abstractmethod
    def description(self) -> str:
        '"A detailed description of what the tool does and its parameters."'

    @abstractmethod
    def run(self, state: WorkflowState, **kwargs) -> ToolResult:
        '"Executes the tool logic given the current workflow state and arguments."'


class FetchTranscriptTool(Tool):
    '"Tool to fetch and store the transcript of a YouTube video."'

    def __init__(self, transcript_service: TranscriptService) -> None:
        self._service = transcript_service

    @property
    def name(self) -> str:
        return "get_transcript"

    @property
    def description(self) -> str:
        return (
            "Retrieves and saves the transcript for a specific YouTube video. "
            "Use this when the user wants to read, save, or access the full text. "
            "Parameters: 'url' (required), 'language' (optional 2-letter code, e.g., 'en', 'ru')."
        )

    def run(self, state: WorkflowState, **kwargs) -> ToolResult:
        url = kwargs.get("url")
        language = kwargs.get("language")

        # If no URL provided, attempt to use the URL from the currently active video context.
        if not url:
            current_video = state.get_current_video()
            if current_video:
                url = current_video.url
            else:
                return ToolResult(
                    message="Error: 'url' parameter is missing and no active video found in context.",
                    success=False
                )

        try:
            video_id = self._service.parse_video_id(url)
        except ValueError:
            return ToolResult(
                message=f"Error: Could not parse a valid YouTube Video ID from url '{url}'.",
                success=False
            )

        # Check if the transcript is already in memory to avoid redundant API calls.
        # Return cached transcript data if available.
        if video_id.value in state.videos:
            existing_ctx = state.videos[video_id.value]
            if existing_ctx.transcript:
                return ToolResult(
                    message=f"Transcript for '{existing_ctx.transcript.title}' is already loaded.",
                    success=True,
                    data={"video_id": video_id, "url": url, "transcript": existing_ctx.transcript}
                )

        result = self._service.get_transcript(video_id, language=language)

        if not result.status.is_ok:
            return ToolResult(
                message=f"Failed to get transcript. Reason: {result.message or result.status.name}",
                success=False
            )

        if not result.transcript:
            return ToolResult(
                message="Error: Internal system error (Transcript object is missing).",
                success=False
            )

        source_msg = "retrieved from cache" if result.transcript.source == TranscriptSource.CACHE else "fetched from YouTube"
        snippet = result.transcript.text[:1000]
        
        msg = (
            f"Success. Transcript for '{result.transcript.title}' "
            f"({result.transcript.language}) has been {source_msg}.\n\n"
            f"--- Snippet (First 1000 characters) ---\n"
            f"{snippet}...\n"
            f"---------------------------------------\n"
            f"[Full transcript text is available in the agent's memory]"
        )

        # Create or update the video context with the new transcript.
        ctx = state.videos.get(video_id.value) or VideoContext(video_id=video_id, url=url)
        ctx.transcript = result.transcript
        # Explicitly add the context to the workflow state.
        state.add_video_context(ctx)

        return ToolResult(
            message=msg, 
            success=True,
            data={"video_id": video_id, "url": url, "transcript": result.transcript}
        )


class ClassifyVideoTool(Tool):
    '"Tool to classify a video into a specific category."'

    def __init__(
        self,
        transcript_service: TranscriptService,
        classifier: Classifier,
    ) -> None:
        self._transcript_service = transcript_service
        self._classifier = classifier

    @property
    def name(self) -> str:
        return "classify_video"

    @property
    def description(self) -> str:
        return (
            "Determines the category of a YouTube video (e.g., History, Vlog, Tutorial). "
            "Use this when the user asks 'what type of video is this?' or wants to know the topic. "
            "Parameters: 'url' (optional if context exists)."
        )

    def run(self, state: WorkflowState, **kwargs) -> ToolResult:
        url = kwargs.get("url")
        language = kwargs.get("language")

        # Fallback to active video context if URL is missing.
        if not url:
            current_video = state.get_current_video()
            if current_video:
                url = current_video.url
            else:
                return ToolResult(
                    message="Error: 'url' parameter is missing and no active video found in context.",
                    success=False
                )

        try:
            video_id = self._transcript_service.parse_video_id(url)
        except ValueError:
            return ToolResult(
                message=f"Error: Could not parse a valid YouTube Video ID from url '{url}'.",
                success=False
            )

        # Retrieve transcript from memory or fetch it if missing.
        transcript = None
        ctx = state.videos.get(video_id.value)
        if ctx and ctx.transcript:
            transcript = ctx.transcript
        else:
            # Fetch transcript if not already in context.
            t_result = self._transcript_service.get_transcript(video_id, language=language)
            if not t_result.status.is_ok or not t_result.transcript:
                return ToolResult(
                    message=f"Cannot classify: Failed to get transcript. Reason: {t_result.message or t_result.status.name}",
                    success=False
                )
            transcript = t_result.transcript

        # Perform classification using the LLM-based classifier.
        c_result = self._classifier.classify(transcript)
        if not c_result.status.is_ok:
            return ToolResult(
                message=f"Classification failed. Reason: {c_result.reason or c_result.status.name}",
                success=False
            )

        category_name = c_result.category.name if c_result.category else "Unknown"
        msg = (
            f"Classification Result for '{transcript.title}':\n"
            f"Category: {category_name}\n"
            f"Reasoning: {c_result.reason or 'None provided.'}"
        )

        # Persist the classification result to the video context.
        if not ctx:
            ctx = VideoContext(video_id=video_id, url=url)
        
        ctx.transcript = transcript
        ctx.classification = c_result
        state.add_video_context(ctx)

        # Return the classification result to the agent.
        return ToolResult(
            message=msg, 
            success=True,
            data={"video_id": video_id, "url": url, "classification": c_result}
        )


class SummarizeVideoTool(Tool):
    '"Tool to generate a summary of the video content."'

    def __init__(
        self,
        transcript_service: TranscriptService,
        classifier: Classifier,
        summarizer: SummarizationService,
    ) -> None:
        self._transcript_service = transcript_service
        self._classifier = classifier
        self._summarizer = summarizer

    @property
    def name(self) -> str:
        return "summarize_video"

    @property
    def description(self) -> str:
        return (
            "Generates a concise summary of a YouTube video. "
            "Use this for requests like 'summarize this', 'tl;dr', or 'what is this about?'. "
            "Parameters: 'url' (optional if context exists)."
        )

    def run(self, state: WorkflowState, **kwargs) -> ToolResult:
        url = kwargs.get("url")
        language = kwargs.get("language")

        if not url:
            current_video = state.get_current_video()
            if current_video:
                url = current_video.url
            else:
                return ToolResult(
                    message="Error: 'url' parameter is missing and no active video found in context.",
                    success=False
                )

        try:
            video_id = self._transcript_service.parse_video_id(url)
        except ValueError:
            return ToolResult(
                message=f"Error: Could not parse a valid YouTube Video ID from url '{url}'.",
                success=False
            )

        # Ensure transcript is available (from memory or fetch).
        transcript = None
        ctx = state.videos.get(video_id.value)
        if ctx and ctx.transcript:
            transcript = ctx.transcript
        else:
            t_result = self._transcript_service.get_transcript(video_id, language=language)
            if not t_result.status.is_ok or not t_result.transcript:
                return ToolResult(
                    message=f"Cannot summarize: Failed to get transcript. Reason: {t_result.message or t_result.status.name}",
                    success=False
                )
            transcript = t_result.transcript

        # Ensure classification is available to select the correct summary prompt.
        classification = ctx.classification if ctx else None
        if not classification:
            classification = self._classifier.classify(transcript)
        
        # Generate the summary.
        s_result = self._summarizer.summarize(
            transcript=transcript,
            classification=classification
        )

        if not s_result.status.is_ok:
            return ToolResult(
                message=f"Summarization failed. Reason: {s_result.message or s_result.status.name}",
                success=False
            )

        msg = (
            f"Summary for '{transcript.title}' ({transcript.author}):\n"
            f"{'-'*40}\n"
            f"{s_result.summary}\n"
            f"{'-'*40}"
        )

        # Update the video context with the new summary.
        if not ctx:
            ctx = VideoContext(video_id=video_id, url=url)
        
        ctx.transcript = transcript
        ctx.classification = classification
        ctx.summary = s_result
        state.add_video_context(ctx)

        # Return the summary text.
        return ToolResult(
            message=msg, 
            success=True,
            data={"video_id": video_id, "url": url, "summary": s_result}
        )


class AnswerQuestionTool(Tool):
    '"Tool to answer specific questions about a video."'

    def __init__(
        self,
        transcript_service: TranscriptService,
        qa_service: QuestionAnsweringService,
    ) -> None:
        self._transcript_service = transcript_service
        self._qa_service = qa_service

    @property
    def name(self) -> str:
        return "answer_question"

    @property
    def description(self) -> str:
        return (
            "Answers specific questions about the content of a video. "
            "Use this when the user asks a specific question like 'what is X?' or 'how does Y work?'. "
            "Parameters: 'url' (optional), 'question' (required string)."
        )

    def run(self, state: WorkflowState, **kwargs) -> ToolResult:
        url = kwargs.get("url")
        question = kwargs.get("question")
        language = kwargs.get("language")

        if not question:
            return ToolResult(message="Error: 'question' parameter is required.", success=False)

        if not url:
            current_video = state.get_current_video()
            if current_video:
                url = current_video.url
            else:
                return ToolResult(
                    message="Error: 'url' parameter is missing and no active video found in context.",
                    success=False
                )

        try:
            video_id = self._transcript_service.parse_video_id(url)
        except ValueError:
            return ToolResult(
                message=f"Error: Could not parse a valid YouTube Video ID from url '{url}'.",
                success=False
            )

        # Ensure transcript is available.
        transcript = None
        ctx = state.videos.get(video_id.value)
        if ctx and ctx.transcript:
            transcript = ctx.transcript
        else:
            t_result = self._transcript_service.get_transcript(video_id, language=language)
            if not t_result.status.is_ok or not t_result.transcript:
                return ToolResult(
                    message=f"Cannot answer question: Failed to get transcript. Reason: {t_result.message or t_result.status.name}",
                    success=False
                )
            transcript = t_result.transcript
            # Update state with the transcript if it was just fetched.
            if not ctx:
                ctx = VideoContext(video_id=video_id, url=url)
            ctx.transcript = transcript
            state.add_video_context(ctx)

        # Generate the answer using the QA service.
        answer_text = self._qa_service.answer_question(transcript, question)
        
        msg = (
            f"Answer for '{transcript.title}':\n"
            f"{'-'*40}\n"
            f"{answer_text}\n"
            f"{'-'*40}"
        )

        # Return the answer.
        return ToolResult(
            message=msg, 
            success=True,
            data={"video_id": video_id, "url": url, "transcript": transcript}
        )


class ListHistoryTool(Tool):
    '"Tool to search through the user\'s video history."'

    def __init__(self, memory_service: MemoryService):
        self._memory = memory_service

    @property
    def name(self) -> str:
        return "list_video_history"

    @property
    def description(self) -> str:
        return (
            "Lists videos available in history. Use this to find IDs BEFORE "
            "comparing videos. Params: 'search_query' (optional), 'category' (optional)."
        )

    def run(self, state: WorkflowState, **kwargs) -> ToolResult:
        url = kwargs.get("url")
        search_query = kwargs.get("search_query")
        category = kwargs.get("category")

        # Retrieve all video metadata from memory.
        videos = asyncio.run(self._memory.list_video_metadata(state.session_id, state.user_id))
        
        results = []
        for v in videos:
            if category and v['category'].lower() != category.lower():
                continue
            if search_query and search_query.lower() not in v['title'].lower():
                continue
            results.append(f"- [ID: {v['video_id']}] {v['title']} ({v['category']})")
            
        if not results:
            return ToolResult(message="No videos found matching criteria.", success=True)
        
        limit = settings.SEARCH_RESULTS_LIMIT
        return ToolResult(
            message="Found Videos:\n" + "\n".join(results[:limit]), 
            success=True
        )


class CompareVideosTool(Tool):
    '"Tool to compare two or more videos based on a specific criteria."'

    def __init__(self, comparator: ComparisonService):
        self._comparator = comparator

    @property
    def name(self) -> str:
        return "compare_videos"

    @property
    def description(self) -> str:
        return (
            "Compares videos by ID. REQUIRED: Use list_video_history first to get IDs. "
            "Params: 'video_ids' (list of strings), 'question' (string)."
        )

    def run(self, state: WorkflowState, **kwargs) -> ToolResult:
        video_ids = kwargs.get("video_ids")
        question = kwargs.get("question")

        if not video_ids or not question:
            return ToolResult(
                message="Error: 'video_ids' and 'question' are required parameters.",
                success=False
            )

        if isinstance(video_ids, str):
            video_ids = [video_ids]

                       
                       
                       
                       
                       
                       
                       
                       
                       
        
        # Ensure inputs are valid before proceeding.
                       
        
        result_text = asyncio.run(
            self._comparator.compare_videos(state.session_id, state.user_id, video_ids, question)
        )
        return ToolResult(message=result_text, success=True)
