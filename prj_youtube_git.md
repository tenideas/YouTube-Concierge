This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: **/*.py, ./*.sh ./*.txt
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
app/
  agent.py
  tools.py
cli/
  main.py
config/
  prompts.py
  settings.py
infra/
  cache.py
  gemini_client.py
  llm_client.py
  youtube_transcript_provider.py
services/
  classifier.py
  comparator.py
  history.py
  memory.py
  qa_service.py
  summarizer.py
  transcript_service.py
tests/
  test_agent_logic.py
  test_comparator.py
  test_config_integrity.py
  test_tools.py
  test_url_parsing.py
check_models.py
domain.py
```

# Files

## File: app/tools.py
````python
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
````

## File: cli/main.py
````python
import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Import InMemorySessionService for local session management.
from google.adk.sessions import InMemorySessionService

from app.agent import Agent
from app.tools import (
    FetchTranscriptTool, 
    ClassifyVideoTool, 
    SummarizeVideoTool, 
    AnswerQuestionTool,
    ListHistoryTool,
    CompareVideosTool
)
from config import settings
from infra.cache import JsonFileCache
from infra.gemini_client import GeminiLlmClient
from infra.llm_client import LlmRateLimitError
from infra.youtube_transcript_provider import (
    TranscriptRateLimitError,
    YoutubeTranscriptProvider,
)
from services.classifier import Classifier
from services.comparator import ComparisonService
from services.history import HistoryManager
from services.memory import MemoryService
from services.qa_service import QuestionAnsweringService
from services.summarizer import SummarizationService
from services.transcript_service import TranscriptService, YtDlpMetadataProvider

logger = logging.getLogger(__name__)


def get_next_log_path(directory: Path, template: str, prefix: str) -> Path:
    '"Determines the next sequential log filename in the directory."'
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    
    max_num = 0
    # Define the glob pattern to find existing log files.
    search_pattern = f"{prefix}_*.log"
    
    # Iterate over existing files to find the highest sequence number.
    for path in directory.glob(search_pattern):
        if path.is_file():
                           
            stem = path.stem # Get the filename without extension.
            if stem.startswith(f"{prefix}_"):
                number_part = stem[len(f"{prefix}_"):]
                if number_part.isdigit():
                    num = int(number_part)
                    if num > max_num:
                        max_num = num
                    
    next_num = max_num + 1
    
    # Format the new filename with the incremented number.
    next_filename = template.format(next_num)
    
    return directory / next_filename


def setup_logging(verbose: bool) -> Path:
    '"Configures the logging system for file and console output."'
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set root logger to capture all debug events.

    # Clear existing handlers to prevent duplicate logs.
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Define a standard log format including timestamp.
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Calculate the path for the new log file.
    log_file_path = get_next_log_path(
        settings.LOG_DIR, 
        settings.LOG_FILENAME_TEMPLATE,
        settings.LOG_FILENAME_PREFIX
    )

    # Attempt to attach the file handler.
    try:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except IOError:
        # Fallback to stderr if file creation fails.
        print(f"Error: Could not open log file at {log_file_path}", file=sys.stderr)

    # Attach console handler, adjusting verbosity based on flags.
    console_handler = logging.StreamHandler(sys.stderr)
    console_level = logging.DEBUG if verbose else logging.WARNING
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return log_file_path


async def main(log_file_path: Path) -> None:
    '"Main entry point for the CLI application."'
    print("--- YouTube Summarizer Agent (AI Mode) ---")
    print("Initializing services...")

    try:
        # Initialize core infrastructure components.
        session_service = InMemorySessionService()
        
        transcript_provider = YoutubeTranscriptProvider()
        summary_cache = JsonFileCache(directory=settings.SUMMARYT_OUTPUT_DIR)
        
        # Initialize independent LLM clients for different agent roles.
        classifier_llm = GeminiLlmClient(
            max_retries=settings.GEMINI_MAX_RETRIES,
            safety_settings=settings.GEMINI_SAFETY_SETTINGS,
        )
        summarizer_llm = GeminiLlmClient(
            max_retries=settings.GEMINI_MAX_RETRIES,
            safety_settings=settings.GEMINI_SAFETY_SETTINGS,
        )
        qa_llm = GeminiLlmClient(
            max_retries=settings.GEMINI_MAX_RETRIES,
            safety_settings=settings.GEMINI_SAFETY_SETTINGS,
        )
        agent_llm = GeminiLlmClient(
            max_retries=settings.GEMINI_MAX_RETRIES,
            safety_settings=settings.GEMINI_SAFETY_SETTINGS,
        )
        history_llm = GeminiLlmClient(
            max_retries=settings.GEMINI_MAX_RETRIES,
            safety_settings=settings.GEMINI_SAFETY_SETTINGS,
        )

        # Initialize memory service wrapped around the session service.
        memory_service = MemoryService(
            session_service=session_service,
            app_name="youtube-agent"
        )
        
        # Initialize the comparison service.
        comparator_service = ComparisonService(
            llm_client=agent_llm,
            memory_service=memory_service
        )

        # Initialize the history manager for conversation context.
        history_manager = HistoryManager(
            memory_service=memory_service,
            llm_client=history_llm,
            model_name=settings.LLM_HISTORY_MODEL_NAME,
            max_tokens=settings.LLM_HISTORY_MAX_TOKENS,
            temperature=settings.LLM_HISTORY_TEMPERATURE,
        )

    except Exception as e:
        print(f"\nCRITICAL ERROR during initialization: {e}")
        sys.exit(1)

    # Initialize the Transcript Service with the YouTube provider.
    transcript_service = TranscriptService(
        output_dir=settings.TRANSCRIPT_OUTPUT_DIR,
        preferred_languages=settings.TRANSCRIPT_PREFERRED_LANGUAGES,
        provider=transcript_provider,
        metadata_provider=YtDlpMetadataProvider(),
    )

    classifier = Classifier(
        llm_client=classifier_llm,
        model_name=settings.LLM_CLASSIFIER_MODEL_NAME,
        max_tokens=settings.LLM_CLASSIFIER_MAX_TOKENS,
        temperature=settings.LLM_CLASSIFIER_TEMPERATURE,
    )

    summarizer = SummarizationService(
        llm_client=summarizer_llm,
        cache=summary_cache,
        model_name=settings.LLM_SUMMARY_MODEL_NAME,
        max_tokens=settings.LLM_SUMMARY_MAX_TOKENS,
        temperature=settings.LLM_SUMMARY_TEMPERATURE,
    )

    qa_service = QuestionAnsweringService(
        llm_client=qa_llm,
        model_name=settings.LLM_QA_MODEL_NAME,
        max_tokens=settings.LLM_QA_MAX_TOKENS,
        temperature=settings.LLM_QA_TEMPERATURE,
    )

    tools = [
        FetchTranscriptTool(transcript_service),
        ClassifyVideoTool(transcript_service, classifier),
        SummarizeVideoTool(transcript_service, classifier, summarizer),
        AnswerQuestionTool(transcript_service, qa_service),
        
        # Register all available tools for the agent.
        ListHistoryTool(memory_service),
        CompareVideosTool(comparator_service)
    ]

    # Instantiate the main Agent.
    agent = Agent(
        llm=agent_llm, 
        tools=tools,
        memory_service=memory_service,
        history_manager=history_manager,
        model_name=settings.LLM_AGENT_MODEL_NAME,
        max_tokens=settings.LLM_AGENT_MAX_TOKENS,
        temperature=settings.LLM_AGENT_TEMPERATURE,
    )

    session_id = str(uuid.uuid4())

    print("Initialization complete.")
    print(f"Logging to: {log_file_path.absolute()}")
    print("Enter a request (e.g. 'Summarize https://youtu.be/... in Russian'), or 'exit' to quit.")

    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nRequest> ")
            user_input = user_input.strip()
            
            # Log user input for debugging.
            logger.debug("User Input: %s", user_input)
            
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting.")
                break
            if not user_input:
                continue
            
            print(f"Thinking...")
            # Run the agent pipeline.
            response = await agent.run(
                user_input, 
                session_id=session_id,
                user_id=settings.DEFAULT_USER_ID
            )
            
            print("\n" + "-" * 60)
            print(response)
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        
        except TranscriptRateLimitError:
            print("\n!!! SYSTEM HALTED !!!")
            logger.critical("System halted due to YouTube Transcript IP Block.")
            sys.exit(1)
            
        except LlmRateLimitError:
            print("\n!!! SYSTEM HALTED !!!")
            logger.critical("System halted due to LLM Rate Limit/Quota.")
            sys.exit(1)

        except Exception as e:
            logger.exception("An unexpected error occurred during the loop.")
            print(f"Error: {e}")


if __name__ == "__main__":
    load_dotenv()
    
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="YouTube Summarizer Agent")
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose logging (DEBUG level) to console."
    )
    args = parser.parse_args()

    # Setup logging before starting the main loop.
    log_path = setup_logging(args.verbose)

    try:
        asyncio.run(main(log_path))
    except KeyboardInterrupt:
        print("\nExiting.")
````

## File: config/prompts.py
````python
from __future__ import annotations

from typing import Dict

from domain import VideoCategory

# -- Classifier Prompts --

CLASSIFIER_SYSTEM_INSTRUCTION: str = (
    "You are a precise video content classifier. Your job is to assign each "
    "YouTube video transcript to exactly one category from a fixed list of "
    "identifiers."
)

_CLASSIFIER_ALLOWED_LABELS: str = ", ".join(
    category.name for category in VideoCategory
)

CLASSIFIER_PROMPT_TEMPLATE: str = (
    "You will be given metadata and a transcript from a YouTube video.\n"
    "Your task is to classify this video into exactly one category from the "
    "following list of identifiers:\n"
    f"{_CLASSIFIER_ALLOWED_LABELS}\n\n"
    "Video Metadata:\n"
    "- Title: {title}\n"
    "- Channel/Author: {author}\n\n"
    "Transcript (Snippet):\n"
    "---------------------\n"
    "{transcript_text}\n"
    "---------------------\n"
    "Rules:\n"
    "1. Respond with only one identifier.\n"
    "2. Do not include any explanation or extra text.\n"
    "3. The identifier must match exactly one of the listed values.\n\n"
    "Answer with only the identifier:"
)

# -- Summarizer Prompts --

GENERIC_SUMMARY_INSTRUCTION: str = (
    "Write a clear, concise summary of the video that captures the main "
    "ideas, key points, and any important conclusions. Assume the reader has "
    "not watched the video."
)

UNCERTAIN_SUMMARY_INSTRUCTION: str = (
    "You do not know the precise category of this video. Write a general "
    "summary that covers the main topics, structure, and any key takeaways in "
    "a way that would help someone decide whether to watch it."
)

SUMMARY_PROMPT_TEMPLATE: str = (
    "{instruction}\n"
    "{language_instruction}\n\n"
    "Transcript:\n"
    "---------------------\n"
    "{transcript_text}\n"
    "---------------------\n"
    "Summary:"
)

VIDEO_SUMMARY_INSTRUCTIONS: Dict[VideoCategory, str] = {
    VideoCategory.VLOG: (
        "Summarize this vlog by describing the main events, locations, and "
        "emotional beats of the video. Highlight what happens in roughly the "
        "order it occurs and what makes the video interesting to watch."
    ),
    VideoCategory.EDUCATIONAL_EXPLAINER: (
        "Summarize this educational explainer by clearly stating the main "
        "topic, the core concepts that are taught, and any step-by-step "
        "reasoning or examples used to explain them."
    ),
    VideoCategory.LECTURE_PRESENTATION: (
        "Summarize this lecture or presentation by outlining the main thesis, "
        "the key sections, and the most important arguments or results "
        "covered by the speaker."
    ),
    VideoCategory.DOCUMENTARY: (
        "Summarize this documentary-style video by describing the central "
        "subject, the narrative arc, and the most important facts, events, or "
        "interviews presented."
    ),
    VideoCategory.HISTORY_EXPLAINER: (
        "Summarize this history-related video by stating the historical "
        "period or events it covers, the main storyline, and the key causes, "
        "consequences, or insights discussed."
    ),
    VideoCategory.SCIENCE_NEWS: (
        "Summarize this science-focused news or update by stating the main "
        "scientific topic, what is new or important, and any implications or "
        "limitations mentioned."
    ),
    VideoCategory.ECONOMICS_EXPLAINER: (
        "Summarize this economics-related explainer by identifying the "
        "economic concept or situation, the main arguments or models used, "
        "and any key examples or policy implications."
    ),
    VideoCategory.NEWS_REPORT: (
        "Summarize this news report by clearly stating what happened, where "
        "and when it occurred, who is involved, and any known causes or next "
        "steps mentioned."
    ),
    VideoCategory.COMEDY_SKETCH: (
        "Summarize this comedy or sketch by describing the basic premise, the "
        "main characters, and the most important beats of the joke or story "
        "without trying to recreate the humor."
    ),
    VideoCategory.INTERVIEW_CONVERSATION: (
        "Summarize this interview or conversation by stating who is speaking, "
        "the main topics discussed, and any notable opinions, stories, or "
        "insights they share."
    ),
    VideoCategory.MUSIC_ANALYSIS: (
        "Summarize this music-related analysis by describing what music or "
        "artist is being discussed, the main musical or lyrical points the "
        "speaker makes, and any conclusions they draw."
    ),
    VideoCategory.TRAVEL_GUIDE: (
        "Summarize this travel-related video by listing the main places "
        "visited, notable experiences or tips, and any practical advice that "
        "would help someone planning a similar trip."
    ),
    VideoCategory.OTHER: GENERIC_SUMMARY_INSTRUCTION,
}

COMPARISON_SYSTEM_INSTRUCTION: str = (
    "You are an expert comparative analyst. Your job is to objectively compare "
    "multiple video sources based on specific user criteria."
)

COMPARISON_PROMPT_TEMPLATE: str = (
    "User Question/Criteria: {question}\n\n"
    "--- Source Materials ---\n"
    "{content_context}\n"
    "------------------------\n\n"
    "Instructions:\n"
    "1. Compare the videos specifically addressing the user's question.\n"
    "2. Cite specific examples or topics from the transcripts where possible.\n"
    "3. If the videos are unrelated to the question, state that clearly.\n\n"
    "Comparison:"
)
````

## File: config/settings.py
````python
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Sequence

# -- Directory Paths --
BASE_DIR: Path = Path.cwd()
DATA_DIR: Path = BASE_DIR / "data"

# Data output directories.
TRANSCRIPT_OUTPUT_DIR: Path = DATA_DIR / "transcripts"
SUMMARYT_OUTPUT_DIR: Path = DATA_DIR / "summaries"
TRANSCRIPT_MAX_FILENAME_LENGTH: int = 128
TRANSCRIPT_PREFERRED_LANGUAGES: Sequence[str] = ("en", "en-US")
TRANSCRIPT_MIN_CHARS: int = 200

# Transcript Provider Retry Settings (Used for transient errors like API hiccups)
TRANSCRIPT_RETRY_COUNT: int = 3
TRANSCRIPT_RETRY_BASE_DELAY_SECONDS: float = 1.0
TRANSCRIPT_RETRY_BACKOFF_FACTOR: float = 2.0
TRANSCRIPT_RETRY_JITTER_SECONDS: float = 0.5

# Log output directory.
LOG_DIR: Path = BASE_DIR / 'logs'
# Prefix for log files.
LOG_FILENAME_PREFIX: str = "agent"
# Template for sequential log files.
LOG_FILENAME_TEMPLATE: str = LOG_FILENAME_PREFIX + "_{:04d}.log"

# -- Model Configurations --
# Classifier Model Settings: deterministic classification (zero temperature).
LLM_CLASSIFIER_MODEL_NAME: str = "gemini-2.5-flash-lite"
LLM_CLASSIFIER_MAX_TOKENS: int = 1024
LLM_CLASSIFIER_TEMPERATURE: float = 0.0

# Summarizer Model Settings: higher temperature for creative, concise summaries.
LLM_SUMMARY_MODEL_NAME: str = "gemini-2.5-flash-lite"                 
LLM_SUMMARY_MAX_TOKENS: int = 2048
LLM_SUMMARY_TEMPERATURE: float = 0.7

# QA Model Settings: low temperature for factual, grounded answers.
LLM_QA_MODEL_NAME: str = "gemini-2.5-flash-lite"
LLM_QA_MAX_TOKENS: int = 1024
LLM_QA_TEMPERATURE: float = 0.2

# Agent Model Settings: zero temperature for deterministic planning (JSON output).
LLM_AGENT_MODEL_NAME: str = "gemini-2.5-flash-lite"
LLM_AGENT_MAX_TOKENS: int = 512
LLM_AGENT_TEMPERATURE: float = 0.0

# -- API Configuration --
GEMINI_API_KEY_ENV_VAR: str = "GOOGLE_API_KEY"
GEMINI_MAX_RETRIES: int = 2
GEMINI_SAFETY_SETTINGS: Dict[str, Any] | None = None

# -- Session Configuration --
DEFAULT_USER_ID: str = "cli_user"

# -- Agent Limits --
AGENT_MAX_LOOPS: int = 5
AGENT_MAX_HISTORY_CHARS: int = 20000  # Limit history size to prevent context overflow.

# -- Tool Configuration --
SEARCH_RESULTS_LIMIT: int = 10
COMPARISON_CONTEXT_CHAR_LIMIT: int = 4000  # Character limit for comparison context to save tokens.

# Comparator Model Settings
LLM_COMPARATOR_MODEL_NAME: str = "gemini-2.5-flash-lite"
LLM_COMPARATOR_MAX_TOKENS: int = 1024
LLM_COMPARATOR_TEMPERATURE: float = 0.5

# History Management Model Settings: Used for summarizing chat history (deterministic, zero temperature).
LLM_HISTORY_MODEL_NAME: str = "gemini-2.5-flash-lite"
LLM_HISTORY_MAX_TOKENS: int = 512
LLM_HISTORY_TEMPERATURE: float = 0.0

# Max turns to keep in full history before compacting.
HISTORY_MAX_TURNS: int = 10
# Number of recent turns to preserve after compaction.
HISTORY_RECENT_TURNS_TO_KEEP: int = 4
````

## File: infra/cache.py
````python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class JsonFileCache:
    '"Simple file-based cache implementation using JSON."'

    def __init__(self, directory: Path) -> None:
        '# Ensure the cache directory exists on initialization.'
        self._directory = directory
        # Create directory if it doesn't exist.
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("Failed to create cache directory: %s", self._directory)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        '"Retrieves a dictionary from the cache by key, or None if missing."'
        path = self._get_path(key)
        if not path.is_file():
            return None

        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to read/parse cache file '%s'", path)
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        '"Saves a dictionary to the cache under the specified key."'
        path = self._get_path(key)
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
            logger.debug("Saved data to cache: %s", path)
        except OSError:
            logger.exception("Failed to write cache file '%s'", path)

    def _get_path(self, key: str) -> Path:
        '"Generates a safe filesystem path for a given cache key."'
        # Sanitize the key to be safe for filenames.
        safe_key = "".join(c for c in key if c.isalnum() or c in ("-", "_", "."))
        return self._directory / f"{safe_key}.json"
````

## File: infra/gemini_client.py
````python
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
````

## File: infra/llm_client.py
````python
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
````

## File: infra/youtube_transcript_provider.py
````python
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    IpBlocked, 
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from domain import VideoId 

logger = logging.getLogger(__name__)
               

class TranscriptProviderError(RuntimeError):
    '"Base exception for transcript provider errors."'


class TranscriptsDisabledError(TranscriptProviderError):
    '"Raised when transcripts are disabled for a video."'


class NoTranscriptError(TranscriptProviderError):
    '"Raised when no transcript exists for the video."'


class UnsupportedTranscriptLanguageError(TranscriptProviderError):
    '"Raised when the requested language is not available."'
    def __init__(
        self,
        message: str,
        available_languages: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(message)
        self.available_languages = list(available_languages or ())


class TranscriptRateLimitError(TranscriptProviderError):
    '"Raised when the provider is rate limited."'


class TranscriptTransientError(TranscriptProviderError):
    '"Raised for temporary errors that might succeed on retry."'


               

@dataclass(frozen=True)
class TranscriptSegment:
    '"A single segment of a transcript."'
    text: str
    start: float
    duration: float


@dataclass(frozen=True)
class FetchedTranscript:
    '"The raw result of a fetch operation."'
    video_id: VideoId
    language: str
    segments: List[TranscriptSegment]


               

class YoutubeTranscriptProvider:
    '"Provider implementation using youtube_transcript_api."'

    def __init__(self) -> None:
        self._api = YouTubeTranscriptApi()

    def fetch(
        self,
        video_id: VideoId,
        preferred_languages: Sequence[str],
        strict_match: bool = False,
    ) -> FetchedTranscript:
        '# Main fetch method implementation.'
        # Retrieve the list of available transcripts.
        transcripts = self._list_transcripts(video_id)

        # Select the best matching transcript object.
        transcript_obj = self._find_transcript_obj(
            video_id, transcripts, preferred_languages, strict_match
        )

        # Download the raw segment data.
        raw_segments = self._fetch_raw_segments(video_id, transcript_obj)
        
        # Convert raw data to domain objects.
        segments = self._convert_segments(raw_segments)
        language = self._resolve_language_code(transcript_obj)

        logger.info(
            "Successfully fetched transcript for '%s' (lang=%s, segments=%d)",
            video_id.value,
            language,
            len(segments),
        )

        return FetchedTranscript(
            video_id=video_id,
            language=language,
            segments=segments,
        )

    def _list_transcripts(self, video_id: VideoId):
        '# Helper to list transcripts handling API-specific exceptions.'
        try:
            logger.debug("Listing transcripts for video '%s'", video_id.value)
            return self._api.list(video_id.value)
            
        except TranscriptsDisabled as error:
            logger.info("Transcripts disabled for '%s'", video_id.value)
            raise TranscriptsDisabledError(
                f"Transcripts are disabled for video {video_id.value}."
            ) from error
            
        except NoTranscriptFound as error:
            logger.info("No transcripts found for '%s'", video_id.value)
            raise NoTranscriptError(
                f"No transcript exists for video {video_id.value}."
            ) from error
            
        except (IpBlocked, RequestBlocked) as error:
            logger.warning("Rate limited listing transcripts for '%s'", video_id.value)
            raise TranscriptRateLimitError("Rate limited while listing.") from error
            
        except Exception as error:
            logger.exception("Unexpected error listing transcripts '%s'", video_id.value)
            raise TranscriptProviderError("Unexpected error listing.") from error

    def _find_transcript_obj(
        self, 
        video_id: VideoId, 
        transcripts, 
        preferred_languages: Sequence[str],
        strict_match: bool,
    ):
        '# Attempt to find the requested language.'
        try:
            # Use the API's find_transcript method.
            return transcripts.find_transcript(list(preferred_languages))
            
        except NoTranscriptFound:
            available = self._extract_available_languages(transcripts)
            
            if strict_match:
                logger.info(
                    "Strict match failed for '%s'. Requested: %s, Available: %s",
                    video_id.value,
                    preferred_languages,
                    available
                )
                raise UnsupportedTranscriptLanguageError(
                    f"None of the requested languages {preferred_languages} were found.",
                    available_languages=available
                )

            logger.info(
                "Preferred languages %s not found for '%s'. Falling back to best available.",
                preferred_languages,
                video_id.value,
            )
            
            # If not strict match, fall back to the first available manual transcript.
            all_transcripts = list(transcripts)
            if not all_transcripts:
                raise NoTranscriptError(f"No transcripts found for {video_id.value}")

            # Prefer manually created captions over auto-generated ones.
            manuals = [t for t in all_transcripts if not t.is_generated]
            if manuals:
                best = manuals[0]
                logger.info(
                    "Fallback: Selected manual transcript '%s' for '%s'", 
                    best.language_code, 
                    video_id.value
                )
                return best
            
            # Fallback to auto-generated captions if no manual ones exist.
            best = all_transcripts[0]
            logger.info(
                "Fallback: Selected generated transcript '%s' for '%s'", 
                best.language_code, 
                video_id.value
            )
            return best
            
        except UnsupportedTranscriptLanguageError:
            raise
        except Exception as error:
            logger.exception("Error selecting transcript language for '%s'", video_id.value)
            raise TranscriptProviderError("Error selecting language.") from error

    def _fetch_raw_segments(self, video_id: VideoId, transcript_obj) -> List[dict]:
        '# Helper to fetch the actual text data from the transcript object.'
        try:
            logger.debug("Downloading segments for '%s'", video_id.value)
            fetched = transcript_obj.fetch()
            
            # Handle different return types from the API.
            if hasattr(fetched, "to_raw_data"):
                return fetched.to_raw_data()
            return fetched
        
        except (IpBlocked, RequestBlocked) as error:
            raise TranscriptRateLimitError("Rate limited during fetch.") from error
        except CouldNotRetrieveTranscript as error:
            raise TranscriptTransientError("Transient error during fetch.") from error
        except Exception as error:
            logger.exception("Unexpected error fetching content for '%s'", video_id.value)
            raise TranscriptProviderError("Unexpected error fetching content.") from error

                   
    @staticmethod
    def _convert_segments(segments: Iterable[dict]) -> List[TranscriptSegment]:
        '"Converts raw API dictionaries to TranscriptSegment objects."'
        converted: List[TranscriptSegment] = []
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            
            start = float(segment.get("start", 0.0))
            duration = float(segment.get("duration", 0.0))
            
            converted.append(
                TranscriptSegment(
                    text=text,
                    start=start,
                    duration=duration,
                )
            )
        return converted

    @staticmethod
    def _resolve_language_code(transcript_obj: object) -> str:
        '"Extracts the language code from a transcript object."'
        language_code = getattr(transcript_obj, "language_code", None)
        language = getattr(transcript_obj, "language", None)

        if isinstance(language_code, str) and language_code.strip():
            return language_code.strip()
        if isinstance(language, str) and language.strip():
            return language.strip()
        return "unknown"

    @staticmethod
    def _extract_available_languages(transcripts: Iterable) -> List[str]:
        '"Extracts a list of available language codes for error reporting."'
        languages: List[str] = []
                       
        for transcript in transcripts:
            code = getattr(transcript, "language_code", None)
            if code:
                languages.append(str(code))
        return sorted(set(languages))
````

## File: services/classifier.py
````python
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
````

## File: services/comparator.py
````python
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
````

## File: services/history.py
````python
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
````

## File: services/memory.py
````python
from __future__ import annotations

import logging
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from google.adk.sessions import Session, BaseSessionService
from google.adk.events import Event, EventActions

from domain import (
    ClassificationResult,
    ClassificationStatus,
    SummaryResult,
    Transcript,
    TranscriptSource,
    VideoCategory,
    VideoContext,
    VideoId,
)

logger = logging.getLogger(__name__)


VIDEOS_STATE_KEY = "videos"
CURRENT_VIDEO_ID_STATE_KEY = "current_video_id"
CHAT_HISTORY_STATE_KEY = "chat_history"


class MemoryService:
    '"Service to manage application state (videos, history) using the Session system."'

    def __init__(self, session_service: BaseSessionService, app_name: str) -> None:
        self._session_service = session_service
        self._app_name = app_name

    async def get_current_video_context(
        self,
        session_id: str,
        user_id: str,
    ) -> Optional[VideoContext]:
        '"Retrieves the VideoContext for the video marked as currently active in the session state."'
        session = await self._get_session(session_id, user_id)
        if not session:
            logger.warning("No session found for %s", session_id)
            return None

        current_vid = session.state.get(CURRENT_VIDEO_ID_STATE_KEY)
        logger.debug("Current video id in state: %s", current_vid)

        if not current_vid:
            return None

        return self._get_video_from_session(session, current_vid)

    async def get_video_context(
        self,
        session_id: str,
        video_id: VideoId,
        user_id: str,
    ) -> Optional[VideoContext]:
        '"Retrieves a specific VideoContext from memory using its VideoId."'
        session = await self._get_session(session_id, user_id)
        if not session:
            return None
        return self._get_video_from_session(session, video_id.value)

    async def save_video_context(
        self,
        session_id: str,
        context: VideoContext,
        user_id: str,
    ) -> None:
        '"Serializes and persists the VideoContext to the session state, marking it as the current active video."'
        session = await self._get_session(session_id, user_id)
        if not session:
            logger.error("Could not get session to save context.")
            return

        # Serialize context to dictionary.
        serialized = self._serialize_context(context)

        # Load existing videos map.
        existing_videos: Dict[str, Any] = dict(
            session.state.get(VIDEOS_STATE_KEY, {})
        )
        existing_videos[context.video_id.value] = serialized

        # Prepare state delta.
        state_changes: Dict[str, Any] = {
            VIDEOS_STATE_KEY: existing_videos,
            CURRENT_VIDEO_ID_STATE_KEY: context.video_id.value,
        }

        logger.debug(
            "Saving state for video '%s' in session '%s'",
            context.video_id.value,
            session_id,
        )

        # Update local session object.
        session.state[VIDEOS_STATE_KEY] = existing_videos
        session.state[CURRENT_VIDEO_ID_STATE_KEY] = context.video_id.value

        # Persist changes using the session service event system.
        try:
            actions_with_update = EventActions(state_delta=state_changes)
            system_event = Event(
                invocation_id=session_id,
                author="system",
                actions=actions_with_update,
                timestamp=time.time(),
            )
            await self._session_service.append_event(session, system_event)
            logger.debug(
                "Persisted context for video '%s' in session '%s'",
                context.video_id.value,
                session_id,
            )
        except Exception:
            logger.exception("Failed to persist session storage.")

    async def get_chat_history(
        self,
        session_id: str,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        '"Retrieves the chat history list from the session state."'
        session = await self._get_session(session_id, user_id)
        if not session:
            return []

        return list(session.state.get(CHAT_HISTORY_STATE_KEY, []))

    async def save_chat_history(
        self,
        session_id: str,
        history: List[Dict[str, Any]],
        user_id: str,
    ) -> None:
        '"Saves the entire chat history list to the session state."'
        session = await self._get_session(session_id, user_id)
        if not session:
            logger.error("Could not get session to save chat history.")
            return

        state_changes: Dict[str, Any] = {
            CHAT_HISTORY_STATE_KEY: history,
        }

        # Update local state object.
        session.state[CHAT_HISTORY_STATE_KEY] = history

        try:
            actions = EventActions(state_delta=state_changes)
            event = Event(
                invocation_id=session_id,
                author="system",
                actions=actions,
                timestamp=time.time(),
            )
            await self._session_service.append_event(session, event)
            logger.debug("Persisted chat history for session '%s'", session_id)
        except Exception:
            logger.exception("Failed to persist chat history.")

    async def get_all_videos(
        self,
        session_id: str,
        user_id: str,
    ) -> List[VideoContext]:
        '"Retrieves and deserializes all stored VideoContext objects for a session."'
        session = await self._get_session(session_id, user_id)
        if not session or VIDEOS_STATE_KEY not in session.state:
            return []

        results: List[VideoContext] = []
        for vid_str in session.state[VIDEOS_STATE_KEY]:
            ctx = self._get_video_from_session(session, vid_str)
            if ctx:
                results.append(ctx)

        return results

    async def list_video_metadata(
        self, 
        session_id: str, 
        user_id: str
    ) -> List[Dict[str, Any]]:
        '"Extracts minimal metadata (ID, title, category) for all stored videos, suitable for listing/comparison." '
        session = await self._get_session(session_id, user_id)
        if not session or VIDEOS_STATE_KEY not in session.state:
            return []

        results = []
        raw_videos = session.state[VIDEOS_STATE_KEY]

        for vid_id, data in raw_videos.items():
            # Handle missing optional fields safely.
            transcript_data = data.get("transcript") or {}
            class_data = data.get("classification") or {}
            
            results.append({
                "video_id": vid_id,
                "title": transcript_data.get("title", "Unknown"),
                "category": class_data.get("category", "Unclassified"),
                "created_at": data.get("created_at")
            })
        
        return results

    async def _get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> Optional[Session]:
        '"Retrieves an existing session or creates a new one if it does not exist."'
        try:
            session = await self._session_service.get_session(
                session_id=session_id,
                app_name=self._app_name,
                user_id=user_id,
            )
            if session:
                return session

            logger.info("Creating new session for ID: %s", session_id)
            return await self._session_service.create_session(
                session_id=session_id,
                app_name=self._app_name,
                user_id=user_id,
            )
        except Exception:
            logger.exception("Failed to retrieve or create session.")
            return None

    def _get_video_from_session(
        self,
        session: Session,
        vid_value: str,
    ) -> Optional[VideoContext]:
        '"Retrieves raw video data from session state and deserializes it into a VideoContext object."'
        videos = session.state.get(VIDEOS_STATE_KEY, {})
        data = videos.get(vid_value)
        if not data:
            return None

        try:
            return self._deserialize_context(data)
        except Exception:
            logger.exception("Failed to deserialize context for video '%s'", vid_value)
            return None

    def _serialize_context(self, context: VideoContext) -> Dict[str, Any]:
        '"Converts VideoContext to a dictionary."'
        data = asdict(context)
        return self._sanitize_for_storage(data)

    def _sanitize_for_storage(self, obj: Any) -> Any:
        '"Recursively sanitizes objects (Dataclasses, Enums, Datetimes) within a dictionary or list for safe JSON storage by converting them to primitives like strings (for enums and datetime) or dicts/lists."'
        if isinstance(obj, dict):
            return {k: self._sanitize_for_storage(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize_for_storage(v) for v in obj]
        if isinstance(obj, datetime):
            return obj.isoformat()
        if is_dataclass(obj):
            return self._sanitize_for_storage(asdict(obj))
        if hasattr(obj, "name"):
            # Convert Enum to string name.
            return obj.name
        return obj

    def _deserialize_context(self, data: Dict[str, Any]) -> VideoContext:
        '"Reconstructs a VideoContext object from a dictionary by safely converting stored primitives (like enum names and ISO strings) back into their corresponding domain objects (VideoId, Transcript, ClassificationResult, etc.)."'
        # Handle video ID reconstruction.
        if "video_id" in data and isinstance(data["video_id"], dict):
            video_id = VideoId(data["video_id"]["value"])
        else:
            video_id = VideoId(data.get("video_id_value") or data["video_id"])

        # Reconstruct Transcript object.
        transcript: Optional[Transcript] = None
        t_data = data.get("transcript")
        if t_data:
            transcript = Transcript(
                video_id=video_id,
                title=t_data["title"],
                author=t_data["author"],
                language=t_data["language"],
                text=t_data["text"],
                source=TranscriptSource[t_data["source"]],
                created_at=datetime.fromisoformat(t_data["created_at"]),
            )

        # Reconstruct ClassificationResult object.
        classification: Optional[ClassificationResult] = None
        c_data = data.get("classification")
        if c_data:
            category: Optional[VideoCategory] = None
            if c_data.get("category"):
                category = VideoCategory[c_data["category"]]

            classification = ClassificationResult(
                status=ClassificationStatus[c_data["status"]],
                category=category,
                reason=c_data.get("reason"),
                raw_model_output=c_data.get("raw_model_output"),
            )

        # Reconstruct SummaryResult object.
        summary: Optional[SummaryResult] = None
        s_data = data.get("summary")
        if s_data:
            used_cat: Optional[VideoCategory] = None
            if s_data.get("used_category"):
                used_cat = VideoCategory[s_data["used_category"]]

            summary = SummaryResult(
                status=ClassificationStatus[s_data["status"]],
                summary=s_data.get("summary"),
                used_category=used_cat,
                instruction=s_data.get("instruction"),
                message=s_data.get("message"),
            )

        # Parse timestamps.
        created_at_str = data.get("created_at")
        last_accessed_str = data.get("last_accessed")

        created_at = (
            datetime.fromisoformat(created_at_str) if created_at_str else datetime.now()
        )
        last_accessed = (
            datetime.fromisoformat(last_accessed_str)
            if last_accessed_str
            else datetime.now()
        )

        return VideoContext(
            video_id=video_id,
            url=data["url"],
            created_at=created_at,
            last_accessed=last_accessed,
            transcript=transcript,
            classification=classification,
            summary=summary,
        )
````

## File: services/qa_service.py
````python
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
````

## File: services/summarizer.py
````python
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
````

## File: services/transcript_service.py
````python
from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence
from urllib.parse import parse_qs, urlparse

import yt_dlp

from config.settings import (
    TRANSCRIPT_MAX_FILENAME_LENGTH,
    TRANSCRIPT_MIN_CHARS,
    TRANSCRIPT_OUTPUT_DIR,
    TRANSCRIPT_PREFERRED_LANGUAGES,
    TRANSCRIPT_RETRY_BACKOFF_FACTOR,
    TRANSCRIPT_RETRY_BASE_DELAY_SECONDS,
    TRANSCRIPT_RETRY_COUNT,
    TRANSCRIPT_RETRY_JITTER_SECONDS,
)
from domain import (
    ClassificationStatus,
    Transcript,
    TranscriptResult,
    TranscriptSource,
    VideoId,
)
from infra.youtube_transcript_provider import (
    FetchedTranscript,
    NoTranscriptError,
    TranscriptProviderError,
    TranscriptRateLimitError,
    TranscriptSegment,
    TranscriptTransientError,
    TranscriptsDisabledError,
    UnsupportedTranscriptLanguageError,
    YoutubeTranscriptProvider,
)

logger = logging.getLogger(__name__)

UNKNOWN_TITLE_TEMPLATE = "Video {video_id}"
UNKNOWN_AUTHOR = "Unknown"

_YT_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{11}")
_YT_HOST_SUFFIXES = (
    "youtu.be",
    "youtube.com",
)


@dataclass(frozen=True)
class VideoMetadata:
    '"Data structure for video metadata (title, author)."'
    title: str
    author: str


class VideoMetadataProvider(Protocol):
    '"Protocol for fetching video metadata."'

    def get_metadata(self, video_id: VideoId) -> VideoMetadata:
        '"Retrieves metadata for a given video ID."'


@dataclass
class DefaultVideoMetadataProvider:
    '"Fallback provider returning generic metadata."'

    def get_metadata(self, video_id: VideoId) -> VideoMetadata:
        return VideoMetadata(
            title=UNKNOWN_TITLE_TEMPLATE.format(video_id=video_id.value),
            author=UNKNOWN_AUTHOR,
        )


class YtDlpMetadataProvider:
    '"Provider using yt-dlp to fetch actual video metadata."'

    def get_metadata(self, video_id: VideoId) -> VideoMetadata:
        '# Fetch metadata using yt-dlp.'
        # Construct URL.
        url = f"https://www.youtube.com/watch?v={video_id.value}"
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            # extract_flat helps avoid full processing/downloading.
            "extract_flat": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False) or {}
                
                title = info.get("title") or UNKNOWN_TITLE_TEMPLATE.format(
                    video_id=video_id.value
                )
                
                # Extract fields with fallbacks.
                author = info.get("uploader") or info.get("channel") or UNKNOWN_AUTHOR
                
                return VideoMetadata(title=title, author=author)

        except Exception:
            logger.warning(
                "Failed to fetch metadata for '%s' via yt-dlp; falling back to generic values.",
                video_id.value,
                exc_info=True,
            )
            return VideoMetadata(
                title=UNKNOWN_TITLE_TEMPLATE.format(video_id=video_id.value),
                author=UNKNOWN_AUTHOR,
            )


class TranscriptService:
    '"Service to manage transcript fetching, caching, and storage."'

    def __init__(
        self,
        output_dir: Path = TRANSCRIPT_OUTPUT_DIR,
        preferred_languages: Sequence[str] = TRANSCRIPT_PREFERRED_LANGUAGES,
        max_filename_length: int = TRANSCRIPT_MAX_FILENAME_LENGTH,
        min_transcript_chars: int = TRANSCRIPT_MIN_CHARS,
        retry_count: int = TRANSCRIPT_RETRY_COUNT,
        retry_base_delay_seconds: float = TRANSCRIPT_RETRY_BASE_DELAY_SECONDS,
        retry_backoff_factor: float = TRANSCRIPT_RETRY_BACKOFF_FACTOR,
        retry_jitter_seconds: float = TRANSCRIPT_RETRY_JITTER_SECONDS,
        metadata_provider: Optional[VideoMetadataProvider] = None,
        provider: Optional[YoutubeTranscriptProvider] = None,
    ) -> None:
        self._output_dir = output_dir
        self._preferred_languages = tuple(preferred_languages)
        self._max_filename_length = max_filename_length
        self._min_transcript_chars = min_transcript_chars
        self._retry_count = retry_count
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._retry_backoff_factor = retry_backoff_factor
        self._retry_jitter_seconds = retry_jitter_seconds
        self._metadata_provider = metadata_provider or DefaultVideoMetadataProvider()
        
        self._provider = provider or YoutubeTranscriptProvider()

        # Create output directory.
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def parse_video_id(url_or_id: str) -> VideoId:
        '"Parses a YouTube Video ID from various URL formats."'
        candidate = url_or_id.strip()
        if TranscriptService._looks_like_video_id(candidate):
            return VideoId(candidate)

        parsed = urlparse(candidate)
        if not parsed.netloc:
            raise ValueError(f"Could not extract video ID from '{url_or_id}'")

        host = parsed.netloc.lower()
        path = parsed.path or ""
        query = parse_qs(parsed.query)

        if any(host.endswith(suffix) for suffix in _YT_HOST_SUFFIXES):
            if "youtu.be" in host:
                vid = TranscriptService._extract_from_short_url(path)
                if vid is not None:
                    return VideoId(vid)

            vid = TranscriptService._extract_from_watch_query(query)
            if vid is not None:
                return VideoId(vid)

            vid = TranscriptService._extract_from_path(path)
            if vid is not None:
                return VideoId(vid)

        vid = TranscriptService._extract_from_text(
            f"{parsed.path}?{parsed.query}",
        )
        if vid is not None:
            return VideoId(vid)

        raise ValueError(f"Could not extract video ID from '{url_or_id}'")

    def get_transcript(
        self, 
        video_id: VideoId, 
        language: Optional[str] = None
    ) -> TranscriptResult:
        '"Checks cache for an existing transcript, and fetches remotely if not found, handling retries."'
        cached = self._load_cached_transcript(video_id, language)
        if cached is not None:
            return TranscriptResult(
                status=ClassificationStatus.OK,
                transcript=cached,
                message=None,
            )

        return self._fetch_remote_transcript_with_retries(video_id, language)

    @staticmethod
    def _looks_like_video_id(value: str) -> bool:
        return bool(_YT_ID_PATTERN.fullmatch(value))

    @staticmethod
    def _extract_from_short_url(path: str) -> Optional[str]:
        '"Extracts ID from short URLs (youtu.be)."'
        parts = [p for p in path.split("/") if p]
        if not parts:
            return None
        candidate = parts[0]
        if TranscriptService._looks_like_video_id(candidate):
            return candidate
        return None

    @staticmethod
    def _extract_from_watch_query(query: dict[str, list[str]]) -> Optional[str]:
        '"Extracts ID from query parameters (v=...)."'
        values = query.get("v")
        if not values:
            return None
        candidate = values[0]
        if TranscriptService._looks_like_video_id(candidate):
            return candidate
        return None

    @staticmethod
    def _extract_from_path(path: str) -> Optional[str]:
        '"Extracts ID from path segments (embed, shorts, etc)."'
        parts = [p for p in path.split("/") if p]
        if not parts:
            return None
        markers = {"embed", "shorts", "live"}
        for index, part in enumerate(parts[:-1]):
            if part in markers:
                candidate = parts[index + 1]
                if TranscriptService._looks_like_video_id(candidate):
                    return candidate
        last = parts[-1]
        if TranscriptService._looks_like_video_id(last):
            return last
        return None

    @staticmethod
    def _extract_from_text(text: str) -> Optional[str]:
        '"Scans arbitrary text for a YouTube ID pattern."'
        match = _YT_ID_PATTERN.search(text)
        if match is None:
            return None
        return match.group(0)

    def _get_cache_path(self, video_id: VideoId, language: str) -> Path:
        '"Generates the filesystem path for a cached transcript, ensuring the filename does not exceed the configured maximum length."'
        clean_lang = language.strip().replace("-", "_")
        filename = f"{video_id.value}_{clean_lang}.json"
        
        if len(filename) > self._max_filename_length:
            # Truncate filename if it exceeds filesystem limits.
            ext = ".json"
            base = filename[:-len(ext)]
            filename = base[: self._max_filename_length - len(ext)] + ext
            
        return self._output_dir / filename

    def _load_cached_transcript(
        self, 
        video_id: VideoId, 
        language: Optional[str] = None
    ) -> Optional[Transcript]:
        '"Tries to load a cached transcript, checking the specific requested language first, then the preferred languages list."'
        # If specific language requested.
        if language:
            path = self._get_cache_path(video_id, language)
            return self._read_transcript_file(path, video_id)

        # If no specific language, check preferred list.
        for pref_lang in self._preferred_languages:
            path = self._get_cache_path(video_id, pref_lang)
            transcript = self._read_transcript_file(path, video_id)
            if transcript:
                logger.info(
                    "Cache hit for '%s' using preferred language '%s'", 
                    video_id.value, 
                    pref_lang
                )
                return transcript
        
        return None

    def _read_transcript_file(self, path: Path, video_id: VideoId) -> Optional[Transcript]:
        '"Reads and parses the transcript JSON file."'
        if not path.is_file():
            return None

        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            logger.exception(
                "Failed to read cached transcript from '%s'",
                path,
            )
            return None

        text = str(data.get("text", "")).strip()
        if not text:
            return None

        title = str(
            data.get("title")
            or UNKNOWN_TITLE_TEMPLATE.format(video_id=video_id.value),
        )
        author = str(data.get("author") or UNKNOWN_AUTHOR)
        language = str(data.get("language") or "unknown")
        
        created_at_str = data.get("created_at")
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except ValueError:
                created_at = datetime.now(timezone.utc)
        else:
            created_at = datetime.now(timezone.utc)

        return Transcript(
            video_id=video_id,
            title=title,
            author=author,
            language=language,
            text=text,
            source=TranscriptSource.CACHE,
            created_at=created_at,
        )

    def _save_transcript_to_cache(self, transcript: Transcript) -> None:
        '"Writes the transcript object to a JSON file."'
        path = self._get_cache_path(transcript.video_id, transcript.language)
        payload = {
            "video_id": transcript.video_id.value,
            "title": transcript.title,
            "author": transcript.author,
            "language": transcript.language,
            "text": transcript.text,
            "source": transcript.source.name,
            "created_at": transcript.created_at.isoformat(),
        }
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            logger.debug("Saved transcript to cache: %s", path)
        except OSError:
            logger.exception(
                "Failed to write transcript cache file '%s'",
                path,
            )

    def _fetch_remote_transcript_with_retries(
        self,
        video_id: VideoId,
        language: Optional[str] = None,
    ) -> TranscriptResult:
        last_result: Optional[TranscriptResult] = None
        for attempt in range(self._retry_count):
            result = self._fetch_remote_transcript_once(video_id, language)
            
            # Return immediately if the result is final (success or definite failure).
            if result.status in (
                ClassificationStatus.OK, 
                ClassificationStatus.UNSUPPORTED_LANGUAGE,
                ClassificationStatus.TRANSCRIPTS_DISABLED,
                ClassificationStatus.NO_TRANSCRIPT
            ):
                return result

            last_result = result
            if attempt >= self._retry_count - 1:
                break

            delay = self._retry_base_delay_seconds * (
                self._retry_backoff_factor ** attempt
            )
            jitter = random.uniform(
                -self._retry_jitter_seconds,
                self._retry_jitter_seconds,
            )
            sleep_seconds = max(0.0, delay + jitter)
            logger.warning(
                "Transient error fetching transcript for '%s', "
                "retrying in %.2f seconds (attempt %d/%d)",
                video_id.value,
                sleep_seconds,
                attempt + 1,
                self._retry_count,
            )
            time.sleep(sleep_seconds)

        if last_result is not None:
            return last_result

        return TranscriptResult(
            status=ClassificationStatus.UNKNOWN_ERROR,
            transcript=None,
            message="Failed to obtain transcript due to repeated errors.",
        )

    def _fetch_remote_transcript_once(
        self,
        video_id: VideoId,
        language: Optional[str] = None,
    ) -> TranscriptResult:
        try:
            # Determine which languages to request.
            strict_match = language is not None
            languages_to_fetch = [language] if language else self._preferred_languages

            raw: FetchedTranscript = self._provider.fetch(
                video_id=video_id,
                preferred_languages=languages_to_fetch,
                strict_match=strict_match,
            )
        except TranscriptsDisabledError as error:
            logger.info("Transcripts are disabled for video '%s'", video_id.value)
            return TranscriptResult(
                status=ClassificationStatus.TRANSCRIPTS_DISABLED,
                transcript=None,
                message=str(error),
            )
        except NoTranscriptError as error:
            logger.info("No transcripts available for video '%s'", video_id.value)
            return TranscriptResult(
                status=ClassificationStatus.NO_TRANSCRIPT,
                transcript=None,
                message=str(error),
            )
        except UnsupportedTranscriptLanguageError as error:
            logger.info("Requested language not found for '%s'", video_id.value)
            
            # Format error message with available languages.
            avail_str = ", ".join(error.available_languages)
            msg = f"Language '{language}' not found. Available languages: {avail_str}"
            
            return TranscriptResult(
                status=ClassificationStatus.UNSUPPORTED_LANGUAGE,
                transcript=None,
                message=msg,
            )
        
        except TranscriptTransientError as error:
            logger.warning(
                "Transient transcript provider error for '%s': %s",
                video_id.value,
                error,
            )
            return TranscriptResult(
                status=ClassificationStatus.UNKNOWN_ERROR,
                transcript=None,
                message=str(error),
            )
        except TranscriptProviderError as error:
            logger.exception(
                "Unexpected transcript provider error for '%s': %s",
                video_id.value,
                error,
            )
            return TranscriptResult(
                status=ClassificationStatus.UNKNOWN_ERROR,
                transcript=None,
                message=str(error),
            )

        text = self._combine_segments(raw.segments)
        if len(text.strip()) < self._min_transcript_chars:
            logger.info(
                "Transcript for '%s' is too short (%d chars)",
                video_id.value,
                len(text.strip()),
            )
            return TranscriptResult(
                status=ClassificationStatus.TRANSCRIPT_TOO_SHORT,
                transcript=None,
                message="Transcript is too short to be useful.",
            )

        language_found = raw.language or "unknown"
        metadata = self._metadata_provider.get_metadata(video_id)
        
        transcript = Transcript(
            video_id=video_id,
            title=metadata.title,
            author=metadata.author,
            language=str(language_found),
            text=text,
            source=TranscriptSource.YOUTUBE_API,
            created_at=datetime.now(timezone.utc),
        )
        self._save_transcript_to_cache(transcript)

        logger.info(
            "Successfully fetched transcript for '%s' (%d chars, lang=%s)",
            video_id.value,
            len(text),
            language_found,
        )
        return TranscriptResult(
            status=ClassificationStatus.OK,
            transcript=transcript,
            message=None,
        )

    @staticmethod
    def _combine_segments(segments: Iterable[TranscriptSegment]) -> str:
        parts: list[str] = []
        for segment in segments:
            text = str(segment.text).strip()
            if text:
                parts.append(text)
        return " ".join(parts)
````

## File: tests/test_comparator.py
````python
import unittest
from unittest.mock import AsyncMock, MagicMock
from domain import VideoContext, VideoId, Transcript
from services.comparator import ComparisonService

class TestComparator(unittest.IsolatedAsyncioTestCase):
    '"Tests the logic of the Comparison Service."'
    async def asyncSetUp(self):
        # Setup mocks for LLM and Memory.
        self.mock_llm = MagicMock()
        self.mock_memory = AsyncMock()
        self.comparator = ComparisonService(self.mock_llm, self.mock_memory)

    async def test_context_stuffing(self):
        '"Tests that video content is correctly injected into the prompt."'
                       
        # Create two mock video contexts.
        v1 = VideoContext(
            video_id=VideoId("111"), 
            url="url1", 
            transcript=Transcript(
                video_id=VideoId("111"), title="Python 101", author="Me", 
                language="en", text="Python is great.", source=None, created_at=None
            )
        )
        v2 = VideoContext(
            video_id=VideoId("222"), 
            url="url2", 
            transcript=Transcript(
                video_id=VideoId("222"), title="Java 101", author="You", 
                language="en", text="Java is verbose.", source=None, created_at=None
            )
        )
        
                       
        async def get_ctx_side_effect(sess, vid, uid):
            '# Side effect to return specific videos by ID.'
            if vid.value == "111": return v1
            if vid.value == "222": return v2
            return None
            
        self.mock_memory.get_video_context.side_effect = get_ctx_side_effect
        
                       
        self.mock_llm.generate_async = AsyncMock()
        self.mock_llm.generate_async.return_value.text = "Comparison Done"

        # Call compare.
        await self.comparator.compare_videos("sess", "user", ["111", "222"], "Diff?")

                       
        # Inspect the prompt sent to the LLM.
        call_args = self.mock_llm.generate_async.call_args
        request_obj = call_args[0][0]
        prompt = request_obj.prompt
        
        self.assertIn("Python 101", prompt)
        self.assertIn("Java 101", prompt)
        self.assertIn("Python is great", prompt)
        self.assertIn("Java is verbose", prompt)

    async def test_missing_video_handling(self):
        '"Tests graceful handling when a video ID is not found."'
        self.mock_memory.get_video_context.return_value = None
        result = await self.comparator.compare_videos("sess", "user", ["999"], "?")
        self.assertIn("Error", result)
        self.assertIn("Could not find", result)
````

## File: tests/test_config_integrity.py
````python
import unittest
from domain import VideoCategory
from config.prompts import VIDEO_SUMMARY_INSTRUCTIONS

class TestConfigIntegrity(unittest.TestCase):
    '"Tests configuration consistency."'

    def test_all_categories_have_summary_instructions(self):
        '"Verifies that every VideoCategory has a corresponding summary prompt."'
        domain_categories = set(VideoCategory)
        configured_categories = set(VIDEO_SUMMARY_INSTRUCTIONS.keys())
        
                       
        missing = domain_categories - configured_categories
        
                       
                       
        # Fail if there are missing instructions.
        self.assertFalse(
            missing, 
            f"The following categories are missing summary instructions in config/prompts.py: {missing}"
        )

if __name__ == "__main__":
    unittest.main()
````

## File: tests/test_tools.py
````python
import unittest
from unittest.mock import AsyncMock, create_autospec, MagicMock
from app.tools import ListHistoryTool, CompareVideosTool
from services.memory import MemoryService
from services.comparator import ComparisonService

class TestListHistoryTool(unittest.TestCase):
    '"Tests for the History listing tool."'
    def setUp(self):
        self.mock_memory = AsyncMock()
        self.tool = ListHistoryTool(self.mock_memory)
        
        # Mock data.
        self.fake_data = [
            {"video_id": "1", "title": "Python Intro", "category": "Tutorial", "created_at": ""},
            {"video_id": "2", "title": "Vlog about Cats", "category": "VLOG", "created_at": ""},
        ]
        self.mock_memory.list_video_metadata.return_value = self.fake_data

    def test_filter_by_category_case_insensitive(self):
        '"Tests filtering history by category."'
        result = self.tool.run(state=MagicMock(), search_query=None, category="vlog")
        
        self.assertIn("Vlog about Cats", result.message)
        self.assertNotIn("Python Intro", result.message)

    def test_filter_by_search_query(self):
        '"Tests filtering history by search query."'
        result = self.tool.run(state=MagicMock(), search_query="Python", category=None)
        self.assertIn("Python Intro", result.message)
        self.assertNotIn("Cats", result.message)


class TestCompareVideosTool(unittest.TestCase):
    '"Tests for the Compare Video tool."'
    def setUp(self):
        self.mock_comparator = create_autospec(ComparisonService, instance=True)
        self.tool = CompareVideosTool(self.mock_comparator)

    def test_single_string_input_normalization(self):
        '"Tests that a single string ID is converted to a list."'
        self.mock_comparator.compare_videos = AsyncMock(return_value="Done")
        
        # Note: Added 'state' mock to align with run signature.
        self.tool.run(state=MagicMock(), video_ids="123", question="Why?")
        
        # Verify it was called with a list.
        self.mock_comparator.compare_videos.assert_called_with(
            unittest.mock.ANY, unittest.mock.ANY, ["123"], "Why?"
        )
````

## File: tests/test_url_parsing.py
````python
import unittest
from services.transcript_service import TranscriptService
from domain import VideoId

class TestUrlParsing(unittest.TestCase):
    '"Tests the regex and parsing logic for YouTube URLs."'

    def test_standard_watch_url(self):
        '"Tests standard watch URLs."'
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        vid = TranscriptService.parse_video_id(url)
        self.assertEqual(vid.value, "dQw4w9WgXcQ")

    def test_shortened_url(self):
        '"Tests shortened URLs."'
        url = "https://youtu.be/dQw4w9WgXcQ"
        vid = TranscriptService.parse_video_id(url)
        self.assertEqual(vid.value, "dQw4w9WgXcQ")

    def test_shorts_url(self):
        '"Tests Shorts URLs."'
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        vid = TranscriptService.parse_video_id(url)
        self.assertEqual(vid.value, "dQw4w9WgXcQ")

    def test_live_url(self):
        '"Tests Live URLs."'
        url = "https://www.youtube.com/live/dQw4w9WgXcQ?feature=share"
        vid = TranscriptService.parse_video_id(url)
        self.assertEqual(vid.value, "dQw4w9WgXcQ")

    def test_embed_url(self):
        '"Tests Embed URLs."'
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        vid = TranscriptService.parse_video_id(url)
        self.assertEqual(vid.value, "dQw4w9WgXcQ")

    def test_raw_id(self):
        '"Tests raw ID strings."'
        raw = "dQw4w9WgXcQ"
        vid = TranscriptService.parse_video_id(raw)
        self.assertEqual(vid.value, "dQw4w9WgXcQ")

    def test_url_with_extra_params(self):
        '"Tests URLs with extra query parameters."'
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&ab_channel=RickAstley"
        vid = TranscriptService.parse_video_id(url)
        self.assertEqual(vid.value, "dQw4w9WgXcQ")

    def test_invalid_url_raises_value_error(self):
        '"Tests that invalid URLs raise ValueError."'
        invalid_urls = [
            "https://google.com",
            "not_a_video_id",
            "https://www.youtube.com/playlist?list=PL12345",
            "",
        ]
        for url in invalid_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    TranscriptService.parse_video_id(url)

if __name__ == "__main__":
    unittest.main()
````

## File: check_models.py
````python
from __future__ import annotations

import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables.
load_dotenv()

def main():
    '# Script entry point.'
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        sys.exit(1)

    genai.configure(api_key=api_key)

    print("Fetching available models...\n")
    
    try:
        # List models from the API.
        for model in genai.list_models():
            # Filter for models that support content generation.
            if "generateContent" in model.supported_generation_methods:
                print(f"Name: {model.name}")
                print(f"  Display Name: {model.display_name}")
                print(f"  Input Token Limit: {model.input_token_limit}")
                print(f"  Output Token Limit: {model.output_token_limit}")
                print("-" * 40)
                
    except Exception as e:
        print(f"Failed to list models: {e}")

if __name__ == "__main__":
    main()
````

## File: domain.py
````python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class ClassificationStatus(Enum):
    '"Enum representing the status of a classification or fetch operation."'

    OK = auto()
    NO_TRANSCRIPT = auto()
    TRANSCRIPTS_DISABLED = auto()
    TRANSCRIPT_TOO_SHORT = auto()
    UNSUPPORTED_LANGUAGE = auto()
    CLASSIFIER_ERROR = auto()
    UNKNOWN_ERROR = auto()
    UNCERTAIN = auto()

    @property
    def is_ok(self) -> bool:
        '"Returns True if the status indicates success."'
        return self is ClassificationStatus.OK

    @property
    def is_error(self) -> bool:
        '"Returns True if the status indicates a failure."'
        return self not in (
            ClassificationStatus.OK,
            ClassificationStatus.UNCERTAIN,
        )

    @property
    def is_uncertain(self) -> bool:
        '"Returns True if the status is uncertain but not an explicit error."'
        return self is ClassificationStatus.UNCERTAIN


class VideoCategory(Enum):
    '"Enum representing the content category of a video."'

    # Personal / Vlogging
    VLOG = auto()

    # Educational / Factual
    EDUCATIONAL_EXPLAINER = auto()
    LECTURE_PRESENTATION = auto()
    DOCUMENTARY = auto()
    HISTORY_EXPLAINER = auto()
    SCIENCE_NEWS = auto()
    ECONOMICS_EXPLAINER = auto()

    # Journalism
    NEWS_REPORT = auto()

    # Entertainment / Art
    COMEDY_SKETCH = auto()
    INTERVIEW_CONVERSATION = auto()
    MUSIC_ANALYSIS = auto()

    # Lifestyle
    TRAVEL_GUIDE = auto()

    # Miscellaneous
    OTHER = auto()


class TranscriptSource(Enum):
    '"Enum indicating where the transcript came from (Cache or API)."'

    CACHE = auto()
    YOUTUBE_API = auto()


@dataclass(frozen=True)
class VideoId:
    '"Value object representing a valid YouTube Video ID."'

    value: str

    def __post_init__(self) -> None:
        '# Validate ID format or presence.'
        if not self.value:
            raise ValueError("video_id value must be a non-empty string")


@dataclass
class Transcript:
    '"Domain entity representing a video transcript."'

    video_id: VideoId
    title: str
    author: str
    language: str
    text: str
    source: TranscriptSource
    created_at: datetime

    @property
    def is_empty(self) -> bool:
        '"Returns True if the transcript text is empty."'
        return not self.text.strip()


@dataclass
class TranscriptResult:
    '"Result object for transcript fetch operations."'

    status: ClassificationStatus
    transcript: Optional[Transcript] = None
    message: Optional[str] = None

    @property
    def has_transcript(self) -> bool:
        '"Returns True if a valid transcript is present."'
        return self.status.is_ok and self.transcript is not None


@dataclass
class ClassificationResult:
    '"Result object for classification operations."'

    status: ClassificationStatus
    category: Optional[VideoCategory] = None
    reason: Optional[str] = None
    raw_model_output: Optional[str] = None

    @property
    def has_category(self) -> bool:
        '"Returns True if a valid category was determined."'
        return self.status.is_ok and self.category is not None


@dataclass
class SummaryResult:
    '"Result object for summarization operations."'

    status: ClassificationStatus
    summary: Optional[str] = None
    used_category: Optional[VideoCategory] = None
    instruction: Optional[str] = None
    message: Optional[str] = None

    @property
    def has_summary(self) -> bool:
        '"Returns True if a summary was successfully generated."'
        return self.status.is_ok and self.summary is not None


@dataclass
class ProcessingResult:
    '"Aggregate result of the entire processing pipeline."'

    url: str
    video_id: Optional[VideoId]
    transcript_result: TranscriptResult
    classification_result: Optional[ClassificationResult] = None
    summary_result: Optional[SummaryResult] = None

    @property
    def is_fully_successful(self) -> bool:
        '"Returns True only if all steps (transcript, class, summary) succeeded."'
        if not self.transcript_result.status.is_ok:
            return False
        if self.classification_result is None:
            return False
        if not self.classification_result.status in (
            ClassificationStatus.OK,
            ClassificationStatus.UNCERTAIN,
        ):
            return False
        if self.summary_result is None or not self.summary_result.status.is_ok:
            return False
        return True


@dataclass
class VideoContext:
    '"Entity representing the state and data associated with a specific video."'

    video_id: VideoId
    # The URL used to access the video.
    url: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    transcript: Optional[Transcript] = None
    classification: Optional[ClassificationResult] = None
    summary: Optional[SummaryResult] = None

    def update_access_time(self) -> None:
        '"Updates the last_accessed timestamp to now."'
        self.last_accessed = datetime.now(timezone.utc)


@dataclass
class ToolResult:
    '"Standardized return type for all Agent Tools."'

    message: str
    success: bool = True
    data: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowState:
    '"Represents the current state of the agent workflow."'

    session_id: str
    user_id: str
    
                   
    # Map of video_id -> VideoContext.
    videos: Dict[str, VideoContext] = field(default_factory=dict)
    
                   
    # Stack of active video IDs (most recent last).
    active_video_ids: List[str] = field(default_factory=list)
    
                   
    # Flag indicating if the last tool execution succeeded.
    last_step_success: bool = True
    
    # Error message from the last failure, if any.
    last_error: Optional[str] = None

    def get_current_video(self) -> Optional[VideoContext]:
        '"Retrieves the VideoContext for the most recently active video."'
        if not self.active_video_ids:
            return None
        return self.videos.get(self.active_video_ids[-1])

    def add_video_context(self, ctx: VideoContext) -> None:
        '"Adds or updates a video context and marks it as active."'
        self.videos[ctx.video_id.value] = ctx
        
        # Move to end of list to mark as most recent.
        if ctx.video_id.value in self.active_video_ids:
            self.active_video_ids.remove(ctx.video_id.value)
        self.active_video_ids.append(ctx.video_id.value)
````

## File: app/agent.py
````python
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
        # Remove markdown code block delimiters if present.
        text = re.sub(r"^\s*```(json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

        try:
            data = json.loads(text)
            # Extract the plan and response fields, defaulting to empty list/None.
            return {
                "plan": data.get("plan", []),
                "response": data.get("response")
            }
        except json.JSONDecodeError as e:
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
````

## File: tests/test_agent_logic.py
````python
import unittest
import json
from unittest.mock import AsyncMock, MagicMock, create_autospec, ANY

from domain import (
    ClassificationStatus,
    ToolResult,
    Transcript,
    TranscriptResult,
    TranscriptSource,
    VideoContext,
    VideoId,
)
from infra.llm_client import LlmResponse
from config import settings

from app.agent import Agent
from app.tools import Tool, ListHistoryTool, FetchTranscriptTool

class TestPlanAndExecuteAgent(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Set up mocks for LLM, Memory, and history.
        self.mock_llm = MagicMock()
        self.mock_memory = AsyncMock()
        self.mock_history_manager = AsyncMock()
        
        # Setup a mock tool with specific return values.
        self.mock_history_tool = create_autospec(ListHistoryTool, instance=True)
        self.mock_history_tool.name = "list_video_history"
        self.mock_history_tool.description = "Lists available videos"
        self.mock_history_tool.run.return_value = ToolResult(message="Found: Python Video (ID: 123)")
        
        # Initialize the agent under test.
        self.agent = Agent(
            llm=self.mock_llm,
            tools=[self.mock_history_tool],
            memory_service=self.mock_memory,
            history_manager=self.mock_history_manager,
            model_name="test-model",
            max_tokens=1024,
            temperature=0.0
        )

    async def test_search_then_compare_workflow(self):
        """Tests a multi-step workflow: Search -> Compare."""
        # Setup a second mock tool for the comparison step
        mock_compare_tool = MagicMock()
        mock_compare_tool.name = "compare_videos"
        mock_compare_tool.description = "Compares videos"
        mock_compare_tool.run.return_value = ToolResult(message="I have compared the videos.")
        
        # Update agent tools to include the compare tool
        self.agent._tools["compare_videos"] = mock_compare_tool

        # Logic Update: The Agent is Plan-and-Execute (One shot). 
        # We provide ONE plan with TWO sequential steps.
        self.mock_llm.generate.return_value = LlmResponse(text=json.dumps({
            "plan": [
                {"tool": "list_video_history", "parameters": {"search_query": "Python"}},
                {"tool": "compare_videos", "parameters": {}}
            ],
            "response": "I will search and then compare."
        }))
        
        # Run the agent.
        response = await self.agent.run("Compare Python and Java", "sess_1", "user_1")
        
        # Verify call counts and final response.
        self.assertEqual(self.mock_llm.generate.call_count, 1) # Only one LLM call for the plan
        self.mock_history_tool.run.assert_called_once()
        mock_compare_tool.run.assert_called_once() # Ensure the second step ran
        
        # The agent returns the message from the LAST executed tool
        self.assertEqual(response, "I have compared the videos.")
        self.mock_history_manager.append_and_compact.assert_called_once()

    async def test_dirty_json_parsing(self):
        """Tests that the agent can extract JSON from markdown blocks."""
        dirty_response = """
        Here is the tool call you requested:
        ```json
        {
            "plan": [{"tool": "list_video_history", "parameters": {}}],
            "response": "Starting list..."
        }
        ```
        """
        self.mock_llm.generate.return_value = LlmResponse(text=dirty_response)
        
        await self.agent.run("test", "sess", "user")
        
        self.assertEqual(self.mock_llm.generate.call_count, 1)
        self.mock_history_tool.run.assert_called_once()

    async def test_sticky_context_injection(self):
        """Tests that active video context is implicitly passed to tools."""
        # Mock a current video context.
        self.mock_memory.get_current_video_context.return_value = VideoContext(
            url="http://youtube.com/test",
            video_id=VideoId("123")
        )
        
        # Logic Update: Make 'url' optional. The Agent passes empty params {}, 
        # relying on the tool logic to fetch from 'state' if 'url' is None.
        class UrlTool(Tool):
            name = "url_tool"
            description = "needs url"
            def run(self, state, url: str = None):
                if not url and state.get_current_video():
                    url = state.get_current_video().url
                return ToolResult(message=f"ok: {url}")

        local_tool = create_autospec(UrlTool, instance=True)
        local_tool.name = "url_tool"
        local_tool.description = "needs url"
        local_tool.run.return_value = ToolResult(message="ok")
        
        self.agent._tools = {"url_tool": local_tool}
        
        # LLM output that omits the 'url' parameter.
        self.mock_llm.generate.return_value = LlmResponse(text=json.dumps({
            "plan": [{"tool": "url_tool", "parameters": {}}],
            "response": "Using context URL."
        }))
        
        # Run agent.
        await self.agent.run("do it", "sess", "user")
        
        # Verify the tool was called with the state object.
        local_tool.run.assert_called_with(state=ANY, url=None)

    async def test_unknown_tool_recovery(self):
        """Tests that the agent recovers when it hallucinates a tool."""
        self.mock_llm.generate.return_value = LlmResponse(text=json.dumps({
            "plan": [{"tool": "fake_tool", "parameters": {}}],
            "response": "Trying fake tool."
        }))
        
        result = await self.agent.run("test", "sess", "user")
        
        # Verify the agent catches the unknown tool error and returns the appropriate message.
        self.assertIn("Error: Plan included unknown tool 'fake_tool'", result)
        self.assertEqual(self.mock_llm.generate.call_count, 1)

class TestTranscriptToolUX(unittest.TestCase):
    """Tests the user experience aspects of the transcript tool."""

    def setUp(self):
        self.mock_service = MagicMock()
        self.tool = FetchTranscriptTool(transcript_service=self.mock_service)
        self.video_id = VideoId("12345678901")

    def test_transcript_preview_length(self):
        """Ensures the tool output contains a preview but not the whole text."""
        # Create a very long transcript.
        long_text = "A" * 2000
        
        mock_transcript = Transcript(
            video_id=self.video_id,
            title="Long Video",
            author="Tester",
            language="en",
            text=long_text,
            source=TranscriptSource.YOUTUBE_API,
            created_at=None
        )

        # Configure mock service.
        self.mock_service.parse_video_id.return_value = self.video_id
        self.mock_service.get_transcript.return_value = TranscriptResult(
            status=ClassificationStatus.OK,
            transcript=mock_transcript
        )

        # Run the tool.
        result = self.tool.run(state=MagicMock(), url="http://fake.url")

        # Inspect result message.
        message = result.message
        
        # Verify message is not the full length.
        self.assertTrue(
            len(message) > 1000, 
            f"Message length {len(message)} is too short, implies truncation logic wasn't updated."
        )
        
        # Verify user guidance text exists.
        self.assertIn("Full transcript text is available in the agent's memory", message)
````
