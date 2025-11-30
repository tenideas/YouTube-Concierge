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
    
