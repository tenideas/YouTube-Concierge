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
