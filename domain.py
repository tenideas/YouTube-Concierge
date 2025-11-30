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
