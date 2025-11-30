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
