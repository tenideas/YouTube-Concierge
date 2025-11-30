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
