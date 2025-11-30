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
