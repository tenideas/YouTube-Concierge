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
