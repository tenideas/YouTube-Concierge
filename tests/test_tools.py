import unittest
from unittest.mock import AsyncMock, create_autospec
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
