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
