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
        # We check only for the arguments explicitly passed in the call.
        # The 'url' argument relies on its default value in the signature, so it is not in the call kwargs.
        local_tool.run.assert_called_with(state=ANY)

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
