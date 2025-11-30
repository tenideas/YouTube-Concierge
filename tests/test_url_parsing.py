import unittest
from services.transcript_service import TranscriptService
from domain import VideoId

class TestUrlParsing(unittest.TestCase):
    '"Tests the regex and parsing logic for YouTube URLs."'

    def test_standard_watch_url(self):
        '"Tests standard watch URLs."'
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        vid = TranscriptService.parse_video_id(url)
        self.assertEqual(vid.value, "dQw4w9WgXcQ")

    def test_shortened_url(self):
        '"Tests shortened URLs."'
        url = "https://youtu.be/dQw4w9WgXcQ"
        vid = TranscriptService.parse_video_id(url)
        self.assertEqual(vid.value, "dQw4w9WgXcQ")

    def test_shorts_url(self):
        '"Tests Shorts URLs."'
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        vid = TranscriptService.parse_video_id(url)
        self.assertEqual(vid.value, "dQw4w9WgXcQ")

    def test_live_url(self):
        '"Tests Live URLs."'
        url = "https://www.youtube.com/live/dQw4w9WgXcQ?feature=share"
        vid = TranscriptService.parse_video_id(url)
        self.assertEqual(vid.value, "dQw4w9WgXcQ")

    def test_embed_url(self):
        '"Tests Embed URLs."'
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        vid = TranscriptService.parse_video_id(url)
        self.assertEqual(vid.value, "dQw4w9WgXcQ")

    def test_raw_id(self):
        '"Tests raw ID strings."'
        raw = "dQw4w9WgXcQ"
        vid = TranscriptService.parse_video_id(raw)
        self.assertEqual(vid.value, "dQw4w9WgXcQ")

    def test_url_with_extra_params(self):
        '"Tests URLs with extra query parameters."'
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&ab_channel=RickAstley"
        vid = TranscriptService.parse_video_id(url)
        self.assertEqual(vid.value, "dQw4w9WgXcQ")

    def test_invalid_url_raises_value_error(self):
        '"Tests that invalid URLs raise ValueError."'
        invalid_urls = [
            "https://google.com",
            "not_a_video_id",
            "https://www.youtube.com/playlist?list=PL12345",
            "",
        ]
        for url in invalid_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    TranscriptService.parse_video_id(url)

if __name__ == "__main__":
    unittest.main()

