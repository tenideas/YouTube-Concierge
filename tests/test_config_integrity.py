import unittest
from domain import VideoCategory
from config.prompts import VIDEO_SUMMARY_INSTRUCTIONS

class TestConfigIntegrity(unittest.TestCase):
    '"Tests configuration consistency."'

    def test_all_categories_have_summary_instructions(self):
        '"Verifies that every VideoCategory has a corresponding summary prompt."'
        domain_categories = set(VideoCategory)
        configured_categories = set(VIDEO_SUMMARY_INSTRUCTIONS.keys())
        
                       
        missing = domain_categories - configured_categories
        
                       
                       
        # Fail if there are missing instructions.
        self.assertFalse(
            missing, 
            f"The following categories are missing summary instructions in config/prompts.py: {missing}"
        )

if __name__ == "__main__":
    unittest.main()
