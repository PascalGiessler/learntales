import unittest
from unittest.mock import MagicMock
from app.services.storyteller import StoryTeller


class TestStoryTeller(unittest.TestCase):
    def setUp(self):
        self.storyteller = StoryTeller(llm=MagicMock(), visionModel=MagicMock())

    def test_generate_images_from_prompt(self):
        self.fail()

    def test_visualize(self):
        expected_output = ["placeholder1", "placeholder2"]
        test_content = "[placeholder1][placeholder2]"
        test_prompt = "[]"
        result = self.storyteller.extract_placeholders_from_text(content=test_content)
        self.assertEqual(result, expected_output)

        expected_output = ["placeholder1"]
        test_content = "[placeholder1]"
        result = self.storyteller.extract_placeholders_from_text(content=test_content)
        self.assertEqual(result, expected_output)

        expected_output = []
        test_content = "[]"
        result = self.storyteller.extract_placeholders_from_text(content=test_content)
        self.assertEqual(result, expected_output)

        expected_output = []
        test_content = ""
        result = self.storyteller.extract_placeholders_from_text(content=test_content)
        self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
