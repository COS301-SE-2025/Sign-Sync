import unittest
from unittest.mock import patch
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corrector import tidy, capitalize, gloss_to_english

class TestCorrector(unittest.TestCase):

    @patch("corrector.capitalize", return_value="FAKE")
    def test_tidy(self, mock_cap):
        print("Corrector: tidy() unit testing")
        # test for punctuation
        self.assertEqual(tidy(["STORE I GO"]), "FAKE")
        mock_cap.assert_called_with("STORE I GO.")
        # test empty input
        self.assertEqual(tidy([]), "FAKE")
        mock_cap.assert_called_with("")
        # test joining of tokens
        self.assertEqual(tidy(["STORE", "I", "GO."]), "FAKE")
        mock_cap.assert_called_with("STORE I GO.")

    def test_capitalize(self):
        print("Corrector: capitalize() unit testing")
        # test empty input
        self.assertIsNone(capitalize(None))
        # test capitalization of non-capitalized word
        self.assertEqual(capitalize("shop"), "Shop")
        # test capitalization of non-capitalized sentence
        self.assertEqual(capitalize("i go to the store"), "I go to the store")
        # test capitalization of capitalized word
        self.assertEqual(capitalize("Shop"), "Shop")
        # test capitalization of capitalized sentence
        self.assertEqual(capitalize("I go to the store"), "I go to the store")

    @patch("corrector.tidy", return_value="MOCKED_TIDY")
    @patch("corrector.article_for", return_value="MOCKED_ARTICLE")
    def test_gloss_to_english(self, mock_article, mock_tidy):
        print("Corrector: gloss_to_english() unit testing")

        # TOPIC PRONOUN VERB
        result = gloss_to_english("STORE I GO")
        self.assertEqual(result, "MOCKED_TIDY")
        mock_article.assert_called_with("store")
        mock_tidy.assert_called_with(["i", "go", "to", "MOCKED_ARTICLE", "store"])

        # PRONOUN VERB TOPIC
        result = gloss_to_english("I GO STORE")
        self.assertEqual(result, "MOCKED_TIDY")
        mock_article.assert_called_with("store")
        mock_tidy.assert_called_with(["i", "go", "to", "MOCKED_ARTICLE", "store"])

        # TOPIC VERB
        result = gloss_to_english("STORE GO")
        self.assertEqual(result, "MOCKED_TIDY")
        mock_article.assert_called_with("store")
        mock_tidy.assert_called_with(["i", "go", "to", "MOCKED_ARTICLE", "store"])

if __name__ == "__main__":
    unittest.main()
