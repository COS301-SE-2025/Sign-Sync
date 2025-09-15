import unittest
from unittest.mock import patch
from corrector import tidy, capitalize

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

    def test_gloss_to_english(self):
        print("Corrector: gloss_to_english() unit testing")
        

if __name__ == "__main__":
    unittest.main()
