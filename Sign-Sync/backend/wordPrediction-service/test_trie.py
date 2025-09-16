import unittest
from unittest.mock import patch, MagicMock
from Trie import Trie
from collections import Counter

class TestTrie(unittest.TestCase):

    def setUp(self):
        patcher = patch("Trie.Node")
        self.mock_node_class = patcher.start()
        self.mock_node_instance = MagicMock()
        self.mock_node_instance.next_counts = Counter()
        self.mock_node_instance.children = {}
        self.mock_node_instance.terminal = False
        self.mock_node_class.return_value = self.mock_node_instance

        self.trie = Trie()

    def tearDown(self):
        patch.stopall()

    def test_trie_inits_root(self):
        self.mock_node_class.assert_called_once()
        self.assertEqual(self.trie.root, self.mock_node_instance)

    def test_add_sentence(self):
        self.mock_node_instance.children = {}

        self.trie.add_sentence("hello world")

        self.assertIn("hello", self.mock_node_instance.next_counts)
        self.assertIn("hello", self.mock_node_instance.children)
        self.assertIn("world", self.mock_node_instance.children["hello"].next_counts)
        self.assertIn("world", self.mock_node_instance.children["hello"].children)
        self.mock_node_class.assert_called()

    def test_descend_with_children(self):
        fake_child = MagicMock()
        self.mock_node_instance.children = {"token": fake_child}

        result = self.trie.descend(["token"])
        self.assertEqual(result, fake_child)

    def test_descend_with_missing_child(self):
        self.mock_node_instance.chidren = {}

        result = self.trie.descend(["missing"])
        self.assertIsNone(result)




if __name__ == "__main__":
    unittest.main()