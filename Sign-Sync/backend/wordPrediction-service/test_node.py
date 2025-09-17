import unittest
from unittest.mock import patch
from Node import Node
from collections import Counter

class TestNode(unittest.TestCase):

    def test_to_dict_empty(self):
        node = Node()
        d = node.to_dict()
        self.assertFalse(d["terminal"])
        self.assertEqual(d["next_counts"], {})
        self.assertEqual(d["children"], {})

    def test_to_dict_not_empty(self):
        node = Node()
        node.next_counts.update(["a", "a", "b"])
        child = Node()
        child.terminal = True
        node.children["x"] = child

        d = node.to_dict()

        self.assertFalse(d["terminal"])
        self.assertEqual(d["next_counts"], {"a": 2, "b": 1})
        self.assertIn("x", d["children"])
        self.assertTrue(d["children"]["x"]["terminal"])

    def test_from_dict_empty(self):
        d = {"terminal": True, "next_counts": {}, "children": {}}
        node = Node.from_dict(d)

        self.assertTrue(node.terminal)
        self.assertEqual(node.next_counts, {})
        self.assertEqual(node.children, {})

    def test_from_dict_not_empty(self):
        d = {
            "terminal": False,
            "next_counts": {"y": 1},
            "children": {
                "y": {"terminal": True, "next_counts": {}, "children": {}}
            }
        }
        node = Node.from_dict(d)

        self.assertFalse(node.terminal)
        self.assertEqual(node.next_counts, Counter({"y": 1}))
        self.assertIn("y", node.children)
        self.assertTrue(node.children["y"].terminal)
    

if __name__ == "__main__":
    unittest.main()

