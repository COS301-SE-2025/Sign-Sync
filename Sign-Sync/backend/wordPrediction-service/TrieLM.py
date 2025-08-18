from collections import Counter
from typing import List, Tuple, Dict, Any, Iterable, Optional

import json
import os

# Trie Language Model for word prediction
class TrieLM:
   
    def __init__(self, order: int = 3, topk_children: Optional[int] = None, min_count: int = 1):
        assert order >= 2, "order must be >= 2"
        self.order = order
        self.topk_children = topk_children
        self.min_count = min_count
        self.root = self._node()
        self.unigrams = Counter()

    @staticmethod
    def _node() -> Dict[str, Any]:
        return {"ch": {}, "cnt": 0, "next": Counter()}