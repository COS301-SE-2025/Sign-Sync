from collections import Counter
from typing import List, Tuple, Dict, Any, Iterable, Optional

import json
import Node

# Trie Language Model for word prediction
class Trie:

    def __init__(self, lowercase: bool = False):
        self.root = Node()
        self.lowercase = lowercase

    
    def tokenize(self, text: str) -> List[str]:
        t = text.strip()
        if self.lowercase:
            t = t.lower()
        return [tok for tok in t.split() if tok]
    
    def add_sentence(self, sentence: str) -> None:
        tokens = self.tokenize(sentence)
        if not tokens:
            return
        
        node = self.root
        for tok in tokens:
            node.next_counts[tok] += 1
            if tok not in node.children:
                node.children[tok] = Node()
            node = node.children[tok]
        node.terminal = True

    def add_sentences(self, lines: List[str]) -> None:
        for line in lines:
            self.add_sentence(line)

    def descend(self, prefix_tokens: List[str]) -> Optional["Node"]:
        node = self.root
        for tok in prefix_tokens:
            node = node.children.get(tok)
            if node is None:
                return None
        return node

        
