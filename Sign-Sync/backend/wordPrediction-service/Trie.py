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
    
    def stream_file(self, path: str, encoding: str = "utf-8") -> Optional["Node"]:
        with open(path, "r", encoding=encoding) as f:
            self.add_sentences(f)
        return self

    def predict_next(
        self,
        prefix: str,
        top_k: int = 5,
        min_count: int = 1,
        backoff: bool = True,
        add_k: float = 0.0, 
    ) -> List[Tuple[str, float]]:
        
        toks = self._tok(prefix)

        starts = range(0, len(toks) + 1) if backoff else range(len(toks), len(toks) + 1)
        for s in starts:
            node = self._descend(toks[s:])
            if node is None:
                continue

            items = node.next_counts.items()
            if add_k <= 0:
                items = [(w, c) for (w, c) in items if c >= min_count]
                total = sum(c for _, c in items)
                if total == 0:
                    continue
                scored = [(w, c / total) for (w, c) in items]
            else:
                V = len(node.next_counts)
                total = sum(node.next_counts.values()) + add_k * V
                scored = [(w, (node.next_counts[w] + add_k) / total) for (w, _) in items]

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

        return []
    
    # Optional unrandked set of next words
    def next_options(self, prefix: str) -> List[str]:
        toks = self.tokenize(prefix)
        node = self.descend(toks)
        return list(node.children.keys()) if node else []
    
    def save_json(self, path: str, encoding: str = "utf-8") -> None:
        payload = {"lowercase": self.lowercase, "root": self.root.to_dict()}
        with open(path, "w", encoding=encoding) as f:
            json.dump(payload, f, ensure_ascii=False)