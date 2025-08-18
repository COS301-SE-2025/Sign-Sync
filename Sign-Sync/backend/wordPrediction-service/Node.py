from typing import Dict, List, Optional
from collections import Counter

class Node:
    def __init__(self) -> None:
        self.children: Dict[str, "Node"] = {}
        self.next_counts: Counter[str] = Counter()
        self.terminal: bool = False

    def to_dict(self) -> dict:
        return {
            "terminal": self.terminal,
            "next_counts": dict(self.next_counts),
            "children": {tok: child.to_dict() for tok, child in self.children.items()},
        }

    def from_dict(d: dict) -> "Node":
        node = Node()
        node.terminal = d.get("terminal", False)
        node.next_counts = Counter(d.get("next_counts", {}))
        node.children = {tok: Node.from_dict(cd) for tok, cd in d.get("children", {}).items()}
        return node