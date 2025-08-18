from typing import Dict, List, Optional
from collections import Counter

class Node:
    def __init__(self):
        self.children: Dict[str, "Node"] = {}
        self.next_counts: Counter[str] = Counter()
        self.terminal: bool = False

