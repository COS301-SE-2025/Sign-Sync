from typing import Dict, List, Optional

class Node:
    def __init__(self):
        self.children: Dict[str, "Node"] = {}
        self.terminal: bool = False

        