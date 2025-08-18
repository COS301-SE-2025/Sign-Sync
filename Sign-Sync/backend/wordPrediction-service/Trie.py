from collections import Counter
from typing import List, Tuple, Dict, Any, Iterable, Optional

import json
import Node

# Trie Language Model for word prediction
class Trie:

    def __init__(self, lowercase: bool = False):
        self.root = Node()
        self.lowercase = lowercase

    
   
