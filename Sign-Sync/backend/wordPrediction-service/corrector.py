import re

PRONOUNS = {"i","you","he","she","we","they","it"}
VERB_PREP = {
    "go": "to", "walk": "to", "run": "to", "travel": "to",
    "come": "from", "return": "from",
    "give": "to", "send": "to", "bring": "to",
    "take": "from", "steal": "from",
    "look": "at", "see": "—", "watch": "—",
    "arrive": "at", "arrive_at": "at",
}


PLACES = {"store","school","home","office","park","university","hospital","bank","shop","supermarket"}

VOWELS = set("aeiou")

def article_for(noun: str) -> str:
    if noun in {"home"}:  
        return ""
    return "the"

def capitalize(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s

def tidy(tokens):
    s = " ".join(t for t in tokens if t)
    s = re.sub(r"\s+([,.!?;:])", r"\1", s).strip()
    if s and s[-1] not in ".!?":
        s += "."
    return capitalize(s)