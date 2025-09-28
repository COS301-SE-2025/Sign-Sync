import re

PRONOUNS = {"i","you","he","she","we","they","it"}

POSSESSIVES = {"my", "your", "our", "their", "his", "her", "its"}

VERB_PREP = {
    "go": "to", "walk": "to", "run": "to", "travel": "to",
    "come": "from", "return": "from",
    "give": "to", "send": "to", "bring": "to",
    "take": "from", "steal": "from",
    "look": "at", "see": "—", "watch": "—",
    "arrive": "at", "arrive_at": "at",
    # extra verbs with no prep
    "drink": None, "eat": None, "have": None, "help": None, "know": None,
    "learn": None, "like": None, "need": None, "play": None, "want": None,
    "work": None
}

PLACES = {
    "store","school","home","office","park","university","hospital",
    "bank","shop","supermarket","city","house","room"
}


VOWELS = set("aeiou")

def article_for(noun: str) -> str:
    if noun in {"home"}:
        return ""
    if any(noun.startswith(p + " ") for p in POSSESSIVES):
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

def gloss_to_english(text: str) -> str:
    toks = text.strip().lower().split()
    if not toks:
        return ""

    # POSSESSIVE + NOUN (merge)
    if len(toks) >= 2 and toks[0] in POSSESSIVES:
        poss, noun, *rest = toks
        merged = f"{poss} {noun}"
        new_tokens = [merged] + rest
        # re-run with merged tokens
        return gloss_to_english(" ".join(new_tokens))

    # TOPIC PRONOUN VERB
    if len(toks) >= 3 and toks[0] not in PRONOUNS and toks[1] in PRONOUNS:
        topic, subj, verb, *rest = toks
        prep = VERB_PREP.get(verb, None)
        parts = [subj, verb]
        if topic in PLACES or any(topic.endswith(" " + p) for p in PLACES):
            if prep:
                art = article_for(topic.split()[-1])
                if prep != "—":
                    parts.append(prep)
                if art:
                    parts.append(art)
            parts.append(topic)
        else:
            parts.append(topic)
        if rest:
            parts.extend(rest)
        return tidy(parts)

    # PRONOUN VERB TOPIC
    if len(toks) >= 3 and toks[0] in PRONOUNS:
        subj, verb, obj, *rest = toks
        prep = VERB_PREP.get(verb, None)
        parts = [subj, verb]
        if obj in PLACES or any(obj.endswith(" " + p) for p in PLACES):
            if prep:
                art = article_for(obj.split()[-1])
                if prep != "—":
                    parts.append(prep)
                if art:
                    parts.append(art)
            parts.append(obj)
        else:
            parts.append(obj)
        if rest:
            parts.extend(rest)
        return tidy(parts)

    # TOPIC VERB
    if len(toks) == 2 and toks[1] in VERB_PREP:
        obj, verb = toks
        prep = VERB_PREP.get(verb)
        parts = ["i", verb]
        if obj in PLACES or any(obj.endswith(" " + p) for p in PLACES):
            if prep:
                art = article_for(obj.split()[-1])
                if prep != "—":
                    parts.append(prep)
                if art:
                    parts.append(art)
            parts.append(obj)
        else:
            parts.append(obj)
        return tidy(parts)

    # Fall back
    return tidy(toks)
