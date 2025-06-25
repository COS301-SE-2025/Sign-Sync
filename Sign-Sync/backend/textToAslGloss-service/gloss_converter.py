import spacy
import re

nlp = spacy.load("en_core_web_sm")
from spacy.lang.en.stop_words import STOP_WORDS
STOPWORDS = STOP_WORDS
ALWAYS_KEEP = {
    "go", "want", "eat", "call", "leave", "play", "one", "two", 
    "three", "four", "five", "six", "seven", "eight", "none",
    "zero", "have", "it", "never", "that", "give", "see", "tell",
    "put", "know", "come", "remember", "think", "ask", "stop", "do",
    "next", "please"
}

PRONOUNS = {"i", "you", "we", "he", "she", "they", "me", "us", "him", "her", "them"}
WH_QUESTIONS = {"what", "who", "where", "when", "why", "how", "which"}
# NEGATIONS = {"don't", "do not", "didn't", "not", "can't", "cannot", "won't", "no"}
NEGATIONS = {"not", "n't"}
TIME_WORDS = {"today", "tomorrow", "yesterday", "now", "tonight", "morning", "afternoon", "evening", "week", "later"}
# time phrases need to be added in some way: this week, next week, just now, etc

def preprocess_contractions(text: str) -> str:
    return (
        text.lower()
        .replace("won't", "will not")
        .replace("can't", "can not")
        .replace("don't", "not")
        .replace("doesn't", "does not")
        .replace("didn't", "did not")
        .replace("shouldn't", "should not")
        .replace("wouldn't", "would not")
        .replace("couldn't", "could not")
        .replace("isn't", "is not")
        .replace("aren't", "are not")
        .replace("wasn't", "was not")
        .replace("weren't", "were not")
        .replace("doing", "do")
        .replace("do you", "you")
    )


def convert_to_gloss(text):
    text = preprocess_contractions(text)
    doc = nlp(text.lower())

    time_tokens = []
    wh_tokens = []
    gloss = []

    for token in doc:
        print(f"TOKEN: {token.text}")

        if token.text in NEGATIONS:
            gloss.append("NOT")
        elif token.text in ALWAYS_KEEP:
            gloss.append(token.lemma_.upper())
        elif token.text in WH_QUESTIONS:
            wh_tokens.append(token.text.upper())
        elif token.text in PRONOUNS:
            gloss.append(token.text.upper())
        elif token.text in TIME_WORDS:
            time_tokens.append(token.text.upper())
        elif token.text not in STOPWORDS and token.is_alpha:
            gloss.append(token.lemma_.upper())

    # print(f"Time words: {time_tokens}")
    # print(f"gloss words: {gloss}")
    # print(f"wh words: {wh_tokens}")

    final_gloss = " ".join(time_tokens + gloss + wh_tokens)
    final_gloss = re.sub(r"[^\w\s]", "", final_gloss)  # remove punctuation
    final_gloss = re.sub(r"\bNOT YOU\b", "YOU NOT", final_gloss)
    final_gloss = re.sub(r"\bHE SAY WHAT\b", "WHAT HE SAY", final_gloss)

    return final_gloss
