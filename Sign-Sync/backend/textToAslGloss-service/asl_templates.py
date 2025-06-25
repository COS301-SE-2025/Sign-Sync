import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_action(text):
    doc = nlp(text)
    tokens = [t.lemma_.upper() for t in doc if t.is_alpha and t.text != "tomorrow"]
    return " ".join(tokens)

def clean_phrase(phrase: str) -> str:
    phrase = re.sub(r"[^\w\s]", "", phrase)          # remove punctuation
    phrase = re.sub(r"\bthe\b", "", phrase).strip()  # remove 'the'
    return " ".join(word for word in phrase.split() if word.isalpha())

def apply_asl_template(text: str) -> str:
    text = text.lower().strip()

    # Template 1: WH-Question "What time do we ___ tomorrow?"
    match = re.match(r"what time do we (.+?)\s*tomorrow\??", text)
    if match:
        action = clean_action(clean_phrase(match.group(1)))
        return f"TIME TOMORROW, WE {action} WHAT"

    # Template 2: "I am going to the ___"
    match = re.match(r"i am going to the (.+)", text)
    if match:
        place = clean_action(clean_phrase(match.group(1)))
        return f"{place} I GO"

    # Template 3: "Can you ___ me?"
    match = re.match(r"can you (.+?) me\??", text)
    if match:
        action = clean_action(clean_phrase(match.group(1)))
        return f"YOU {action} ME"

    # Template 4: "My name is John"
    match = re.match(r"my name is (.+)", text)
    if match:
        name = clean_phrase(match.group(1))
        return f"NAME ME {name.upper()}"

    # Template 5: "I'll call you at 5"
    match = re.match(r"i(?: will|'ll) call you at (\d+)", text)
    if match:
        time = match.group(1)
        return f"{time} OCLOCK I CALL YOU"

    # Template 6: "What are you doing later today?"
    match = re.match(r"what are you doing (.+)", text)
    if match:
        time_phrase = clean_phrase(match.group(1))
        return f"{time_phrase.upper()} YOU DO WHAT"

    # Template 7: "Where did she put the book?"
    match = re.match(r"where did she put (.+)", text)
    if match:
        obj = clean_phrase(match.group(1))
        return f"SHE PUT {obj.upper()} WHERE"

    return None
