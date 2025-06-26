from gloss_converter import convert_to_gloss
from phrase_matcher import match_phrase
from asl_templates import apply_asl_template


# -------- Phrase Matcher Tests --------
def test_phrase_how_are_you():
    assert match_phrase("how are you") == "HOW YOU"

def test_phrase_thank_you():
    assert match_phrase("thank you") == "THANK-YOU"

def test_phrase_not_found():
    assert match_phrase("something random and weird") is None

# -------- ASL Template Tests --------
def test_template_time_question():
    result = apply_asl_template("What time do we leave tomorrow?")
    assert result == "TIME TOMORROW, WE LEAVE WHAT"

def test_template_name():
    result = apply_asl_template("My name is John")
    assert result == "NAME ME JOHN"

def test_template_call_time():
    result = apply_asl_template("I will call you at 5")
    assert result == "5 OCLOCK I CALL YOU"

def test_template_put_object():
    result = apply_asl_template("Where did she put the keys?")
    assert result == "SHE PUT KEYS WHERE"

def test_template_unknown_fallback():
    assert apply_asl_template("This sentence matches nothing") is None

# -------- Gloss Converter Tests --------
def test_gloss_simple_statement():
    assert convert_to_gloss("I am happy") == "I HAPPY"

def test_gloss_negation_contraction():
    assert convert_to_gloss("I don't know") == "I NOT KNOW"

def test_gloss_negation_explicit():
    assert convert_to_gloss("I do not want that") == "I DO NOT WANT THAT"

def test_gloss_wh_question():
    assert convert_to_gloss("Why are you late?") == "YOU LATE WHY"

def test_gloss_time_first():
    assert convert_to_gloss("I will go tomorrow") == "TOMORROW I GO"

def test_gloss_complex_negation_question():
    assert convert_to_gloss("Why didn't you help me yesterday?") == "YESTERDAY YOU NOT HELP ME WHY"

def test_gloss_negation_fix_not_you():
    assert convert_to_gloss("Why didn't you call me?") == "YOU NOT CALL ME WHY"

def test_gloss_subjective_thought():
    assert convert_to_gloss("I think we should leave") == "I THINK WE LEAVE"

def test_gloss_command_stop_talking():
    assert convert_to_gloss("Please stop talking") == "PLEASE STOP TALK"

def test_gloss_dont_touch_that():
    assert convert_to_gloss("Don't touch that!") == "NOT TOUCH THAT"
