import string

import numpy
dictionary = {
    "sad": [1.0, 0.0, 0.0, 0.0],
    "unhappy": [1.0, 0.0, 0.0, 0.0],
    "die": [1.0, 0.0, 0.0, 0.0],
    "fail": [0.75, 0.0, 0.0, 0.0],
    "bad": [0.5, 0.0, 0.0, 0.0],
    "happy": [0.0, 1.0, 0.0, 0.0],
    "glad": [0.0, 0.75, 0.0, 0.0],
    "pass": [0.0, 0.75, 0.0, 0.0],
    "birthday": [0.0, 0.75, 0.0, 0.0],
    "surprise": [0.0, 0.0, 1.0, 0.0],
    "how": [0.0, 0.0, 5.0, 0.0],
    "where": [0.0, 0.0, 10.0, 0.0],
    "who": [0.0, 0.0, 10.0, 0.0],
    "what": [0.0, 0.0, 10.0, 0.0],
    "when": [0.0, 0.0, 10.0, 0.0],
    "why": [0.0, 0.0, 10.0, 0.0],
    "anger": [0.0, 0.0, 0.0, 1.0],
    "angry": [0.0, 0.0, 0.0, 1.0],
    "fury": [0.0, 0.0, 0.0, 1.0],
    "furious": [0.0, 0.0, 0.0, 1.0],
    "rage": [0.0, 0.0, 0.0, 1.0],
    "irritate": [0.0, 0.0, 0.0, 0.5],
    "mad": [0.0, 0.0, 0.0, 0.75],
}
def classify(sentence):
    emotions = ["sad", "happy", "surprise", "anger"]
    matrix = [0.0, 0.0, 0.0, 0.0]
    words = sentence.lower().strip(string.punctuation).split(" ")
    negative = False
    for word in words:
        if word == "no" or word == "not":
            negative = True
        if word in dictionary:
            matrix = numpy.add(matrix,dictionary[word])
    if max(matrix) == 0:
        result = "neutral"
    else:
        result = emotions[numpy.argmax(matrix)]
        if negative:
            if result == "happy":result="sad"
            if result == "sad": result = "happy"
            if result == "anger": result = "neutral"

    return result