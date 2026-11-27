"""English syllable counts, to size a translated line to the time it has on screen.

A spelling heuristic (shared with Svara): right for most everyday words, and
off by one now and then, which a time budget absorbs easily.
"""

from __future__ import annotations

import re

EXCEPTIONS = {
    "area": 3, "being": 2, "create": 2, "created": 3, "creation": 3, "diet": 2,
    "every": 2, "everyone": 3, "everything": 3, "fire": 1, "hour": 1, "idea": 3,
    "ideas": 3, "lion": 2, "naive": 2, "our": 1, "people": 2, "piano": 3,
    "poem": 2, "poet": 2, "quiet": 2, "real": 1, "recipe": 3, "science": 2,
    "smile": 1, "the": 1, "video": 3, "whole": 1, "world": 1, "yes": 1,
    "you": 1, "your": 1, "heaven": 2, "flower": 2, "power": 2, "different": 3,
    "family": 3, "evening": 2, "one": 1, "once": 1, "some": 1, "come": 1,
    "done": 1, "none": 1, "love": 1, "above": 2, "business": 2, "favorite": 3,
    "government": 3, "interest": 3, "interesting": 4, "usually": 4, "actually": 4,
}

WORD = re.compile(r"[A-Za-z']+")
VOWEL_GROUP = re.compile(r"[aeiouy]+")
NUMBER = re.compile(r"\d+")


def words(text: str) -> list[str]:
    return [w.lower().strip("'") for w in WORD.findall(text.replace("-", " ")) if w.strip("'")]


def word_syllables(word: str) -> int:
    w = word.replace("'", "")
    if not w:
        return 0
    if w in EXCEPTIONS:
        return EXCEPTIONS[w]
    if len(w) <= 3:
        return 1
    n = len(VOWEL_GROUP.findall(w))
    if w.endswith("e") and not re.search(r"[^aeiouy]le$", w) and re.search(r"[^aeiouy]e$", w):
        n -= 1
    elif re.search(r"[^aeiouy]es$", w) and not re.search(r"(s|x|z|ch|sh|ce|ge)es$|[cg]es$", w):
        n -= 1
    elif re.search(r"[^aeiouytd]ed$", w):
        n -= 1
    n += len(re.findall(r"(?<![cstgx])i[ao]|ua(?!r)|[aeiouy]ing$", w))
    return max(1, n)


def count(text: str) -> int:
    # Digits are spoken as words; two syllables a number is a fair guess.
    return sum(word_syllables(w) for w in words(text)) + 2 * len(NUMBER.findall(text))
