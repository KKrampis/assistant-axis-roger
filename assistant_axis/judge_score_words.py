"""Judge scores written in words (2026-11-02).

GPT-5.6 Luna now and then answers a percentile rubric in words, ``SCORE: thirty-five``, in about
0.2% of its replies and usually for a whole batch at a time (GPT-6 Luna and gpt-4.1-mini much more
rarely), and the digit-only parsers scored those replies as missing.  Roger, 2026-11-02: accept
them, with UK and US spellings and any case.

:func:`parse_number_words` reads such a value.  :func:`marker_value` applies the digits-first rule
the rubric parsers use: a reply with a digit marker parses exactly as it did before, so this module
can only turn a missing score into a score.  In particular ``SCORE: thirty? Must integer no words.
SCORE: 32`` stays 32.  Pipeline step 3 does not use it: its rubrics score 0 to 3, and none of
4.8 million saved step-3 markers held a number in words.
"""

from __future__ import annotations

import re
from typing import Optional, Pattern

_ONES = {
    "zero": 0, "nought": 0, "naught": 0, "nil": 0,  # nought UK, naught US
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
#: A sign: a word, or a character before the number.
_NEGATIVE = {"minus", "negative"}
_POSITIVE = {"plus", "positive"}
#: Words that may follow a number without making it something else ("thirty percent").
_UNIT_WORDS = {"percent", "per", "percentile", "pct", "out", "points", "point"}
#: A number run together: "thirtyfive".
_RUN_TOGETHER = re.compile(rf"^({'|'.join(_TENS)})({'|'.join(k for k, v in _ONES.items() if 1 <= v <= 9)})$")
_TOKEN = re.compile(r"[a-z]+")
#: Markup a judge may put round a value, and dashes it may join words with or sign them with.
_MARKUP = "*_`\"'“”‘’"
_DASHES = "-‐‑‒–—−"
_PARENTHETICAL_DIGITS = re.compile(r"^\s*\(\s*([+-]?\d+)\s*\)")
#: The value after a SCORE marker, to the end of its line.
WORD_MARKER = re.compile(r"SCORE\s*:[ \t]*([^\n]*)", flags=re.IGNORECASE)


def parse_number_words(text: str) -> Optional[int]:
    """The integer ``text`` opens with in words, or None.

    Any case; hyphens, dashes, spaces or nothing between the parts (``Thirty-Five``, ``thirty five``,
    ``thirtyfive``); UK and US forms (``one hundred and five``, ``one hundred five``, ``a hundred``,
    ``nought``, ``naught``, ``zero``); a sign (``-forty``, ``- forty``, ``minus forty``); markup round
    it (``**forty**``).  The number must end there: followed by the end of the text, punctuation or a
    unit (``thirty?``, ``thirty percent``), not by other words, so that ``one of the highest`` is not
    1.  A number repeated in digits must agree: ``sixty-five (65)`` is 65, ``sixty-five (70)`` None.
    """
    s = text.lower().strip().lstrip(_MARKUP).strip()
    sign = 1
    if s and (s[0] in _DASHES or s[0] == "+"):
        sign = 1 if s[0] == "+" else -1
        s = s[1:].lstrip()
    pos = 0
    tokens = []
    for m in _TOKEN.finditer(s):
        gap = s[pos:m.start()]
        # words of one number are joined by spaces or dashes only
        if gap.strip(" \t" + _DASHES) or len(gap) > 3:
            break
        tokens.append((m.group(0), m.end()))
        pos = m.end()
    if not tokens:
        return None
    if tokens[0][0] in _NEGATIVE | _POSITIVE:
        if tokens[0][0] in _NEGATIVE:
            sign = -1
        tokens = tokens[1:]
    value, used = _read_number([t for t, _ in tokens])
    if value is None:
        return None
    end = tokens[used - 1][1]
    if used < len(tokens) and tokens[used][0] not in _UNIT_WORDS:
        return None
    rest = s[end:].lstrip(_MARKUP)
    if rest[:1].isalnum():
        return None
    again = _PARENTHETICAL_DIGITS.match(rest)
    if again and abs(int(again.group(1))) != value:
        return None
    return sign * value


def _read_number(tokens: list) -> tuple:
    """``(value, tokens used)`` of the number ``tokens`` opens with (0 to 999), or ``(None, 0)``."""
    i = total = 0
    if i < len(tokens) and tokens[i] in ("a", "an") and i + 1 < len(tokens) and tokens[i + 1] == "hundred":
        total, i = 100, 2
    elif i < len(tokens) and tokens[i] == "hundred":
        total, i = 100, 1
    elif i + 1 < len(tokens) and tokens[i] in _ONES and 1 <= _ONES[tokens[i]] <= 9 and tokens[i + 1] == "hundred":
        total, i = _ONES[tokens[i]] * 100, 2
    if total:
        j = i + 1 if i < len(tokens) and tokens[i] == "and" else i
        rest, used = _read_below_hundred(tokens[j:])
        if rest is not None:
            return total + rest, j + used
        return total, i
    return _read_below_hundred(tokens)


def _read_below_hundred(tokens: list) -> tuple:
    if not tokens:
        return None, 0
    first = tokens[0]
    if first in _TENS:
        if len(tokens) > 1 and tokens[1] in _ONES and 1 <= _ONES[tokens[1]] <= 9:
            return _TENS[first] + _ONES[tokens[1]], 2
        return _TENS[first], 1
    if first in _ONES:
        return _ONES[first], 1
    run = _RUN_TOGETHER.match(first)
    if run:
        return _TENS[run.group(1)] + _ONES[run.group(2)], 1
    return None, 0


def marker_value(text: str, digit_marker: Pattern[str]) -> Optional[int]:
    """The one score ``text`` holds under a SCORE marker: from the digit markers when it has any
    (exactly one, or None, as the digit-only parsers read it), else from markers holding a number in
    words (exactly one that reads as a number, or None)."""
    digits = digit_marker.findall(text)
    if digits:
        return int(digits[0]) if len(digits) == 1 else None
    words = [v for v in map(parse_number_words, WORD_MARKER.findall(text)) if v is not None]
    return words[0] if len(words) == 1 else None
