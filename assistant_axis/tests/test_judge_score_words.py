"""Judge scores written in words (Luna's quirk, 2026-11-02)."""
import re

import pytest

from assistant_axis.judge_score_words import marker_value, parse_number_words

DIGITS = re.compile(r"SCORE\s*:\s*\**([+-]?\d+)\b", flags=re.IGNORECASE)


@pytest.mark.parametrize("text,value", [
    # as Luna writes them
    ("thirty-five", 35), ("thirty", 30), ("thirty?", 30), ("sixty-five (65)", 65), ("forty (40)", 40),
    ("- forty", -40), ("- forty-five", -45), ("fifty?", 50),
    # case
    ("Thirty-Five", 35), ("THIRTY FIVE", 35), ("One Hundred", 100),
    # joined by a hyphen, a dash, a space or nothing
    ("thirty five", 35), ("thirty–five", 35), ("thirty—five", 35), ("thirtyfive", 35),
    # UK and US forms
    ("one hundred and five", 105), ("one hundred five", 105), ("a hundred", 100), ("hundred", 100),
    ("nought", 0), ("naught", 0), ("zero", 0), ("nil", 0), ("fourty", 40),
    # a sign
    ("-forty", -40), ("−forty", -40), ("+forty", 40), ("minus forty", -40), ("negative forty", -40), ("plus ten", 10),
    # markup and what may follow
    ("**forty**", 40), ("`forty`", 40), ("\"forty\"", 40), ("forty.", 40), ("forty,", 40), ("forty/100", 40),
    ("thirty percent", 30), ("thirty-five out of 100", 35), ("twenty-one, because", 21),
    # every value of the percentile scale
    ("nineteen", 19), ("twenty", 20), ("ninety-nine", 99),
])
def test_number_words_read(text, value):
    assert parse_number_words(text) == value


@pytest.mark.parametrize("text", [
    "", "   ", "high", "n/a", "the", "about thirty", "32", "**", "-",
    "one of the highest",      # reasoning that happens to follow "score:"
    "thirty apples",
    "sixty-five (70)",         # the digits disagree with the words
    "forty two three",         # not one number
])
def test_not_a_number_in_words(text):
    assert parse_number_words(text) is None


def test_every_value_from_0_to_100_reads_back():
    ones = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    tens = "twenty thirty forty fifty sixty seventy eighty ninety".split()
    words = {0: "zero", 100: "one hundred"}
    words.update({i + 1: w for i, w in enumerate(ones)})
    for t, w in enumerate(tens):
        words[20 + 10 * t] = w
        for u in range(1, 10):
            words[20 + 10 * t + u] = f"{w}-{ones[u - 1]}"
    assert sorted(words) == list(range(101))
    for value, text in words.items():
        assert parse_number_words(text) == value
        assert parse_number_words(text.upper()) == value


@pytest.mark.parametrize("reply,value", [
    ("It leans. SCORE: 32", 32),
    ("It leans. SCORE: thirty-five", 35),
    ("It leans.\nscore: Thirty-Five\n", 35),
    # a digit marker wins: Luna correcting itself reads as it always did
    ("SCORE: thirty? Must integer no words. SCORE: 32", 32),
    # exactly one score, in either form
    ("SCORE: 3 and SCORE: 4", None),
    ("SCORE: thirty\nSCORE: forty", None),
    # a word marker that is no number does not count against the one that is
    ("Hard to give a score: the evidence is thin.\nSCORE: thirty", 30),
    ("A high score: about 30", None),
    ("No marker at all, thirty", None),
])
def test_marker_value(reply, value):
    assert marker_value(reply, DIGITS) == value


def test_a_reply_with_digit_markers_parses_as_the_digit_only_parser_did():
    """The fix can only turn a missing score into a score."""
    for reply in ("SCORE: 7", "SCORE: 7 SCORE: 8", "SCORE: seven SCORE: 8", "**SCORE: **9", "SCORE: 12%"):
        old = DIGITS.findall(reply)
        assert marker_value(reply, DIGITS) == (int(old[0]) if len(old) == 1 else None)
