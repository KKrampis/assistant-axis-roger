"""Boundary-input hardening tests for ``optimal_axis_for_judge``.

The script accepts a user-supplied ``--scores_file`` whose JSON keys
sometimes leak the display form (spaces, hyphens, capitals,
apostrophes -- e.g. ``"Aligned Artificial Intelligence"``) instead
of the file-name form used as canonical keys throughout the
pipeline.  Without normalisation those keys silently miss on every
multi-word entity at the downstream
``set(scores) & set(geometry_names)`` intersection.

These tests pin down ``_normalize_scores_file_keys``, which is the
single boundary that converts display-form keys to file-name form
with a WARNING and fatals on collisions.

See AGENT_NOTES "File-name vs display-name convention".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from results_analysis import optimal_axis_for_judge as oafj


SRC = Path("/tmp/dummy_scores.json")


class TestNormalizeScoresFileKeys:
    def test_passthrough_file_form(self):
        """File-name input is the common path: pass through unchanged
        with no warning."""
        scores = {
            "patient": 0.5,
            "stoic": -0.3,
            "aligned_artificial_intelligence": 0.7,
        }
        out = oafj._normalize_scores_file_keys(scores, src=SRC)
        assert out == scores

    def test_passthrough_disambiguated_ids(self):
        """``name|R``/``name|T`` ids are already canonical; the
        suffix prevents lower-casing the kind tag."""
        scores = {"patient|R": 0.5, "patient|T": -0.5}
        out = oafj._normalize_scores_file_keys(scores, src=SRC)
        assert out == scores

    def test_spaces_to_underscores(self, capsys):
        scores = {
            "aligned artificial intelligence": 0.7,
            "systems thinker": 0.3,
        }
        out = oafj._normalize_scores_file_keys(scores, src=SRC)
        assert out == {
            "aligned_artificial_intelligence": 0.7,
            "systems_thinker": 0.3,
        }
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "coerced" in err
        assert "2 display-form key(s)" in err

    def test_hyphens_to_underscores(self, capsys):
        scores = {"systems-thinker": 0.3}
        out = oafj._normalize_scores_file_keys(scores, src=SRC)
        assert out == {"systems_thinker": 0.3}
        assert "WARNING" in capsys.readouterr().err

    def test_capitals_lowercased(self, capsys):
        scores = {"Aligned Artificial Intelligence": 0.7, "PATIENT": 0.4}
        out = oafj._normalize_scores_file_keys(scores, src=SRC)
        assert out == {
            "aligned_artificial_intelligence": 0.7,
            "patient": 0.4,
        }
        assert "WARNING" in capsys.readouterr().err

    def test_apostrophes_stripped(self, capsys):
        scores = {"devil's advocate": 0.5}
        out = oafj._normalize_scores_file_keys(scores, src=SRC)
        assert out == {"devils_advocate": 0.5}
        assert "WARNING" in capsys.readouterr().err

    def test_collision_raises_systemexit(self):
        """If two distinct keys normalise to the same file-name form
        (one display, one file), fatal -- silently dropping one
        would lose data."""
        scores = {
            "Patient": 0.5,
            "patient": 0.7,  # collides under normalisation
        }
        with pytest.raises(SystemExit) as exc:
            oafj._normalize_scores_file_keys(scores, src=SRC)
        assert "collide" in str(exc.value).lower()

    def test_collision_systemexit_lists_offenders(self):
        scores = {
            "systems-thinker": 0.5,
            "Systems Thinker": 0.7,  # also -> systems_thinker
            "patient": 0.4,
        }
        with pytest.raises(SystemExit) as exc:
            oafj._normalize_scores_file_keys(scores, src=SRC)
        msg = str(exc.value)
        assert "systems_thinker" in msg
        assert "systems-thinker" in msg or "Systems Thinker" in msg

    def test_no_warning_when_all_file_form(self, capsys):
        """The common path must be quiet to avoid alarm fatigue."""
        scores = {"patient": 0.5, "stoic": -0.3}
        oafj._normalize_scores_file_keys(scores, src=SRC)
        assert "WARNING" not in capsys.readouterr().err

    def test_mixed_file_and_display(self, capsys):
        """Mostly-file-form input with a single leaked display key
        still gets normalised (the realistic ergonomic accident)."""
        scores = {
            "patient": 0.5,
            "stoic": -0.3,
            "Aligned Artificial Intelligence": 0.7,  # one stray
        }
        out = oafj._normalize_scores_file_keys(scores, src=SRC)
        assert out == {
            "patient": 0.5,
            "stoic": -0.3,
            "aligned_artificial_intelligence": 0.7,
        }
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "1 display-form key(s)" in err

    def test_warning_truncates_long_lists(self, capsys):
        """When more than 5 keys are coerced, the message says
        '...' rather than dumping all of them."""
        scores = {f"Word{i} Word{i}": float(i) for i in range(8)}
        oafj._normalize_scores_file_keys(scores, src=SRC)
        err = capsys.readouterr().err
        assert "8 display-form key(s)" in err
        assert "..." in err

    def test_curly_apostrophe_stripped(self, capsys):
        """U+2019 (smart-quote, default from word processors) must
        normalise the same as ASCII apostrophe."""
        scores = {"devil\u2019s advocate": 0.5}
        out = oafj._normalize_scores_file_keys(scores, src=SRC)
        assert out == {"devils_advocate": 0.5}
        assert "WARNING" in capsys.readouterr().err

    def test_empty_dict(self):
        """Empty input is well-defined: empty output, no warning."""
        assert oafj._normalize_scores_file_keys({}, src=SRC) == {}

    def test_disambiguated_id_does_not_get_lowercased(self):
        """Defensive: ``Patient|R`` (capital P) is malformed but we
        currently preserve the suffix tag and refuse to mangle the
        bare-name half — the SystemExit branch only fires on
        post-normalisation collisions, not on weird-but-legal
        inputs.  This documents that behaviour so a refactor
        doesn't accidentally ``.lower()`` the bare name."""
        scores = {"Patient|R": 0.5}
        out = oafj._normalize_scores_file_keys(scores, src=SRC)
        # Pass-through (the |R suffix marks it as already-canonical
        # for our purposes; if the user wanted it lower-cased they
        # would have written that in the first place).
        assert out == {"Patient|R": 0.5}
