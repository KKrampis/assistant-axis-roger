"""Tests for ``assistant_axis.entity_id``.

The 11 known content-bearing collision names (nine from the May 2026
corpus audit; ``specialist`` and ``parent`` joined in September 2026)
are covered explicitly so a future regression to bare-name keys is
caught by name.  ``test_collision_regression.py`` checks the list
against the stems actually on disk, so it cannot go stale silently.
"""
import pytest

from assistant_axis.entity_id import (
    EntityId,
    ID_SEPARATOR,
    KIND_LONG,
    KIND_R,
    KIND_T,
    display_label,
    entity_id,
    is_entity_id,
    kind_long,
    kind_short,
    parse_entity_id,
)


COLLISION_NAMES = [
    "ascetic",
    "contrarian",
    "cosmopolitan",
    "generalist",
    "pacifist",
    "parent",
    "patient",
    "perfectionist",
    "romantic",
    "specialist",
    "stoic",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_constants():
    assert KIND_R == "R"
    assert KIND_T == "T"
    assert ID_SEPARATOR == "|"
    assert KIND_LONG == {"R": "roles", "T": "traits"}


# ---------------------------------------------------------------------------
# entity_id()
# ---------------------------------------------------------------------------

def test_entity_id_basic_role():
    assert entity_id("patient", "roles") == "patient|R"


def test_entity_id_basic_trait():
    assert entity_id("patient", "traits") == "patient|T"


def test_entity_id_short_kind_tags():
    assert entity_id("patient", "R") == "patient|R"
    assert entity_id("patient", "T") == "patient|T"


def test_entity_id_singular_kind_tokens():
    assert entity_id("patient", "role") == "patient|R"
    assert entity_id("patient", "trait") == "patient|T"


@pytest.mark.parametrize(
    "kind_token,expected",
    [
        ("Roles", "R"),
        ("ROLES", "R"),
        ("rOlEs", "R"),
        ("Traits", "T"),
        ("TRAITS", "T"),
        ("  roles  ", "R"),  # whitespace-tolerant
    ],
)
def test_entity_id_case_and_whitespace(kind_token, expected):
    assert entity_id("foo", kind_token) == f"foo|{expected}"


@pytest.mark.parametrize("name", COLLISION_NAMES)
def test_entity_id_all_collisions_distinguished(name):
    """Each collision name must yield distinct ids for the two kinds."""
    role_id = entity_id(name, "roles")
    trait_id = entity_id(name, "traits")
    assert role_id != trait_id
    assert role_id == f"{name}|R"
    assert trait_id == f"{name}|T"


def test_entity_id_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        entity_id("foo", "rolez")


def test_entity_id_rejects_non_string_kind():
    with pytest.raises(ValueError, match="must be a string"):
        entity_id("foo", 1)  # type: ignore[arg-type]


def test_entity_id_rejects_pipe_in_name():
    with pytest.raises(ValueError, match="reserved separator"):
        entity_id("foo|bar", "R")


def test_entity_id_rejects_empty_name():
    with pytest.raises(ValueError, match="must not be empty"):
        entity_id("", "R")


def test_entity_id_rejects_non_string_name():
    with pytest.raises(ValueError, match="must be a string"):
        entity_id(42, "R")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_entity_id()
# ---------------------------------------------------------------------------

def test_parse_entity_id_basic():
    assert parse_entity_id("patient|R") == EntityId("patient", "roles")
    assert parse_entity_id("patient|T") == EntityId("patient", "traits")


def test_parse_entity_id_named_tuple_fields():
    parsed = parse_entity_id("perfectionist|T")
    assert parsed.name == "perfectionist"
    assert parsed.kind == "traits"


@pytest.mark.parametrize("name", COLLISION_NAMES)
def test_parse_entity_id_collision_round_trip(name):
    for kind in ("roles", "traits"):
        eid = entity_id(name, kind)
        parsed = parse_entity_id(eid)
        assert parsed.name == name
        assert parsed.kind == kind


def test_parse_entity_id_rejects_bare_name():
    with pytest.raises(ValueError, match="not a disambiguated id"):
        parse_entity_id("patient")


def test_parse_entity_id_rejects_extra_separator():
    with pytest.raises(ValueError, match="not a disambiguated id"):
        parse_entity_id("patient|R|extra")


def test_parse_entity_id_rejects_unknown_short_tag():
    with pytest.raises(ValueError, match="unknown kind tag"):
        parse_entity_id("patient|X")


def test_parse_entity_id_rejects_lowercase_short_tag():
    """Short tags are case-sensitive on the parse side; use the long
    form on input if that's what you have."""
    with pytest.raises(ValueError, match="unknown kind tag"):
        parse_entity_id("patient|r")


def test_parse_entity_id_rejects_empty_name():
    with pytest.raises(ValueError, match="empty name"):
        parse_entity_id("|R")


def test_parse_entity_id_rejects_non_string():
    with pytest.raises(ValueError, match="expected str"):
        parse_entity_id(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# is_entity_id()
# ---------------------------------------------------------------------------

def test_is_entity_id_positive():
    assert is_entity_id("patient|R") is True
    assert is_entity_id("patient|T") is True


def test_is_entity_id_negative():
    assert is_entity_id("patient") is False
    assert is_entity_id("patient|X") is False
    assert is_entity_id("patient|R|extra") is False
    assert is_entity_id("") is False
    assert is_entity_id("|R") is False


def test_is_entity_id_non_string():
    assert is_entity_id(42) is False
    assert is_entity_id(None) is False
    assert is_entity_id(["patient", "R"]) is False


# ---------------------------------------------------------------------------
# display_label()
# ---------------------------------------------------------------------------

def test_display_label_strips_disambiguated_id():
    assert display_label("patient|R") == "patient"
    assert display_label("patient|T") == "patient"


def test_display_label_passthrough_for_bare_name():
    assert display_label("teacher") == "teacher"


def test_display_label_passthrough_for_unknown_short_tag():
    """Unrecognised tag → not a valid eid → returned unchanged so we
    don't silently mangle data we don't understand."""
    assert display_label("patient|X") == "patient|X"


# ---------------------------------------------------------------------------
# kind_short / kind_long
# ---------------------------------------------------------------------------

def test_kind_short_idempotent_on_canonical():
    assert kind_short("R") == "R"
    assert kind_short("T") == "T"


def test_kind_long_idempotent_on_canonical():
    assert kind_long("roles") == "roles"
    assert kind_long("traits") == "traits"


def test_kind_long_from_short():
    assert kind_long("R") == "roles"
    assert kind_long("T") == "traits"


def test_kind_short_from_long():
    assert kind_short("roles") == "R"
    assert kind_short("traits") == "T"


# ---------------------------------------------------------------------------
# Use as dict keys: the original motivation
# ---------------------------------------------------------------------------

def test_collision_dict_keeps_both_kinds():
    """Smoke test for the use case the module exists to fix: stick a
    role and a trait with the same bare name into one dict and watch
    them coexist."""
    scores: dict[str, float] = {}
    for name in COLLISION_NAMES:
        scores[entity_id(name, "roles")] = 0.5
        scores[entity_id(name, "traits")] = -0.5
    assert len(scores) == 2 * len(COLLISION_NAMES)
    for name in COLLISION_NAMES:
        assert scores[entity_id(name, "roles")] == 0.5
        assert scores[entity_id(name, "traits")] == -0.5


def test_module_dunder_all_complete():
    """Anything publicly used should be in __all__ so reverse-imports
    don't surprise consumers."""
    import importlib
    import sys

    # Note: ``from assistant_axis import entity_id`` would resolve to
    # the function (re-exported in ``__init__.py``).  Resolve to the
    # actual submodule via ``sys.modules`` to avoid the shadowing.
    importlib.import_module("assistant_axis.entity_id")
    eid_mod = sys.modules["assistant_axis.entity_id"]
    expected = {
        "EntityId", "KIND_R", "KIND_T", "KIND_LONG", "ID_SEPARATOR",
        "entity_id", "parse_entity_id", "is_entity_id",
        "display_label", "kind_short", "kind_long",
        "normalize_to_file_name",
    }
    assert expected.issubset(set(eid_mod.__all__))


# ---------------------------------------------------------------------------
# normalize_to_file_name(): display-form -> file-form input hardening
# ---------------------------------------------------------------------------

from assistant_axis.entity_id import normalize_to_file_name


class TestNormalizeToFileName:
    """Boundary-input hardening: convert display-form names to the
    file-name form used as canonical keys throughout the pipeline.
    See AGENT_NOTES "File-name vs display-name convention"."""

    def test_idempotent_on_file_name_form(self):
        """File-name input passes through unchanged (the common path)."""
        for name in [
            "patient",
            "stoic",
            "aligned_artificial_intelligence",
            "systems_thinker",
            "kind_to_animals",
            "stream_of_consciousness",
            "obama_administration_health_team",
        ]:
            assert normalize_to_file_name(name) == name

    def test_spaces_to_underscores(self):
        assert normalize_to_file_name("aligned artificial intelligence") \
            == "aligned_artificial_intelligence"
        assert normalize_to_file_name("systems thinker") == "systems_thinker"

    def test_hyphens_to_underscores(self):
        assert normalize_to_file_name("systems-thinker") == "systems_thinker"
        assert normalize_to_file_name("obama-administration-health-team") \
            == "obama_administration_health_team"

    def test_lowercases(self):
        assert normalize_to_file_name("Aligned Artificial Intelligence") \
            == "aligned_artificial_intelligence"
        assert normalize_to_file_name("PATIENT") == "patient"

    def test_strips_apostrophes(self):
        """``devil's advocate`` -> ``devils_advocate`` (matches the
        actual file-name form), NOT the underscore-padded
        ``devil_s_advocate``."""
        assert normalize_to_file_name("devil's advocate") == "devils_advocate"

    def test_strips_curly_apostrophe(self):
        """U+2019 (right single quotation mark) is the default from
        word processors / smart-quote auto-replace; treat it the same
        as ASCII apostrophe."""
        assert normalize_to_file_name("devil\u2019s advocate") \
            == "devils_advocate"

    def test_collapses_runs_of_separators(self):
        """Multiple consecutive spaces / hyphens / mixes collapse
        into a single underscore."""
        assert normalize_to_file_name("multi   word") == "multi_word"
        assert normalize_to_file_name("multi - word") == "multi_word"
        assert normalize_to_file_name("a -- b -- c") == "a_b_c"

    def test_strips_outer_whitespace(self):
        assert normalize_to_file_name("  patient  ") == "patient"
        assert normalize_to_file_name("\taligned artificial intelligence\n") \
            == "aligned_artificial_intelligence"

    def test_idempotent_under_double_normalisation(self):
        """Normalising a normalised name is a no-op (important when
        boundary code can't tell whether the caller already
        normalised)."""
        for name in [
            "Aligned Artificial Intelligence",
            "devil's advocate",
            "systems-thinker",
            "  multi   word  ",
        ]:
            once = normalize_to_file_name(name)
            twice = normalize_to_file_name(once)
            assert once == twice

    def test_detection_via_inequality(self):
        """The canonical detection idiom: ``norm != raw`` -> input
        was display-form and got converted; callers should warn."""
        assert normalize_to_file_name("patient") == "patient"
        assert normalize_to_file_name("aligned artificial intelligence") \
            != "aligned artificial intelligence"
        assert normalize_to_file_name("Patient") != "Patient"

    def test_collision_names_unchanged(self):
        """The 11 collision names (Bug A) are all single-word lowercase
        already; normalise must not mangle them."""
        for name in COLLISION_NAMES:
            assert normalize_to_file_name(name) == name


# ---------------------------------------------------------------------------
# display_form_name(): file-form -> display-form for plot labels + LLM rubrics
# ---------------------------------------------------------------------------

from assistant_axis.entity_id import display_form_name


class TestDisplayFormName:
    """File-form -> display-form conversion used at every site that
    renders entity names to a human OR an LLM (plot labels, axis
    annotations, judge rubric body).  See AGENT_NOTES "File-name vs
    display-name convention"."""

    def test_underscore_to_space(self):
        assert display_form_name("aligned_artificial_intelligence") \
            == "aligned artificial intelligence"
        assert display_form_name("systems_thinker") == "systems thinker"
        assert display_form_name("devils_advocate") == "devils advocate"
        assert display_form_name("obama_administration_health_team") \
            == "obama administration health team"

    def test_idempotent_on_display_form(self):
        """Display-form input passes through unchanged (callers don't
        have to track which form they have)."""
        for s in [
            "aligned artificial intelligence",
            "systems thinker",
            "patient",
            "stoic",
        ]:
            assert display_form_name(s) == s

    def test_single_word_passthrough(self):
        """Single-word names (the common case for our v2 axes) are
        invariant: the function is a no-op for ~97 % of corpus
        entries."""
        for name in [
            "patient", "stoic", "ascetic", "harmless", "harmful",
            "helpful", "concise", "verbose",
        ]:
            assert display_form_name(name) == name

    def test_preserves_capitalisation(self):
        """Capitalisation is intentional at the display side (humans
        / LLMs decide on case); the helper does NOT lowercase."""
        assert display_form_name("Patient") == "Patient"
        assert display_form_name("System_Thinker") == "System Thinker"

    def test_disambiguated_id_passthrough(self):
        """``patient|R`` is a canonical id, not a name to render --
        passes through untouched.  Callers that want to render it
        should call ``display_label`` first to strip the suffix and
        then ``display_form_name`` for the underscore swap."""
        assert display_form_name("patient|R") == "patient|R"
        assert display_form_name("patient|T") == "patient|T"
        assert display_form_name("aligned_artificial_intelligence|R") \
            == "aligned_artificial_intelligence|R"

    def test_round_trip_via_normalize(self):
        """``normalize_to_file_name(display_form_name(x)) == x`` for
        plain underscored file-form input.  The reverse is lossy
        when the original had apostrophes (``devil's advocate`` ->
        ``devils_advocate`` -> ``devils advocate``), which is
        documented + fine for display purposes."""
        for file_form in [
            "patient",
            "aligned_artificial_intelligence",
            "systems_thinker",
            "obama_administration_health_team",
        ]:
            display = display_form_name(file_form)
            assert normalize_to_file_name(display) == file_form

    def test_collision_names_unchanged(self):
        """All 11 collision names are single-word; display-form
        rendering must not mangle them."""
        for name in COLLISION_NAMES:
            assert display_form_name(name) == name

    def test_does_not_introduce_apostrophes(self):
        """The helper is the minimal ``_ -> space`` transform; it
        does NOT re-introduce apostrophes that ``normalize_to_file_name``
        stripped (the reverse direction is lossy and that's OK)."""
        assert display_form_name("devils_advocate") == "devils advocate"
        # NOT "devil's advocate"; that would require corpus lookup

    def test_empty_string(self):
        assert display_form_name("") == ""


# ---------------------------------------------------------------------------
# corpus_display_name(): corpus-backed display form for plots / console
# ---------------------------------------------------------------------------

import json
from pathlib import Path

from assistant_axis.entity_id import (
    ROLE_DISPLAY_OVERRIDES,
    clear_corpus_display_cache,
    corpus_display_name,
    default_data_dir,
)


class TestCorpusDisplayName:
    """Lookup-backed display names (Sep 2026).  ``display_form_name``
    stays mechanical because it feeds judge rubrics; this helper is
    for human-facing text and reads the stored ``positive_label`` /
    role overrides from ``data/``.  See AGENT_NOTES "File-name vs
    display-name convention"."""

    def test_trait_positive_label_wins(self):
        assert corpus_display_name("systems_thinker") == "systems-thinker"
        assert corpus_display_name("systems_thinker", "traits") == "systems-thinker"
        assert corpus_display_name("big_picture") == "big-picture"
        assert corpus_display_name("kind_to_animals") == "kind-to-animals"

    def test_entity_id_input_supplies_kind(self):
        assert corpus_display_name("systems_thinker|T") == "systems-thinker"
        assert corpus_display_name("devils_advocate|R") == "devil's advocate"
        assert corpus_display_name("patient|R") == "patient"

    def test_role_override_and_mechanical_fallback(self):
        assert corpus_display_name("devils_advocate") == "devil's advocate"
        assert corpus_display_name("devils_advocate", "role") == "devil's advocate"
        assert corpus_display_name("aligned_artificial_intelligence") \
            == "aligned artificial intelligence"

    def test_unknown_name_falls_back_to_display_form_name(self):
        assert corpus_display_name("obama_administration_health_team") \
            == "obama administration health team"
        assert corpus_display_name("") == ""

    def test_collision_names_unchanged(self):
        for name in COLLISION_NAMES:
            assert corpus_display_name(name) == name
            assert corpus_display_name(name, "traits") == name
            assert corpus_display_name(name, "roles") == name

    def test_never_used_as_key_round_trips_whole_corpus(self):
        """``normalize_to_file_name(corpus_display_name(stem, kind)) == stem``
        for every trait and role file on disk -- the one direction the
        convention guarantees (stem -> display is lossy, display ->
        stem is not)."""
        data = default_data_dir()
        for kind, sub in (("traits", "traits"), ("roles", "roles")):
            stems = [p.stem for p in (data / sub / "instructions").glob("*.json")]
            assert len(stems) > 100, f"corpus not found under {data}"
            bad = [s for s in stems
                   if normalize_to_file_name(corpus_display_name(s, kind)) != s]
            assert bad == [], bad

    def test_standard_suffixed_label_round_trips(self, tmp_path: Path):
        """The Sep-2026 naming convention for traits imported from a
        named standard: label ``<pole> <Standard>`` with capitals and
        an internal hyphen, stem = ``normalize_to_file_name(label)``.
        The lookup must return the label verbatim and the round trip
        must hold, using an isolated data dir so the cache for the
        real corpus is untouched."""
        tdir = tmp_path / "traits" / "instructions"; tdir.mkdir(parents=True)
        (tmp_path / "roles" / "instructions").mkdir(parents=True)
        label = "traditional (Inglehart-Welzel)"
        stem = normalize_to_file_name(label)
        assert stem == "traditional_inglehart_welzel"
        (tdir / f"{stem}.json").write_text(json.dumps({
            "positive_label": label, "negative_label": "secular-rational Inglehart-Welzel",
            "description": "x", "instruction": [], "questions": [], "eval_prompt": "x",
        }))
        try:
            assert corpus_display_name(stem, data_dir=tmp_path) == label
            assert corpus_display_name(f"{stem}|T", data_dir=tmp_path) == label
            assert normalize_to_file_name(corpus_display_name(stem, data_dir=tmp_path)) == stem
            # mechanical helper stays lossy on purpose (prompt-stable)
            assert display_form_name(stem) == "traditional inglehart welzel"
        finally:
            clear_corpus_display_cache()

    def test_missing_data_dir_degrades_to_mechanical(self, tmp_path: Path):
        try:
            assert corpus_display_name("systems_thinker", data_dir=tmp_path / "nope") \
                == "systems thinker"
        finally:
            clear_corpus_display_cache()

    def test_overrides_in_sync_with_data_analysis(self):
        from data_analysis.regenerate_role_instructions import _ROLE_NAME_OVERRIDES
        assert ROLE_DISPLAY_OVERRIDES == _ROLE_NAME_OVERRIDES

    def test_display_form_name_unchanged_by_lookup(self):
        """Regression guard: the prompt-stable helper must NOT start
        consulting the corpus (that would be a silent rubric change)."""
        assert display_form_name("systems_thinker") == "systems thinker"
        assert display_form_name("devils_advocate") == "devils advocate"


# ---------------------------------------------------------------------------
# normalize_to_file_name(): ASCII folding (Sep 2026)
# ---------------------------------------------------------------------------


class TestNormalizeToFileNameAsciiFold:
    """Diacritics fold to ASCII so stems stay ASCII on every filesystem;
    the label keeps them (``Gemeinschaft Tönnies`` -> ``gemeinschaft_tonnies``).
    See AGENT_NOTES "Standard-derived trait labels", rule 2."""

    def test_umlaut_in_standard_suffix(self):
        assert normalize_to_file_name("Gemeinschaft (Tönnies)") == "gemeinschaft_tonnies"
        assert normalize_to_file_name("Gesellschaft (Tönnies)") == "gesellschaft_tonnies"
        # Idempotent on the folded result.
        assert normalize_to_file_name("gemeinschaft_tonnies") == "gemeinschaft_tonnies"

    def test_accents_ligatures_and_special_letters(self):
        assert normalize_to_file_name("Bahá'í") == "bahai"
        assert normalize_to_file_name("naïve") == "naive"
        assert normalize_to_file_name("Straße") == "strasse"
        assert normalize_to_file_name("Ærø") == "aero"
        assert normalize_to_file_name("Łódź") == "lodz"

    def test_nfc_and_nfd_input_fold_identically(self):
        import unicodedata
        nfc = "Tönnies"
        nfd = unicodedata.normalize("NFD", nfc)
        assert nfc != nfd
        assert normalize_to_file_name(nfc) == normalize_to_file_name(nfd) == "tonnies"

    def test_en_and_em_dashes_become_underscores(self):
        assert normalize_to_file_name("Inglehart\u2013Welzel") == "inglehart_welzel"
        assert normalize_to_file_name("self\u2014expression") == "self_expression"

    def test_ascii_input_unchanged(self):
        for name in ["patient", "systems_thinker", "traditional_inglehart_welzel"]:
            assert normalize_to_file_name(name) == name


class TestNormalizeToFileNameParentheses:
    """Sep 2026 label convention: the standard is a parenthesised suffix in
    the label (``open (Big Five)``) and absent from the stem."""

    def test_parenthesised_standard(self):
        assert normalize_to_file_name("open (Big Five)") == "open_big_five"
        assert normalize_to_file_name("honest-humble (HEXACO)") == "honest_humble_hexaco"
        assert normalize_to_file_name("secular-rational (Inglehart-Welzel)") == "secular_rational_inglehart_welzel"
        assert normalize_to_file_name("the fool (Tarot)") == "the_fool_tarot"

    def test_no_stray_underscores(self):
        assert normalize_to_file_name("open ( Big Five )") == "open_big_five"
        assert normalize_to_file_name("(Big Five) open") == "big_five_open"
        assert normalize_to_file_name("blood type [A]") == "blood_type_a"

    def test_old_unparenthesised_form_gives_same_stem(self):
        assert normalize_to_file_name("open Big Five") == normalize_to_file_name("open (Big Five)")


# ---------------------------------------------------------------------------
# resolve_renamed_stem: old stem -> the stem the corpus uses now (Sep 2026)
# ---------------------------------------------------------------------------

from assistant_axis.entity_id import resolve_renamed_stem  # noqa: E402


class TestResolveRenamedStem:
    """An older pair list or steering config names an entity by the stem it
    had when that file was written; the corpus file records the old stem in
    ``renamed_from``.  Isolated data dirs, so the real corpus's cache is
    untouched."""

    @staticmethod
    def _corpus(root: Path, kind: str, files: dict) -> Path:
        d = root / kind / "instructions"; d.mkdir(parents=True, exist_ok=True)
        for stem, doc in files.items():
            (d / f"{stem}.json").write_text(json.dumps({"description": "x", **doc}))
        return root

    def test_renamed_role_resolves_and_kind_aliases_work(self, tmp_path: Path):
        self._corpus(tmp_path, "roles", {
            "instrumentally_aligned_ai": {"renamed_from": {"stem": "aligned_artificial_intelligence", "date": "2026-09-28"}},
            "paperclip_maximizer": {},
        })
        try:
            for kind in ("roles", "role", "R"):
                assert resolve_renamed_stem("aligned_artificial_intelligence", kind, data_dir=tmp_path) \
                    == "instrumentally_aligned_ai"
            assert resolve_renamed_stem("paperclip_maximizer", "roles", data_dir=tmp_path) == "paperclip_maximizer"
        finally:
            clear_corpus_display_cache()

    def test_existing_file_wins_over_a_recorded_rename(self, tmp_path: Path):
        """A stem that was freed by a rename and later reused keeps its own file."""
        self._corpus(tmp_path, "traits", {"kind": {}, "generous": {"renamed_from": "kind"}})
        try:
            assert resolve_renamed_stem("kind", "traits", data_dir=tmp_path) == "kind"
        finally:
            clear_corpus_display_cache()

    def test_dict_bare_string_and_list_forms(self, tmp_path: Path):
        """The three shapes ``renamed_from`` takes.  An entity renamed more
        than once needs the list form to keep its earlier stems: the record
        lives on the current file, so a stem left out of it is lost."""
        self._corpus(tmp_path, "traits", {
            "immune": {"renamed_from": {"stem": "resistant", "date": "2026-09-26"}},
            "lazy": {"renamed_from": "slothful"},
            "easygoing": {"renamed_from": [{"stem": "chill"}, "mellow"]},
        })
        try:
            assert resolve_renamed_stem("resistant", "traits", data_dir=tmp_path) == "immune"
            assert resolve_renamed_stem("slothful", "traits", data_dir=tmp_path) == "lazy"
            assert resolve_renamed_stem("chill", "traits", data_dir=tmp_path) == "easygoing"
            assert resolve_renamed_stem("mellow", "traits", data_dir=tmp_path) == "easygoing"
            assert resolve_renamed_stem("avoidant", "traits", data_dir=tmp_path) == "avoidant"
        finally:
            clear_corpus_display_cache()

    def test_unknown_stem_and_missing_dir_are_returned_unchanged(self, tmp_path: Path):
        self._corpus(tmp_path, "traits", {"calm": {}})
        try:
            assert resolve_renamed_stem("never_existed", "traits", data_dir=tmp_path) == "never_existed"
            assert resolve_renamed_stem("calm", "roles", data_dir=tmp_path) == "calm"
            assert resolve_renamed_stem("calm", "traits", data_dir=tmp_path / "nope") == "calm"
        finally:
            clear_corpus_display_cache()

    def test_kind_none_looks_in_both_kinds_traits_first(self, tmp_path: Path):
        """A steering config's pole names carry no kind (housekeeping item 5 (a))."""
        self._corpus(tmp_path, "traits", {"lazy": {"renamed_from": "slothful"},
                                          "twice": {"renamed_from": "both"}})
        self._corpus(tmp_path, "roles", {
            "instrumentally_aligned_ai": {"renamed_from": "aligned_artificial_intelligence"},
            "slothful": {}, "twice_role": {"renamed_from": "both"}})
        try:
            assert resolve_renamed_stem("aligned_artificial_intelligence", None, data_dir=tmp_path) \
                == "instrumentally_aligned_ai"
            # a file of either kind keeps its stem
            assert resolve_renamed_stem("slothful", None, data_dir=tmp_path) == "slothful"
            assert resolve_renamed_stem("lazy", None, data_dir=tmp_path) == "lazy"
            # an old stem recorded by both kinds goes to the trait
            assert resolve_renamed_stem("both", None, data_dir=tmp_path) == "twice"
            assert resolve_renamed_stem("free text (HEXACO)", None, data_dir=tmp_path) \
                == "free text (HEXACO)"
        finally:
            clear_corpus_display_cache()

    def test_kinds_do_not_leak_into_each_other(self, tmp_path: Path):
        self._corpus(tmp_path, "traits", {"aggressive": {"renamed_from": {"stem": "militant"}}})
        self._corpus(tmp_path, "roles", {"soldier": {}})
        try:
            assert resolve_renamed_stem("militant", "traits", data_dir=tmp_path) == "aggressive"
            assert resolve_renamed_stem("militant", "roles", data_dir=tmp_path) == "militant"
        finally:
            clear_corpus_display_cache()

    def test_real_corpus_renamed_pole(self):
        """The one judged pole renamed so far (2026-09-28)."""
        assert resolve_renamed_stem("aligned_artificial_intelligence", "roles") == "instrumentally_aligned_ai"
        assert resolve_renamed_stem("paperclip_maximizer", "roles") == "paperclip_maximizer"

    def test_unknown_kind_raises(self, tmp_path: Path):
        with pytest.raises(ValueError):
            resolve_renamed_stem("calm", "axes", data_dir=tmp_path)


# ---------------------------------------------------------------------------
# judge_label / judge_negative_label / judge_form_of_label (W19, 2026-10-09)
# ---------------------------------------------------------------------------

import re  # noqa: E402
import unicodedata  # noqa: E402

from assistant_axis.entity_id import (  # noqa: E402
    JUDGE_LABEL_FORM,
    STANDARD_SUFFIX_FORMS,
    judge_form_of_label,
    judge_form_of_negative_label,
    judge_label,
    judge_negative_label,
)

_SUFFIX = re.compile(r"\(([^()]*)\)\s*$")


def _corpus_traits() -> dict:
    tdir = default_data_dir() / "traits" / "instructions"
    out = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in tdir.glob("*.json")}
    assert len(out) > 100, f"corpus not found under {tdir}"
    return out


def _corpus_role_stems() -> list:
    stems = [p.stem for p in (default_data_dir() / "roles" / "instructions").glob("*.json")]
    assert len(stems) > 100
    return stems


def _capitalised_suffix(label: str):
    m = _SUFFIX.search(label or "")
    if m and m.group(1).strip()[:1].isupper():
        return unicodedata.normalize("NFC", m.group(1).strip())
    return None


class TestJudgeFormOfLabel:
    """The string-level suffix rewrite, and the table Roger left to the
    main agent (2026-10-09)."""

    def test_table_is_the_agreed_one(self):
        assert STANDARD_SUFFIX_FORMS == {
            "HEXACO": "HEXACO", "VALS": "VALS", "DISC": "DISC", "VARK": "VARK",
            "Tönnies": "Tönnies",
            "Big Five": "the Big Five", "MBTI": "the MBTI", "Enneagram": "the Enneagram",
            "BFAS": "the BFAS", "IPIP-NEO": "the IPIP-NEO", "Tarot": "the Tarot",
            "Dark Tetrad": "the Dark Tetrad", "Light Triad": "the Light Triad",
            "Holland": "Holland's RIASEC", "Bartle": "Bartle's player types",
            "Baumrind": "Baumrind's parenting styles", "Hall": "Edward Hall",
            "Kohlberg": "Kohlberg's stages", "Allport": "Allport's religious orientation",
            "Gelfand": "Gelfand's tight and loose cultures",
            "Inglehart-Welzel": "the Inglehart-Welzel map",
        }
        assert JUDGE_LABEL_FORM == "judge-display-v1"

    @pytest.mark.parametrize("label, expected", [
        ("careless (HEXACO)", "careless (from HEXACO)"),
        ("Organization (HEXACO)", "Organization (from HEXACO)"),
        ("Intellect (BFAS)", "Intellect (from the BFAS)"),
        ("open (Big Five)", "open (from the Big Five)"),
        ("ENFJ (MBTI)", "ENFJ (from the MBTI)"),
        ("achiever (VALS)", "achiever (from VALS)"),
        ("artistic (Holland)", "artistic (from Holland's RIASEC)"),
        ("high-context (Hall)", "high-context (from Edward Hall)"),
        ("secular-rational (Inglehart-Welzel)", "secular-rational (from the Inglehart-Welzel map)"),
        ("Gemeinschaft (Tönnies)", "Gemeinschaft (from Tönnies)"),
        ("the fool (Tarot)", "the fool (from the Tarot)"),
        ("non-careless (HEXACO)", "non-careless (from HEXACO)"),
        # an unlisted standard takes the default form
        ("cultural-capital (Bourdieu)", "cultural-capital (from Bourdieu)"),
        # not a standard: lower-case suffix, or no suffix at all
        ("humorous/serious (tentative, PC10)", "humorous/serious (tentative, PC10)"),
        ("systems-thinker", "systems-thinker"),
        ("devil's advocate", "devil's advocate"),
        ("", ""),
    ])
    def test_rewrite(self, label, expected):
        assert judge_form_of_label(label) == expected

    def test_idempotent(self):
        for label in ("careless (HEXACO)", "artistic (Holland)", "open (Big Five)", "plain"):
            once = judge_form_of_label(label)
            assert judge_form_of_label(once) == once

    def test_decomposed_umlaut_finds_the_table_entry(self):
        nfd = unicodedata.normalize("NFD", "Gemeinschaft (Tönnies)")
        assert unicodedata.normalize("NFC", judge_form_of_label(nfd)) == "Gemeinschaft (from Tönnies)"


class TestJudgeLabel:
    """The judge display form of every corpus entity (whole corpus)."""

    def test_examples(self):
        assert judge_label("careless_hexaco") == "careless (from HEXACO)"
        assert judge_label("careless_hexaco|T") == "careless (from HEXACO)"
        assert judge_label("enfj_mbti", "traits") == "ENFJ (from the MBTI)"
        assert judge_label("gemeinschaft_tonnies") == "Gemeinschaft (from Tönnies)"
        assert judge_label("systems_thinker") == "systems-thinker"
        assert judge_label("devils_advocate", "roles") == "devil's advocate"
        assert judge_label("instrumentally_aligned_ai|R") == "instrumentally-aligned AI"
        assert judge_label("coral_reef", "roles") == "coral reef"

    def test_unknown_names_fall_back_to_the_mechanical_form(self):
        # a renamed pole's old stem, a stem with no file: exactly as under v3
        assert judge_label("aligned_artificial_intelligence") == "aligned artificial intelligence"
        assert judge_label("careless_hexaco", "roles") == "careless hexaco"
        assert judge_label("") == ""
        # free text from a config is rewritten too
        assert judge_label("careless (HEXACO)") == "careless (from HEXACO)"

    def test_role_override_applies_without_a_file(self, tmp_path: Path):
        try:
            assert judge_label("devils_advocate", "roles", data_dir=tmp_path / "nope") == "devil's advocate"
        finally:
            clear_corpus_display_cache()

    def test_every_trait_renders_from_its_label(self):
        bad = []
        for stem, doc in _corpus_traits().items():
            label = doc["positive_label"]
            got = judge_label(stem, "traits")
            if got != judge_form_of_label(label) or "_" in got:
                bad.append((stem, label, got))
                continue
            # capitals, hyphens and diacritics come from the label: the label
            # minus its suffix is kept verbatim
            m = _SUFFIX.search(label)
            head = label[:m.start()].rstrip() if m and _capitalised_suffix(label) else label
            if not got.startswith(head):
                bad.append((stem, label, got))
        assert bad == [], bad[:10]

    def test_every_role_renders_like_the_role_generator(self):
        from data_analysis.regenerate_role_instructions import role_display_name
        bad = [(s, judge_label(s, "roles")) for s in _corpus_role_stems()
               if judge_label(s, "roles") != judge_form_of_label(role_display_name(s))
               or "_" in judge_label(s, "roles")]
        assert bad == [], bad[:10]

    def test_every_capitalised_suffix_in_the_corpus_is_in_the_table(self):
        found = set()
        for doc in _corpus_traits().values():
            for key in ("positive_label", "negative_label"):
                s = _capitalised_suffix(doc.get(key))
                if s:
                    found.add(s)
        assert found, "expected the standards-derived traits"
        assert found - set(STANDARD_SUFFIX_FORMS) == set()

    def test_every_capitalised_suffix_in_the_live_queue_is_in_the_table(self):
        """Entries not adopted or superseded may carry a suffix with the
        default form (Bourdieu); everything else must be decided here."""
        queue = json.loads((default_data_dir() / "seed_queue.json").read_text(encoding="utf-8"))
        missing = {}
        for e in queue["entries"]:
            if e.get("status") in ("not_adopted", "superseded"):
                continue
            s = _capitalised_suffix(e.get("label"))
            if s and s not in STANDARD_SUFFIX_FORMS:
                missing.setdefault(s, []).append(e["stem"])
        assert missing == {}, missing


class TestJudgeNegativeLabel:
    def test_examples(self):
        assert judge_negative_label("careless_hexaco") == "conscientious (from HEXACO)"
        assert judge_negative_label("enfj_mbti") == "non-ENFJ (from the MBTI)"
        assert judge_negative_label("open_big_five|T") == "closed (from the Big Five)"

    def test_roles_and_unknown_stems_raise(self):
        with pytest.raises(KeyError):
            judge_negative_label("devils_advocate|R")
        with pytest.raises(KeyError):
            judge_negative_label("no_such_trait_at_all")

    def test_string_level_rules(self, tmp_path: Path):
        tdir = tmp_path / "traits" / "instructions"; tdir.mkdir(parents=True)
        (tdir / "systems_thinker.json").write_text(json.dumps(
            {"positive_label": "systems-thinker", "negative_label": "non-systems-thinker"}))
        try:
            # 1. names a corpus trait: that trait's own label, whatever the spelling
            assert judge_form_of_negative_label("systems thinker", "x", data_dir=tmp_path) == "systems-thinker"
            # 2. the trait's own non-X placeholder
            assert judge_form_of_negative_label("non-careless (HEXACO)", "careless (HEXACO)",
                                                data_dir=tmp_path) == "non-careless (from HEXACO)"
            # 3. anything else: the stored string (suffix rewrite only)
            assert judge_form_of_negative_label("academic", "experiential", data_dir=tmp_path) == "academic"
            assert judge_form_of_negative_label("", "x", data_dir=tmp_path) == ""
        finally:
            clear_corpus_display_cache()

    def test_every_trait_renders(self):
        traits = _corpus_traits()
        bad = []
        for stem, doc in traits.items():
            got = judge_negative_label(stem)
            neg = doc["negative_label"]
            partner = normalize_to_file_name(neg)
            if "_" in got:
                bad.append((stem, got))
            elif partner in traits:
                if got != judge_label(partner, "traits"):
                    bad.append((stem, got))
            elif neg == f"non-{doc['positive_label']}":
                if got != "non-" + judge_label(stem, "traits"):
                    bad.append((stem, got))
            elif got != judge_form_of_label(neg):
                bad.append((stem, got))
        assert bad == [], bad[:10]


def test_judge_helpers_are_exported():
    import importlib
    import sys
    importlib.import_module("assistant_axis.entity_id")
    mod = sys.modules["assistant_axis.entity_id"]
    assert {"STANDARD_SUFFIX_FORMS", "JUDGE_LABEL_FORM", "judge_form_of_label", "judge_label",
            "judge_negative_label", "judge_form_of_negative_label"} <= set(mod.__all__)
