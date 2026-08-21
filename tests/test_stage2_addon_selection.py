"""
Choosing which add-on outputs join the group file.

The join used to be unconditional: the only way to keep an add-on out of a
manuscript's group table was to move the file. These cover the switch, the
per-add-on picker, and the one rule that makes the picker trustworthy --
that it offers exactly what the join would accept.
"""

import json
import os
import pathlib

import pandas as pd
import pytest

from mep_cmap import stage2
from mep_cmap.stage2 import (_s2_addon_tag, _s2_core_prefix,
                             _s2_discover_addons, _s2_is_addon_file,
                             _s2_join_addon_sidecars)

PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
PREFIX = "sub-01_ses-1"


@pytest.fixture
def session(tmp_path):
    """A results tree with a core trials file and two add-on outputs."""
    res = tmp_path / "results"
    (res / "trial-level").mkdir(parents=True)
    (res / "add-ons").mkdir(parents=True)

    core = pd.DataFrame({
        "File": ["rec1"] * 3,
        "StimType": ["A", "A", "B"],
        "Segment": [1, 2, 1],
        "Condition": ["pre", "pre", "pre"],
        "PTP(mV)": [1.2, 1.4, 0.8],
    })
    core_path = res / "trial-level" / f"{PREFIX}_trials.csv"
    core.to_csv(core_path, index=False)

    for tag, col in (("temporal_decomposition", "Bin1"),
                     ("mepfeatx", "Sharpness")):
        pd.DataFrame({
            "StimType": ["A", "A", "B"],
            "Segment": [1, 2, 1],
            "Condition": ["pre", "pre", "pre"],
            col: [1.0, 2.0, 3.0],
        }).to_csv(res / "add-ons" / f"{PREFIX}_{tag}.csv", index=False)

    return core, str(core_path)


# ── the shared naming rule ───────────────────────────────────────────────────

def test_both_core_names_give_the_same_prefix():
    """The sidecars are named after the prefix, so it must not depend on which
    core table the merge was built from."""
    assert _s2_core_prefix(f"{PREFIX}_trials.csv") == PREFIX
    assert _s2_core_prefix(f"{PREFIX}_trials_selected.csv") == PREFIX


def test_a_non_core_file_has_no_prefix():
    assert _s2_core_prefix(f"{PREFIX}_summary.csv") is None


def test_core_outputs_are_not_add_ons():
    for name in ("trials.csv", "trials_selected.csv", "summary.csv",
                 "onset_methods.csv", "trials_with_outliers.csv"):
        assert not _s2_is_addon_file(f"{PREFIX}_{name}")


def test_the_shipped_demonstration_is_not_offered():
    """rectified_area exists to show an add-on author the shape. Its numbers
    must not reach a group table because someone clicked it once."""
    assert not _s2_is_addon_file(f"{PREFIX}_rectified_area.csv")


def test_the_tag_rule_is_shared_by_picker_and_join():
    """If the picker derived names differently it would offer add-ons that
    never join and hide ones that do."""
    src = (PKG / "stage2.py").read_text(encoding="utf-8")
    body = src[src.index("def _s2_join_addon_sidecars"):]
    body = body[:body.index("\nclass ")]
    assert "tag = _s2_addon_tag(fn, prefix)" in body
    assert _s2_addon_tag(f"{PREFIX}_mepfeatx.csv", PREFIX) == "mepfeatx"


# ── discovery ────────────────────────────────────────────────────────────────

def test_discovery_finds_what_is_on_disk(session):
    _core, core_path = session
    counts = _s2_discover_addons([{"_trials_csv": core_path}])
    assert counts == {"temporal_decomposition": 1, "mepfeatx": 1}


def test_discovery_counts_sessions_not_files(session, tmp_path):
    """The picker reports coverage, so an add-on present in only some sessions
    can be seen before it is chosen."""
    _core, core_path = session
    res2 = tmp_path / "s2" / "results"
    (res2 / "trial-level").mkdir(parents=True)
    (res2 / "add-ons").mkdir(parents=True)
    p2 = "sub-02_ses-1"
    pd.DataFrame({"StimType": ["A"], "Segment": [1]}).to_csv(
        res2 / "trial-level" / f"{p2}_trials.csv", index=False)
    pd.DataFrame({"StimType": ["A"], "Segment": [1], "Bin1": [1.0]}).to_csv(
        res2 / "add-ons" / f"{p2}_temporal_decomposition.csv", index=False)

    counts = _s2_discover_addons([
        {"_trials_csv": core_path},
        {"_trials_csv": str(res2 / "trial-level" / f"{p2}_trials.csv")},
    ])
    assert counts["temporal_decomposition"] == 2
    assert counts["mepfeatx"] == 1


def test_discovery_ignores_core_and_example_outputs(session, tmp_path):
    _core, core_path = session
    res = pathlib.Path(core_path).parent.parent
    pd.DataFrame({"a": [1]}).to_csv(
        res / "add-ons" / f"{PREFIX}_rectified_area.csv", index=False)
    pd.DataFrame({"a": [1]}).to_csv(
        res / "summary" / f"{PREFIX}_summary.csv", index=False) \
        if (res / "summary").exists() else None
    counts = _s2_discover_addons([{"_trials_csv": core_path}])
    assert "rectified_area" not in counts


def test_discovery_survives_a_session_with_no_results(tmp_path):
    counts = _s2_discover_addons([{"_trials_csv": str(tmp_path / "nope.csv")}])
    assert counts == {}


# ── the join honours the selection ───────────────────────────────────────────

def _join(core, path, allowed):
    notes = []
    out = _s2_join_addon_sidecars(core.copy(), path, notes.append,
                                  allowed=allowed)
    return out, notes


def test_none_means_every_add_on(session):
    core, path = session
    out, _notes = _join(core, path, None)
    assert "Bin1" in out.columns and "Sharpness" in out.columns


def test_a_selection_joins_only_what_it_names(session):
    core, path = session
    out, _notes = _join(core, path, {"mepfeatx"})
    assert "Sharpness" in out.columns
    assert "Bin1" not in out.columns


def test_an_empty_selection_is_not_none(session):
    """An empty set means "no add-ons"; None means "all of them". Collapsing
    them would make "none" silently join everything."""
    core, path = session
    out, _notes = _join(core, path, set())
    assert "Bin1" not in out.columns and "Sharpness" not in out.columns
    assert list(out.columns) == list(core.columns)


def test_the_join_is_still_additive(session):
    """Core measurements are never touched, whatever is selected."""
    core, path = session
    out, _notes = _join(core, path, None)
    assert out["PTP(mV)"].tolist() == core["PTP(mV)"].tolist()
    assert len(out) == len(core)


def test_excluding_an_add_on_is_silent(session):
    """A note per excluded add-on per session would bury the ones that mean
    something."""
    core, path = session
    _out, notes = _join(core, path, {"mepfeatx"})
    assert not any("temporal_decomposition" in n for n in notes)


# ── the switch and its persistence ───────────────────────────────────────────

def _stage2_src():
    return (PKG / "stage2.py").read_text(encoding="utf-8")


def test_add_ons_are_on_by_default():
    """Every study built before this switch existed had them joined."""
    src = _stage2_src()
    assert "self._s2_addons_var = tk.BooleanVar(value=True)" in src
    assert "self._s2_addon_allow = None" in src


def test_the_join_is_skipped_not_undone_when_switched_off():
    """Joining and then dropping would let an unreadable sidecar fail a run
    that never wanted it."""
    src = _stage2_src()
    body = src[src.index("def _s2_run"):]
    i_guard = body.index("if self._s2_addons_var.get():")
    i_join = body.index("_s2_join_addon_sidecars(")
    assert i_guard < i_join


def test_the_design_file_records_what_the_table_contains():
    """A study rebuilt with a different source or add-on set is a different
    table, and nothing would have said which produced the manuscript."""
    src = _stage2_src()
    body = src[src.index("def _s2_save_design"):]
    body = body[:body.index("def _s2_load_design")]
    for key in ('"column_source"', '"include_addons"', '"addon_allow"'):
        assert key in body


def test_an_old_design_loads_as_the_defaults_it_was_built_under():
    """Absent keys must mean Full columns and add-ons on, not off."""
    src = _stage2_src()
    body = src[src.index("def _s2_load_design"):]
    assert 'design.get("include_addons", True)' in body
    assert 'else "Full"' in body


def test_the_old_source_name_still_loads():
    """The option was called "Selected" when the design file first learned to
    store it. A design saved then must rebuild the same table, not revert to
    Full without saying so."""
    src = _stage2_src()
    body = src[src.index("def _s2_load_design"):]
    assert '_src == "Selected"' in body
    assert '_src = "Trimmed"' in body


def test_the_source_value_is_spelt_the_same_everywhere():
    """The combobox value is compared in one place and persisted in another;
    a mismatch would silently always read the full file."""
    src = _stage2_src()
    assert 'values=["Full", "Trimmed"]' in src
    body = src[src.index("def _s2_resolve_source"):]
    assert '!= "Trimmed"' in body[:1200]


def test_everything_ticked_is_stored_as_none_not_as_a_list():
    """Under None an add-on run after the design was saved still joins; under
    an explicit list it silently would not."""
    src = _stage2_src()
    body = src[src.index("def _s2_choose_addons"):]
    body = body[:body.index("def _s2_resolve_source")]
    assert "None if chosen == set(counts) else chosen" in body


def test_the_report_distinguishes_excluded_from_none_found():
    """"No add-on columns" is otherwise ambiguous between "none were there"
    and "you turned them off"."""
    src = _stage2_src()
    body = src[src.index("def _s2_run"):]
    assert "Add-on columns: excluded." in body
