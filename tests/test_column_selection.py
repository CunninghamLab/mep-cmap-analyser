"""
Column selection for the narrowed trial file.

_trials.csv carries every column, always. Selection produces a SECOND file,
_trials_selected.csv, beside it. These tests exist mainly to make one failure
mode impossible: a metric appended to LAT_COLS without being assigned to a
group would otherwise be absent from every narrowed file ever written, with
nothing to say so.
"""

import ast
import pathlib

import pandas as pd
import pytest

from mep_cmap import column_groups as cg
from mep_cmap import results_layout as rl

PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"


def _lat_cols():
    """LAT_COLS parsed from source.

    pipeline.py is importable, but parsing keeps this test honest about the
    written schema rather than about whatever a running import happens to
    hold, and matches how the other schema tests read it.
    """
    tree = ast.parse((PKG / "pipeline.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and \
                any(getattr(t, "id", "") == "LAT_COLS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("LAT_COLS not found in pipeline.py")


def _written_columns():
    """Every column the trial frame has once written, in order."""
    return list(_lat_cols()) + list(cg.POST_BUILD_COLUMNS)


# ── coverage: the whole point ────────────────────────────────────────────────

def test_every_written_column_is_assigned():
    """A metric added to LAT_COLS without a group FAILS here.

    The alternative is that it silently vanishes from every narrowed file,
    which nobody would notice until an analysis came up a column short.
    """
    missing = cg.unassigned(_written_columns())
    assert not missing, (
        "these columns are in no group and not protected, so they would "
        f"disappear from _trials_selected.csv: {missing}")


def test_no_column_is_claimed_twice():
    """Two groups claiming one column makes the group list a description of
    what is offered rather than of what is written."""
    assert not cg.duplicated()


def test_no_group_names_a_column_that_does_not_exist():
    """A renamed column would otherwise leave a group quietly selecting
    nothing."""
    written = set(_written_columns())
    known = set(cg.PROTECTED) | {c for _k, _l, cs in cg.GROUPS for c in cs}
    assert not (known - written)


def test_channel_is_protected_but_not_in_lat_cols():
    """It is added by _tag_channel after the rows are built, so defining the
    protected set against LAT_COLS would silently omit it."""
    assert "Channel" in cg.PROTECTED
    assert "Channel" not in _lat_cols()
    assert "Channel" in cg.POST_BUILD_COLUMNS


def test_the_protected_set_can_rejoin_the_two_files():
    """The property the whole feature rests on: nothing is lost by narrowing,
    because the narrow file can always be merged back against the full one."""
    for key in ("File", "Channel", "StimType", "Segment", "Condition"):
        assert key in cg.PROTECTED


def test_every_protected_column_says_why():
    """It is shown ticked and disabled; a control that cannot be changed and
    gives no reason reads as a bug."""
    for name, why in cg.PROTECTED.items():
        assert why and len(why) > 10, f"{name}: no reason given"


# ── dependencies ─────────────────────────────────────────────────────────────

def test_carson_qr_pulls_in_the_reference_group():
    """Normalised_Adjusted_PTP_QR is a ratio to a reference mean that can
    differ BETWEEN STIM TYPES. Without Reference_Type the column mixes
    denominators while looking uniform."""
    keys, pulled = cg.resolve(["carson_qr"])
    assert "reference" in keys
    assert any(d == "carson_qr" and r == "reference" for d, r, _why in pulled)


def test_a_pulled_dependency_carries_a_reason():
    """It is reported in the run log; "also added reference" explains
    nothing."""
    _keys, pulled = cg.resolve(["carson_qr"])
    for _d, _r, why in pulled:
        assert why and len(why) > 20


def test_resolving_an_already_complete_selection_pulls_nothing():
    _keys, pulled = cg.resolve(["carson_qr", "reference"])
    assert pulled == []


def test_amplitude_is_deselectable():
    """Deliberately NOT a dependency of anything. Pulling it in everywhere
    would make raw amplitude impossible to drop, which is the case the
    feature exists for."""
    for key in cg.GROUP_KEYS:
        assert cg.DEPENDENCIES.get(key) != "amplitude"
    keys, _pulled = cg.resolve(["within_file_z", "reference", "detrended"])
    assert "amplitude" not in keys


def test_every_dependency_names_a_real_group():
    for dependent, required in cg.DEPENDENCIES.items():
        assert dependent in cg.GROUP_KEYS
        assert required in cg.GROUP_KEYS


# ── selection ────────────────────────────────────────────────────────────────

def test_selection_keeps_schema_order_not_tick_order():
    """Two sessions that chose the same groups must produce the same column
    order however the boxes were ticked, or their files look different when
    they are not."""
    written = _written_columns()
    a = cg.select(written, ["onset", "amplitude"])
    b = cg.select(written, ["amplitude", "onset"])
    assert a == b
    assert a == [c for c in written if c in set(a)]


def test_an_empty_selection_still_writes_the_protected_columns():
    """Distinct from writing no file at all: it is a real choice, and the
    result is still a joinable table."""
    cols = cg.select(_written_columns(), [])
    assert set(cols) == set(cg.PROTECTED)


def test_selection_never_invents_a_column():
    written = _written_columns()
    assert set(cg.select(written, cg.GROUP_KEYS)).issubset(set(written))


def test_selecting_everything_keeps_everything():
    assert cg.select(_written_columns(), cg.GROUP_KEYS) == _written_columns()


def test_an_unknown_group_key_selects_nothing_extra():
    """A group retired in a later version must not raise mid-run."""
    cols = cg.select(_written_columns(), ["no_such_group"])
    assert set(cols) == set(cg.PROTECTED)


# ── layout routing ───────────────────────────────────────────────────────────

def test_the_narrowed_file_is_routed_beside_the_full_one():
    assert rl.family_for("sub-01_ses-1_trials_selected.csv") == "trial-level"
    assert rl.family_for("sub-01_ses-1_trials.csv") == "trial-level"


def test_the_longest_suffix_wins_over_trials_csv():
    """'_trials_selected.csv' does not end with '_trials.csv', but both are in
    FAMILIES and the prefix must come out the same from either."""
    assert "trials_selected.csv" in rl.FAMILIES
    got = rl.sibling("/x/results/trial-level/sub-01_trials_selected.csv",
                     "summary.csv")
    assert got.endswith("sub-01_summary.csv")


def test_stage2_treats_the_narrowed_file_as_a_core_output():
    """Derived from FAMILIES, so it must not need a second hand-edit. If it
    were missing, the sidecar join would mistake it for an add-on output and
    try to join the core table to itself."""
    from mep_cmap.stage2 import _S2_CORE_SUFFIXES
    assert "_trials_selected.csv" in _S2_CORE_SUFFIXES


def test_the_addon_join_accepts_either_core_table():
    """A Selected run returned early and silently joined NO add-on outputs,
    because the prefix was only derived from '_trials.csv'.

    Asserted on behaviour rather than on where the literal lives: the rule was
    later factored into _s2_core_prefix so the add-on picker could share it,
    and a test pinned to the old location would have failed a refactor that
    changed nothing.
    """
    from mep_cmap.stage2 import _s2_core_prefix
    assert _s2_core_prefix("sub-01_ses-1_trials.csv") == "sub-01_ses-1"
    assert _s2_core_prefix("sub-01_ses-1_trials_selected.csv") == "sub-01_ses-1"

    src = (PKG / "stage2.py").read_text(encoding="utf-8")
    body = src[src.index("def _s2_join_addon_sidecars"):]
    body = body[:body.index("\nclass ")]
    assert "_s2_core_prefix(base)" in body, \
        "the join must derive its prefix through the shared rule"


# ── preferences ──────────────────────────────────────────────────────────────

def test_both_preference_keys_are_registered():
    """preferences.load() keeps only `if k in DEFAULTS`. An unregistered key
    appears to save, works for the rest of the session, and is discarded on
    restart -- a setting that forgets itself overnight."""
    from mep_cmap.preferences import DEFAULTS
    assert "trials_selected_enabled" in DEFAULTS
    assert "trials_selected_groups" in DEFAULTS


def test_the_feature_is_off_by_default():
    """A study that never touches this writes exactly the files it always
    did."""
    from mep_cmap.preferences import DEFAULTS
    assert DEFAULTS["trials_selected_enabled"] is False


def test_the_default_groups_are_real_groups():
    from mep_cmap.preferences import DEFAULTS
    for key in DEFAULTS["trials_selected_groups"]:
        assert key in cg.GROUP_KEYS


# ── the session override ─────────────────────────────────────────────────────

def _app_src():
    return (PKG / "app.py").read_text(encoding="utf-8")


def test_the_override_is_stored_at_the_top_level():
    """NOT inside session["settings"], whose keys are all restored against a
    hardcoded literal. Absent has to mean "no override, use the preference",
    not "force this literal"."""
    src = _app_src()
    body = src[src.index("def _session_payload"):]
    body = body[:body.index("\n    def session_path")]
    assert '"column_selection":' in body
    settings = body[body.index("s = {"):body.index("# ── Compute study root")]
    assert "column_selection" not in settings


def test_the_override_is_restored_outside_the_settings_try_block():
    """One malformed value in that block silently drops every setting after
    it."""
    src = _app_src()
    body = src[src.index("def _apply_loaded_session"):]
    body = body[:body.index("def save_session_copy")]
    i_restore = body.index('sess.get("column_selection")')
    i_settings = body.index('s=sess.get("settings",{})')
    assert i_restore < i_settings


def test_a_session_without_the_key_means_no_override():
    """Every session written before this feature. It must fall through to the
    preference rather than force anything."""
    src = _app_src()
    body = src[src.index("def _apply_loaded_session"):]
    line = body[body.index('_colsel = sess.get("column_selection")'):][:220]
    assert "isinstance(_colsel, dict)" in line
    assert "else None" in line


def test_the_resolver_keeps_off_here_distinct_from_not_set_here():
    """A recording deliberately opted out must not opt itself back in the
    moment the global preference is switched on."""
    src = _app_src()
    body = src[src.index("def _effective_column_selection"):]
    body = body[:body.index("\n    def _refresh_colsel_control")]
    assert "isinstance(override, dict)" in body
    i_dict = body.index("isinstance(override, dict)")
    i_pref = body.index("prefs.trials_selected_enabled")
    assert i_dict < i_pref, "the preference is consulted before the override"


# ── the per-recording override control ───────────────────────────────────────

def test_the_override_control_offers_three_states():
    """A checkbox cannot express "use the preference"."""
    src = _app_src()
    for name in ("_COLSEL_INHERIT", "_COLSEL_ON", "_COLSEL_OFF"):
        assert f"{name} " in src or f"{name}," in src or f"{name}]" in src
    assert "values=[_COLSEL_INHERIT, _COLSEL_ON, _COLSEL_OFF]" in src


def test_inheriting_stores_none_not_a_dict():
    """A dict means "this recording decided"; only None defers."""
    src = _app_src()
    body = src[src.index("def _colsel_on_mode_change"):]
    body = body[:body.index("\n    def _colsel_choose")]
    i_mode = body.index("if mode == _COLSEL_INHERIT:")
    i_none = body.index("self.column_selection = None")
    assert i_mode < i_none


def test_skipping_stores_an_explicit_disabled_dict():
    """Not None. "Off here" and "not set here" must stay distinct on disk."""
    src = _app_src()
    body = src[src.index("def _colsel_on_mode_change"):]
    body = body[:body.index("\n    def _colsel_choose")]
    assert '"enabled": mode == _COLSEL_ON' in body


def test_flipping_off_and_back_on_keeps_the_chosen_groups():
    """Otherwise switching off would silently empty a selection the analyst
    made, and switching back on would look like it had never been set."""
    src = _app_src()
    body = src[src.index("def _colsel_on_mode_change"):]
    body = body[:body.index("\n    def _colsel_choose")]
    assert 'existing.get("groups")' in body


def test_the_control_is_refreshed_when_a_session_loads():
    """The widgets do not watch the attribute, so a restored session would
    leave the previous recording's answer on screen while the run used this
    one's."""
    src = _app_src()
    body = src[src.index("def _apply_loaded_session"):]
    body = body[:body.index("def save_session_copy")]
    i_set = body.index('sess.get("column_selection")')
    i_refresh = body.index("self._refresh_colsel_control()")
    assert i_set < i_refresh


def test_refreshing_does_not_rewrite_the_state_it_just_read():
    """The combobox trace fires on set(), which would overwrite the restored
    override with one derived from the widget."""
    src = _app_src()
    body = src[src.index("def _refresh_colsel_control"):]
    body = body[:body.index("\n    def _colsel_on_mode_change")]
    assert "self._colsel_suspend = True" in body
    guard = src[src.index("def _colsel_on_mode_change"):]
    assert 'getattr(self, "_colsel_suspend", False)' in guard[:400]


def test_changing_the_override_marks_the_session_dirty():
    """Otherwise the choice lives only in memory until something else
    happens to trigger a save."""
    src = _app_src()
    for fn in ("def _colsel_on_mode_change", "def _colsel_choose"):
        body = src[src.index(fn):]
        body = body[:body.index("\n    def ", 10)]
        assert "self._session_dirty = True" in body


def test_the_control_reads_groups_from_the_one_source_of_truth():
    """A second hardcoded list of groups in the GUI is exactly the drift
    column_groups exists to prevent."""
    src = _app_src()
    body = src[src.index("def _colsel_choose"):]
    body = body[:body.index("\n    def _refresh_run_button")]
    assert "from .column_groups import" in body
    assert "GROUPS" in body and "PROTECTED" in body


# ── the write path ───────────────────────────────────────────────────────────

def test_the_narrowing_happens_only_on_a_copy_at_write_time():
    """LAT_COLS is a positional contract behind ~20 index constants that
    assert their own positions. Narrowing anything the row builders share
    would put a variable schema behind them."""
    src = (PKG / "pipeline.py").read_text(encoding="utf-8")
    body = src[src.index("def pipeline_write_outputs"):]
    body = body[:body.index("def pipeline_generate_plots")]
    i_full = body.index('to_csv(_p("trials.csv")')
    i_narrow = body.index('to_csv(_p("trials_selected.csv")')
    assert i_full < i_narrow, "the full file must be written first"


def test_nothing_reassigns_lat_cols():
    """It is built once and read by name everywhere else."""
    src = (PKG / "pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    assigns = [n for n in ast.walk(tree)
               if isinstance(n, ast.Assign)
               and any(getattr(t, "id", "") == "LAT_COLS" for t in n.targets)]
    assert len(assigns) == 1


def test_no_narrowed_file_is_written_when_no_selection_is_given():
    """The default. A run that never configures this produces exactly the
    files it always did."""
    src = (PKG / "pipeline.py").read_text(encoding="utf-8")
    body = src[src.index("def pipeline_write_outputs"):]
    body = body[:body.index("def pipeline_generate_plots")]
    assert "if column_selection is not None:" in body


def test_an_empty_selection_is_not_confused_with_no_selection():
    """`or None` anywhere on this path would turn "protected columns only"
    into "write no file"."""
    src = (PKG / "pipeline.py").read_text(encoding="utf-8")
    assert "column_selection=column_selection," in src
    assert "column_selection=column_selection or None" not in src
