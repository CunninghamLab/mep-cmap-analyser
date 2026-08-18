"""
Assigning stimulus events to named conditions.

A recording's markers say what kind of stimulus fired, not what it was for.
Twenty pulses labelled ``A`` may be ten before an intervention and ten after,
and nothing in the file distinguishes them. A condition is a second axis
alongside the stimulus type: ``A`` decides how a response is detected, ``pre``
and ``post`` decide what the trial means.

The property everything else rests on is the last test in the first section: a
table nobody has edited produces exactly the analysis groups the recording
produced before conditions existed.
"""

import pytest

from mep_cmap import conditions as C


# ── the safety property ──────────────────────────────────────────────────────

def test_an_untouched_table_reproduces_the_existing_groups():
    """Doing nothing must change nothing.

    This is what makes the conditions table a refinement rather than a step
    that has to be completed before a recording can be analysed at all.
    """
    stim = {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0]}
    rows = C.validate(C.rows_from_events(stim), stim)
    groups, _decoded = C.group_events(C.to_event_rows(rows, stim))
    assert sorted(groups) == sorted(stim)
    assert [len(v) for _k, v in sorted(groups.items())] == [3, 2]


def test_a_blank_condition_composes_to_the_stimulus_type_alone():
    assert C.compose("A", "") == "A"
    assert C.compose("A", None) == "A"
    assert C.compose("A", "n/a") == "A"
    assert C.compose("A", "pre") == "A" + C.SEPARATOR + "pre"


def test_compose_and_decompose_are_inverses():
    for stim, cond in (("A", ""), ("A", "pre"), ("Trigger", "post_10min")):
        assert C.decompose(C.compose(stim, cond)) == (stim, cond)


# ── trial lists ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("1", (0,)),
    ("1,2,3", (0, 1, 2)),
    ("1-3", (0, 1, 2)),
    ("3,1,2", (0, 1, 2)),
    ("1-3, 2-4", (0, 1, 2, 3)),
    ("1-20, 25, 30-35", tuple(range(20)) + (24,) + tuple(range(29, 35))),
    ("", ()),
    (None, ()),
])
def test_trial_lists_parse(text, expected):
    assert C.parse_trials(text) == expected


def test_trial_lists_are_written_one_based_and_stored_zero_based():
    """Displayed as the inspector and the trial file number them; stored as
    the arrays are indexed. Getting that boundary wrong is an off-by-one in
    every trial of a condition."""
    assert C.parse_trials("1") == (0,)
    assert C.format_trials([0]) == "1"


def test_runs_collapse_into_ranges():
    """Forty consecutive trials should read 1-40, not forty numbers: the table
    is meant to be checked at a glance."""
    assert C.format_trials(range(40)) == "1-40"
    assert C.format_trials([0, 1, 2, 5, 9, 10]) == "1-3, 6, 10-11"


def test_a_reversed_range_is_accepted():
    assert C.parse_trials("5-3") == (2, 3, 4)


def test_trial_zero_is_refused():
    with pytest.raises(C.ConditionError, match="numbered from 1"):
        C.parse_trials("0")


def test_a_trial_beyond_the_recording_is_refused():
    """Silently keeping the half that exists would leave the analyst believing
    they had assigned forty."""
    with pytest.raises(C.ConditionError, match="does not exist"):
        C.parse_trials("1-40", n_available=20)


def test_nonsense_is_refused_with_the_syntax():
    with pytest.raises(C.ConditionError, match="1-20, 25, 30-35"):
        C.parse_trials("first ten")


# ── names ────────────────────────────────────────────────────────────────────

def test_spaces_become_underscores():
    assert C.sanitise_name("  post 10 min ") == "post_10_min"


def test_the_separator_is_refused_not_replaced():
    """A renamed condition is a different condition, and doing that silently
    is how a trial ends up in a group nobody made."""
    with pytest.raises(C.ConditionError, match="may not contain"):
        C.sanitise_name("pre" + C.SEPARATOR + "post")


def test_tabs_are_refused_because_the_file_is_tab_separated():
    with pytest.raises(C.ConditionError, match="tab-separated"):
        C.sanitise_name("pre\tpost")


def test_duplicate_names_are_numbered_within_a_stimulus_type():
    pairs = C.deduplicate_names([("A", "x"), ("A", "x"), ("A", "x")])
    assert [c for _s, c in pairs] == ["x", "x_2", "x_3"]


def test_the_same_name_on_two_stimulus_types_is_not_a_duplicate():
    """Uniqueness is required of the group key, not the name.

    A·x and B·x are different groups. Renaming on the name alone turned an
    untouched table into A and B·_2.
    """
    pairs = C.deduplicate_names([("A", "x"), ("B", "x")])
    assert pairs == [("A", "x"), ("B", "x")]


def test_blank_conditions_are_never_numbered():
    """A blank is the absence of a name, not a name that happens to be empty."""
    pairs = C.deduplicate_names([("A", ""), ("B", ""), ("C", "")])
    assert [c for _s, c in pairs] == ["", "", ""]


# ── splitting ────────────────────────────────────────────────────────────────

def test_splitting_moves_the_selection_and_keeps_the_rest():
    stim = {"A": [float(i) for i in range(20)]}
    rows = C.rows_from_events(stim)
    rows = C.split_row(rows, 0, range(10), new_condition="pre",
                       keep_condition="post")
    assert [r.condition for r in rows] == ["post", "pre"]
    assert rows[1].trials == tuple(range(10))
    assert rows[0].trials == tuple(range(10, 20))


def test_splitting_loses_no_trials():
    """Expressed as a split rather than two assignments precisely so that no
    trial can be dropped or duplicated between them."""
    stim = {"A": [float(i) for i in range(20)]}
    rows = C.split_row(C.rows_from_events(stim), 0, [2, 4, 6],
                       new_condition="odd")
    total = sorted(t for r in rows for t in r.trials)
    assert total == list(range(20))


def test_splitting_everything_is_refused():
    stim = {"A": [1.0, 2.0]}
    with pytest.raises(C.ConditionError, match="renames the condition"):
        C.split_row(C.rows_from_events(stim), 0, [0, 1], new_condition="x")


def test_splitting_on_trials_not_in_the_row_is_refused():
    stim = {"A": [1.0, 2.0], "B": [3.0]}
    rows = C.rows_from_events(stim)
    with pytest.raises(C.ConditionError, match="none of the selected"):
        C.split_row(rows, 1, [5, 6], new_condition="x")


def test_autofill_deals_trials_into_runs():
    stim = {"A": [float(i) for i in range(10)]}
    rows = C.autofill(C.rows_from_events(stim), 0, per_row=5, n_available=10)
    assert len(rows) == 2
    assert rows[0].trials == (0, 1, 2, 3, 4)
    assert rows[1].trials == (5, 6, 7, 8, 9)


# ── completeness ─────────────────────────────────────────────────────────────

def test_unassigned_trials_are_reported():
    """A trial in no condition vanishes between the recording and the analysis.

    That is the same silent loss as a marker filtered out of an events file,
    and harder to notice when the analyst did the removing themselves.
    """
    stim = {"A": [float(i) for i in range(10)]}
    rows = [C.ConditionRow("A", "pre", (0, 1, 2))]
    assert C.unassigned(rows, stim) == {"A": tuple(range(3, 10))}


def test_apply_refuses_while_trials_are_unassigned():
    stim = {"A": [float(i) for i in range(10)]}
    rows = [C.ConditionRow("A", "pre", (0, 1, 2))]
    with pytest.raises(C.ConditionError, match="in no condition"):
        C.validate(rows, stim)


def test_unassigned_may_be_allowed_explicitly():
    stim = {"A": [float(i) for i in range(10)]}
    rows = [C.ConditionRow("A", "pre", (0, 1, 2))]
    assert C.validate(rows, stim, allow_unassigned=True)


def test_a_trial_in_two_conditions_is_refused():
    """The analysis would count it twice."""
    stim = {"A": [float(i) for i in range(4)]}
    rows = [C.ConditionRow("A", "x", (0, 1)), C.ConditionRow("A", "y", (1, 2, 3))]
    with pytest.raises(C.ConditionError, match="more than one condition"):
        C.validate(rows, stim)


def test_the_error_names_the_trials():
    """'some trials are unassigned' cannot be acted on."""
    stim = {"A": [float(i) for i in range(10)]}
    rows = [C.ConditionRow("A", "pre", (0, 1, 2))]
    with pytest.raises(C.ConditionError, match="4-10"):
        C.validate(rows, stim)


# ── the events file ──────────────────────────────────────────────────────────

def test_event_rows_carry_the_two_axes_separately():
    """A condition is a factor to model, not a substring to parse from a name."""
    stim = {"A": [1.0, 2.0]}
    rows = [C.ConditionRow("A", "pre", (0, 1))]
    ev = C.to_event_rows(rows, stim)
    assert ev[0] == {"onset": 1.0, "duration": 0.0,
                     "trial_type": "A", "condition": "pre"}


def test_excluded_trials_are_written_not_dropped():
    """A reader can then tell a trial that was excluded from one that was never
    there, which a file of only the kept trials cannot express."""
    stim = {"A": [1.0, 2.0, 3.0]}
    rows = [C.ConditionRow("A", "keep", (0, 1)),
            C.ConditionRow("A", "", (2,), excluded=True)]
    ev = C.to_event_rows(rows, stim)
    assert len(ev) == 3
    assert ev[2]["trial_type"] == C.NA


def test_event_rows_are_sorted_by_onset():
    stim = {"A": [5.0, 1.0], "B": [3.0]}
    rows = [C.ConditionRow("A", "", (0, 1)), C.ConditionRow("B", "", (0,))]
    onsets = [r["onset"] for r in C.to_event_rows(rows, stim)]
    assert onsets == sorted(onsets)


# ── composition ──────────────────────────────────────────────────────────────

def test_grouping_composes_the_two_columns():
    ev = [{"onset": 1.0, "trial_type": "A", "condition": "pre"},
          {"onset": 2.0, "trial_type": "A", "condition": "post"},
          {"onset": 3.0, "trial_type": "B", "condition": "n/a"}]
    groups, decoded = C.group_events(ev)
    assert sorted(groups) == ["A" + C.SEPARATOR + "post",
                              "A" + C.SEPARATOR + "pre", "B"]
    assert decoded["B"] == ("B", "")


def test_grouping_skips_excluded_rows():
    ev = [{"onset": 1.0, "trial_type": "A", "condition": "x"},
          {"onset": 2.0, "trial_type": C.NA, "condition": "x"}]
    groups, _ = C.group_events(ev)
    assert sum(len(v) for v in groups.values()) == 1


def test_grouping_survives_a_malformed_onset():
    """A row that cannot be read is skipped rather than aborting the file."""
    ev = [{"onset": "nonsense", "trial_type": "A", "condition": ""},
          {"onset": 2.0, "trial_type": "A", "condition": ""}]
    groups, _ = C.group_events(ev)
    assert groups == {"A": [2.0]}


def test_a_file_without_a_condition_column_still_groups():
    """Every events file written before this existed."""
    ev = [{"onset": 1.0, "trial_type": "A"}, {"onset": 2.0, "trial_type": "A"}]
    groups, decoded = C.group_events(ev)
    assert groups == {"A": [1.0, 2.0]}
    assert decoded["A"] == ("A", "")


def test_group_events_is_the_only_place_that_composes():
    """The pipeline and the preview must not reconcile the two independently.

    A preview composing differently from the run would offer trials the
    analysis does not group the same way -- the failure this codebase has
    produced four times in other guises.
    """
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
    offenders = []
    for path in sorted(pkg.glob("*.py")):
        if path.name == "conditions.py":
            continue
        src = path.read_text(encoding="utf-8")
        code = "\n".join(l for l in src.split("\n")
                         if not l.strip().startswith("#"))
        if "SEPARATOR" in code and "group_events" not in code:
            offenders.append(path.name)
    assert not offenders, (
        "these compose group keys without going through group_events: "
        + ", ".join(offenders))


# ── the pipeline seam ────────────────────────────────────────────────────────

import pathlib

PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
PIPE = (PKG / "pipeline.py").read_text(encoding="utf-8")
APP = (PKG / "app.py").read_text(encoding="utf-8")
S2 = (PKG / "stage2.py").read_text(encoding="utf-8")
PREV = (PKG / "preview.py").read_text(encoding="utf-8")


def _cfg(**kw):
    from mep_cmap.pipeline import PipelineConfig
    return PipelineConfig(**kw)


def test_the_condition_column_is_appended_not_inserted():
    """`common` is a positional list of literals padded to the row width.

    Inserting a column mid-list shifts every field after it, which is the
    six-column misalignment this schema already learned once.
    """
    from mep_cmap.pipeline import LAT_COLS
    assert LAT_COLS[-1] == "Condition"
    assert LAT_COLS[1] == "StimType", "the historical prefix must not move"


def test_the_column_is_written_by_name_resolved_index():
    assert '_C_COND    = LAT_COLS.index("Condition")' in PIPE
    assert "_row[_C_COND]" in PIPE


def test_stimtype_reports_the_stimulus_not_the_group_key():
    """A row for A·pre says StimType=A, Condition=pre.

    A condition is a factor the group analysis can model; a composite name is
    a string someone has to parse.
    """
    assert "_base_stim, _cond = split_group_key(cfg, stim_type)" in PIPE
    assert "name, _base_stim, custom_labels.get(stim_type" in PIPE


def test_splitting_a_key_falls_back_to_the_stimulus_type():
    """Every recording whose conditions were never assigned."""
    from mep_cmap.pipeline import split_group_key
    assert split_group_key(_cfg(), "A") == ("A", "")
    cfg = _cfg(condition_map={"A\u00b7pre": ("A", "pre")})
    assert split_group_key(cfg, "A\u00b7pre") == ("A", "pre")
    assert split_group_key(cfg, "B") == ("B", "")


def test_only_one_place_separates_a_key():
    """A key is opaque everywhere else, which is what lets the composition
    live entirely in conditions.py."""
    import ast

    tree = ast.parse(PIPE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "split_group_key":
            return
    raise AssertionError("split_group_key not found")


def test_both_loaders_compose_through_the_same_call():
    """A preview grouping trials differently from the run would offer a set
    the analysis does not produce -- the failure this codebase has made in
    four other guises."""
    assert "from .conditions import group_events" in PIPE
    assert "group_events(event_rows)" in PIPE
    assert 'event_rows=params.get("event_rows")' in PREV


def test_every_loader_call_passes_event_rows():
    """Walks the calls rather than naming the ones already known."""
    import ast

    offenders = []
    for path in sorted(PKG.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and \
                    getattr(node.func, "id", "") == "pipeline_load_file":
                kw = {k.arg for k in node.keywords if k.arg}
                if "event_rows" not in kw:
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "these would group by stimulus type while the run groups by condition: "
        + ", ".join(offenders))


def test_an_unassigned_recording_passes_no_event_rows():
    """None until the tab has been applied, which is what makes an untouched
    recording behave exactly as it did."""
    a = APP.index("def _snapshot_analysis_params")
    b = APP.index("\n    def ", a + 10)
    body = APP[a:b]
    assert "or None" in body[body.index("event_rows"):body.index("event_rows") + 120]


# ── the group join ───────────────────────────────────────────────────────────

def test_condition_is_an_optional_join_key():
    """An add-on written before conditions, including a third-party one, emits
    no such column and must still join."""
    assert '_S2_OPTIONAL_JOIN_KEYS = ("Condition",)' in S2
    assert '_S2_JOIN_KEYS = ("StimType", "Segment")' in S2


def test_an_ambiguous_join_is_refused_not_guessed():
    """With conditions assigned, A·pre and A·post both report StimType=A and
    both have a Segment 1. A sidecar without the column cannot say which row
    it describes, and a left join would attach one condition's measurements to
    another's trial."""
    assert 'if "Condition" in df.columns and "Condition" not in use:' in S2
    assert "cannot be" in S2 and "Re-run the add-on" in S2


def test_the_optional_key_is_used_when_present():
    assert "_S2_JOIN_KEYS + _S2_OPTIONAL_JOIN_KEYS" in S2


# ── the selected data range ──────────────────────────────────────────────────

def test_cropping_keeps_only_the_stimuli_in_range():
    from mep_cmap.pipeline import crop_stim_times
    stim = {"A": [1.0, 5.0, 9.0], "B": [2.0, 20.0]}
    out = crop_stim_times(stim, crop_start=0.0, crop_end=10.0)
    assert out == {"A": [1.0, 5.0, 9.0], "B": [2.0]}


def test_a_type_with_nothing_in_range_is_dropped():
    """A stimulus type with no trials in range is not a group."""
    from mep_cmap.pipeline import crop_stim_times
    out = crop_stim_times({"A": [1.0], "B": [99.0]},
                          crop_start=0.0, crop_end=10.0)
    assert set(out) == {"A"}


def test_discontinuous_ranges_are_honoured():
    """A selection of two blocks is two blocks, not their outer bounds."""
    from mep_cmap.pipeline import crop_stim_times
    out = crop_stim_times({"A": [1.0, 50.0, 91.0]},
                          crop_ranges=[(0.0, 10.0), (90.0, 100.0)])
    assert out == {"A": [1.0, 91.0]}


def test_no_crop_returns_everything():
    from mep_cmap.pipeline import crop_stim_times
    stim = {"A": [1.0, 2.0]}
    assert crop_stim_times(stim) == stim


def test_cropping_does_not_mutate_its_input():
    """The caller's dict is often the cached event list for the whole file."""
    from mep_cmap.pipeline import crop_stim_times
    stim = {"A": [1.0, 99.0]}
    crop_stim_times(stim, crop_start=0.0, crop_end=10.0)
    assert stim == {"A": [1.0, 99.0]}


def test_the_loader_and_the_tab_crop_by_the_same_rule():
    """Conditions are assigned by trial INDEX.

    A table built from the whole recording numbers its trials differently from
    the analysis, so every assignment lands on the wrong trial -- and the
    waveforms drawn beside the list are already cropped, so the two halves of
    the tab would disagree with each other as well.
    """
    tab = (PKG / "conditions_tab.py").read_text(encoding="utf-8")
    assert "crop_stim_times(" in tab
    assert "crop_stim_times(stim_times, crop_ranges, crop_start, crop_end)" \
        in PIPE, "the loader must use the shared rule, not its own copy"


def test_the_loader_keeps_no_second_copy_of_the_rule():
    a = PIPE.index("def pipeline_load_file")
    b = PIPE.index("\ndef ", a + 10)
    body = PIPE[a:b]
    assert "stim_times.pop(k)" not in body, \
        "the inline crop was replaced by the shared helper"


# ── per-condition epochs ─────────────────────────────────────────────────────

def test_a_row_takes_the_default_until_given_a_window():
    row = C.ConditionRow("A", "pre", (0, 1))
    assert row.window is None


def test_a_row_with_a_window_reports_it():
    row = C.ConditionRow("A", "pre", (0, 1), pre_ms=20.0, post_ms=50.0)
    assert row.window == (20.0, 50.0)


def test_the_window_map_is_keyed_by_group_not_stimulus_type():
    """A-pre and A-post are separate rows on the labels tab and may want
    different windows, which is the whole reason for setting one per
    condition."""
    rows = [C.ConditionRow("A", "pre", (0,), pre_ms=20.0, post_ms=50.0),
            C.ConditionRow("A", "post", (1,), pre_ms=100.0, post_ms=500.0)]
    wm = C.window_map_from_rows(rows)
    assert wm == {"A" + C.SEPARATOR + "pre": (20.0, 50.0),
                  "A" + C.SEPARATOR + "post": (100.0, 500.0)}


def test_an_untouched_table_contributes_no_windows():
    """Which is the same as having no per-type windows at all."""
    stim = {"A": [1.0, 2.0], "B": [3.0]}
    assert C.window_map_from_rows(C.rows_from_events(stim)) == {}


def test_an_excluded_condition_contributes_no_window():
    rows = [C.ConditionRow("A", "x", (0,), excluded=True,
                           pre_ms=20.0, post_ms=50.0)]
    assert C.window_map_from_rows(rows) == {}


def test_one_side_only_is_carried_through():
    """Widening only the tail should not require restating the lead-in."""
    rows = [C.ConditionRow("A", "x", (0,), post_ms=500.0)]
    assert C.window_map_from_rows(rows) == {"A" + C.SEPARATOR + "x":
                                            (None, 500.0)}
