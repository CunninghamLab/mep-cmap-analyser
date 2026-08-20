"""
One projection, so two writers cannot disagree, and a revert that always works.

The Conditions tab and BIDS-ify both describe the same deliveries. When they
each built their own answer, a recording could carry two _events.tsv files that
contradicted each other, and nothing said which was right.
"""

import json
import os

import pytest

from mep_cmap import bidsify
from mep_cmap import events_model as em
from mep_cmap import stim_params as sp
from mep_cmap.conditions import ConditionRow


RAW = [{"onset": 1.0, "code": "A"}, {"onset": 2.0, "code": "A"},
       {"onset": 3.0, "code": "A"}, {"onset": 4.0, "code": "A"},
       {"onset": 5.0, "code": "G"}]


# ── raw, unconditioned ───────────────────────────────────────────────────────

def test_without_conditions_the_raw_arrangement_is_written(tmp_path):
    """The honest description of a recording nobody has grouped yet."""
    cols, rows = em.project(RAW)
    assert cols == ["onset", "duration", "trial_type"]
    assert [r["trial_type"] for r in rows] == ["A", "A", "A", "A", "G"]


def test_an_all_blank_condition_column_is_not_written():
    """A column of n/a on every row is noise in a file other people parse."""
    cols, _ = em.project(RAW)
    assert "condition" not in cols


def test_events_come_out_in_time_order():
    cols, rows = em.project(list(reversed(RAW)))
    assert [r["onset"] for r in rows] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_stim_times_keep_file_order_not_sorted_order():
    """A ConditionRow names trials by index within its stim type, so
    re-ordering here would silently reassign which trial is in which group."""
    times = em.stim_times_from_raw(RAW)
    assert times["A"] == [1.0, 2.0, 3.0, 4.0]
    assert times["G"] == [5.0]


# ── the split ────────────────────────────────────────────────────────────────

def _split_rows():
    """Half of A at 100 mA, half at 150 mA."""
    return [ConditionRow(stim_type="A", condition="100mA", trials=(0, 1)),
            ConditionRow(stim_type="A", condition="150mA", trials=(2, 3)),
            ConditionRow(stim_type="G", condition="MEP", trials=(0,))]


def test_a_split_code_carries_its_condition():
    cols, rows = em.project(RAW, _split_rows())
    assert "condition" in cols
    assert [r["condition"] for r in rows] == ["100mA", "100mA",
                                              "150mA", "150mA", "MEP"]


def test_each_half_of_a_split_gets_its_own_parameter_set():
    """The case a per-code assignment could not express: one code, two
    intensities, so two rows of _nibs.tsv."""
    sets = [sp.StimParamSet("A_100", nibs_type="PNS", position="tibial"),
            sp.StimParamSet("A_150", nibs_type="PNS", position="tibial")]
    code_sets = {em.pair_key("A", "100mA"): "A_100",
                 em.pair_key("A", "150mA"): "A_150"}
    _, rows = em.project(RAW, _split_rows(), code_sets, sets)
    ids = [r["nibs_event_id"] for r in rows]
    assert ids[:4] == ["A_100", "A_100", "A_150", "A_150"]


def test_a_pair_assignment_overrides_the_bare_code():
    """Splitting must override the protocol the code carried beforehand."""
    sets = [sp.StimParamSet("A_100"), sp.StimParamSet("A_old")]
    code_sets = {"A": "A_old", em.pair_key("A", "100mA"): "A_100"}
    _, rows = em.project(RAW, _split_rows(), code_sets, sets)
    assert rows[0]["nibs_event_id"] == "A_100"


def test_an_unsplit_code_still_uses_its_bare_assignment():
    sets = [sp.StimParamSet("MEP_120")]
    code_sets = {"G": "MEP_120"}
    _, rows = em.project(RAW, _split_rows(), code_sets, sets)
    assert rows[-1]["nibs_event_id"] == "MEP_120"


def test_only_genuinely_split_codes_need_a_pair_assignment():
    """Asking for a set per half of something never halved is busywork."""
    pairs = em.split_codes(_split_rows())
    assert sorted(pairs) == [("A", "100mA"), ("A", "150mA")]


def test_an_unnamed_group_still_counts_as_a_split():
    """A real table: one group of ten named, the rest left blank. Requiring two
    NAMED groups hid this entirely -- the code looked unsplit, so the dialogue
    offered one dropdown for two protocols."""
    rows = [ConditionRow(stim_type="Trigger", condition="10",
                         trials=tuple(range(10))),
            ConditionRow(stim_type="Trigger", condition="",
                         trials=tuple(range(10, 162)))]
    assert em.unnamed_splits(rows) == ["Trigger"]
    # The named half is still offered; the blank one cannot be, and the caller
    # is told why rather than left with a silently short list.
    assert em.split_codes(rows) == [("Trigger", "10")]


def test_a_code_with_one_group_is_not_a_split():
    rows = [ConditionRow(stim_type="Start Task", condition="",
                         trials=tuple(range(6)))]
    assert em.split_codes(rows) == []
    assert em.unnamed_splits(rows) == []


def test_a_fully_named_split_raises_no_complaint():
    assert em.unnamed_splits(_split_rows()) == []


def test_an_excluded_trial_is_still_accounted_for():
    """A reader must be able to tell an excluded trial from one that was never
    there, which a file of only the kept trials cannot express."""
    rows = [ConditionRow(stim_type="A", condition="x", trials=(0,),
                         excluded=True),
            ConditionRow(stim_type="A", condition="keep", trials=(1, 2, 3))]
    _, out = em.project(RAW, rows)
    assert len(out) == 4
    assert out[0]["trial_type"] == "n/a"


# ── conditions come from the recording's session, not from BIDS-ify ──────────

def _write_session(tmp_path, rows, stem="rec"):
    """A session JSON exactly where session_path_for says it goes."""
    from mep_cmap.app import session_path_for
    src = os.path.join(str(tmp_path), "raw", f"{stem}.smr")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    path = session_path_for(src, None, os.path.join(str(tmp_path), "deriv"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"condition_rows": rows,
                   # Also stored, and deliberately NOT what is read: this is
                   # the model already projected, and trusting it would give
                   # two projections again.
                   "condition_event_rows": [{"onset": 99.0, "duration": 0,
                                             "trial_type": "WRONG"}]}, fh)
    return src


def test_conditions_are_read_from_the_recordings_session(tmp_path):
    src = _write_session(tmp_path, [
        {"stim_type": "A", "condition": "100mA", "trials": [0, 1]},
        {"stim_type": "A", "condition": "150mA", "trials": [2, 3]}])
    rows = bidsify.conditions_for(src, None, os.path.join(str(tmp_path), "deriv"))
    assert [r.condition for r in rows] == ["100mA", "150mA"]
    assert rows[0].trials == (0, 1)


def test_the_model_is_read_not_the_stored_projection(tmp_path):
    """condition_event_rows is the table already turned into events. Reading it
    would mean two projections of one table, which is the disagreement this
    whole design exists to remove."""
    src = _write_session(tmp_path, [
        {"stim_type": "A", "condition": "100mA", "trials": [0, 1, 2, 3]}])
    rows = bidsify.conditions_for(src, None, os.path.join(str(tmp_path), "deriv"))
    _, out = em.project(RAW, rows)
    assert all(r["trial_type"] != "WRONG" for r in out)
    assert all(float(r["onset"]) != 99.0 for r in out)


def test_a_recording_with_no_session_has_no_conditions(tmp_path):
    """The ordinary case: most analysts never open the Conditions tab, and
    their recordings must still convert."""
    src = os.path.join(str(tmp_path), "raw", "never-analysed.smr")
    rows = bidsify.conditions_for(src, None, os.path.join(str(tmp_path), "deriv"))
    assert rows == []
    cols, out = em.project(RAW, rows)
    assert [r["trial_type"] for r in out] == ["A", "A", "A", "A", "G"]


def test_an_unreadable_session_does_not_stop_a_conversion(tmp_path):
    from mep_cmap.app import session_path_for
    src = os.path.join(str(tmp_path), "raw", "broken.smr")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    path = session_path_for(src, None, os.path.join(str(tmp_path), "deriv"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{not json at all")
    assert bidsify.conditions_for(src, None,
                                  os.path.join(str(tmp_path), "deriv")) == []


def test_malformed_condition_rows_are_skipped_not_fatal(tmp_path):
    src = _write_session(tmp_path, [
        "not a dict", None,
        {"stim_type": "A", "condition": "keep", "trials": [0]}])
    rows = bidsify.conditions_for(src, None, os.path.join(str(tmp_path), "deriv"))
    assert [r.condition for r in rows] == ["keep"]


def test_reverting_is_deleting_the_layer(tmp_path):
    """No stored snapshot is needed: the raw arrangement comes from the
    recording, which BIDS-ify is reading anyway in order to convert it. So a
    revert is removing the conditions, and what is left is what the file said."""
    src = _write_session(tmp_path, [])
    rows = bidsify.conditions_for(src, None, os.path.join(str(tmp_path), "deriv"))
    assert rows == []
    _, out = em.project(RAW, rows)
    assert [r["trial_type"] for r in out] == ["A", "A", "A", "A", "G"]


def test_bidsify_state_holds_no_conditions():
    """They belong to the recording. A copy here would be a second truth, and
    would file the work of analysts who never open this tab under a feature
    they do not use."""
    from mep_cmap.bidsify_state import FileBidsRecord
    rec = FileBidsRecord(rel_path="a.smr")
    assert not hasattr(rec, "condition_rows")
    assert not hasattr(rec, "raw_events")


# ── finding the session when the filename cannot say who it belongs to ───────

def _session_at(tmp_path, sub, stem, rows, file_path=None):
    """Write a session where the app would, for a given participant."""
    d = os.path.join(str(tmp_path), "deriv", "derivatives", sub, "ses-01")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{sub}_ses-01_{stem}_session.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"condition_rows": rows,
                   "file_path": file_path or os.path.join("rawdata", "Spike",
                                                          stem + ".smr")}, fh)
    return p


def test_a_session_is_found_when_the_filename_names_no_participant(tmp_path):
    """`rawdata/Spike/Example Data 1.smr` has no participant to parse, so the
    constructed path resolves to sub-unknown and misses a session saved under
    the metadata the analyst typed. The conditions exist; they were not found."""
    _session_at(tmp_path, "sub-333", "Example Data 1",
                [{"stim_type": "trigger", "condition": "first",
                  "trials": [0, 1, 2, 3, 4]},
                 {"stim_type": "trigger", "condition": "last",
                  "trials": [5, 6, 7, 8, 9]}])
    src = os.path.join(str(tmp_path), "rawdata", "Spike", "Example Data 1.smr")
    deriv = os.path.join(str(tmp_path), "deriv")
    rows = bidsify.conditions_for(src, None, deriv)
    assert sorted(r.condition for r in rows) == ["first", "last"]
    assert em.split_codes(rows) == [("trigger", "first"), ("trigger", "last")]


def test_the_match_is_confirmed_against_the_stored_source(tmp_path):
    """Matching on the filename stem alone would be a guess. A session whose
    own file_path names a different recording is not this recording's."""
    _session_at(tmp_path, "sub-1", "Example Data 1",
                [{"stim_type": "x", "condition": "wrong", "trials": [0]}],
                file_path=os.path.join("rawdata", "Other", "Different.smr"))
    src = os.path.join(str(tmp_path), "rawdata", "Spike", "Example Data 1.smr")
    deriv = os.path.join(str(tmp_path), "deriv")
    assert bidsify.conditions_for(src, None, deriv) == []


def test_the_most_recent_claimant_wins(tmp_path):
    """Correcting a participant id leaves the old session orphaned, still
    claiming the same recording."""
    import time
    _session_at(tmp_path, "sub-1", "Example Data 1", [])
    time.sleep(0.01)
    _session_at(tmp_path, "sub-333", "Example Data 1",
                [{"stim_type": "trigger", "condition": "first",
                  "trials": [0]}])
    src = os.path.join(str(tmp_path), "rawdata", "Spike", "Example Data 1.smr")
    found = bidsify.find_session_for(src, None,
                                     os.path.join(str(tmp_path), "deriv"))
    assert "sub-333" in found


def test_a_recording_with_no_session_anywhere_is_still_fine(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "deriv", "derivatives"),
                exist_ok=True)
    src = os.path.join(str(tmp_path), "rawdata", "nothing.smr")
    assert bidsify.find_session_for(src, None,
                                    os.path.join(str(tmp_path), "deriv")) == ""
    assert bidsify.conditions_for(src, None,
                                  os.path.join(str(tmp_path), "deriv")) == []


# ── who the recording belongs to ─────────────────────────────────────────────

def _session_with_meta(tmp_path, sub, stem):
    d = os.path.join(str(tmp_path), "deriv", "derivatives", sub, "ses-01")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{sub}_ses-01_{stem}_session.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"condition_rows": [],
                   "file_path": os.path.join("rawdata", "Spike",
                                             stem + ".smr"),
                   "study_metadata": {"participant_id": sub,
                                      "session": "ses-01", "task": "",
                                      "timepoint": "", "limb": "",
                                      "measure": "", "acq": ""}}, fh)
    return p


def test_the_recorded_participant_is_used_not_the_filename(tmp_path):
    """The conversion wrote to sub-unknown while the analyst had typed
    sub-333. The Study Metadata window is the only place a participant is
    entered, so a filename that says nothing is not evidence of nothing."""
    _session_with_meta(tmp_path, "sub-333", "Example Data 1")
    src = os.path.join(str(tmp_path), "rawdata", "Spike", "Example Data 1.smr")
    meta = bidsify.recorded_metadata_for(
        src, None, os.path.join(str(tmp_path), "deriv"))
    assert meta is not None
    assert meta.participant_id == "sub-333"


def test_no_session_leaves_the_filename_guess_in_place(tmp_path):
    """A BIDS-named file needs no session to be named correctly."""
    os.makedirs(os.path.join(str(tmp_path), "deriv", "derivatives"),
                exist_ok=True)
    src = os.path.join(str(tmp_path), "raw", "sub-015_ses-2_emg.smr")
    assert bidsify.recorded_metadata_for(
        src, None, os.path.join(str(tmp_path), "deriv")) is None


def test_a_session_without_a_participant_is_not_used(tmp_path):
    """Half-written metadata must not override a filename that does say."""
    d = os.path.join(str(tmp_path), "deriv", "derivatives", "sub-1", "ses-01")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "sub-1_ses-01_rec_session.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"file_path": os.path.join("raw", "rec.smr"),
                   "study_metadata": {"participant_id": ""}}, fh)
    src = os.path.join(str(tmp_path), "raw", "rec.smr")
    assert bidsify.recorded_metadata_for(
        src, None, os.path.join(str(tmp_path), "deriv")) is None


# ── provenance ───────────────────────────────────────────────────────────────

def test_the_events_file_says_where_its_grouping_came_from():
    assert "No conditions" in em.describe_source([])
    assert "Conditions tab" in em.describe_source(_split_rows())


def test_round_trip_between_raw_and_stim_times():
    assert em.raw_from_stim_times(em.stim_times_from_raw(RAW)) == RAW
