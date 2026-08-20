"""
Conditions must survive conversion and reopening.

A recording converted with its grouping written into *_events.tsv was reopened
as if it had never been grouped: the reader took onset and trial_type and threw
the condition column away. And confirming conditions on a converted recording
rewrote that same file with four columns, deleting the nibs_event_id and
nibs_position_id links BIDS-ify had put there -- because for a converted file
"beside the recording" IS the BIDS events file.
"""

import csv
import os

import pytest

from mep_cmap import events_model as em
from mep_cmap.conditions import ConditionRow
from mep_cmap.formats.edf import read_events_rows


def _write(path, rows, cols):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ── the round trip ───────────────────────────────────────────────────────────

def test_a_grouping_survives_being_written_and_read_back():
    """The whole point: convert with conditions, reopen, get them back."""
    rows = [ConditionRow(stim_type="Trigger", condition="first",
                         trials=(0, 1)),
            ConditionRow(stim_type="Trigger", condition="last", trials=(2, 3)),
            ConditionRow(stim_type="Start Task", condition="start",
                         trials=(0,))]
    raw = [{"onset": 1.0, "code": "Start Task"},
           {"onset": 2.0, "code": "Trigger"}, {"onset": 3.0, "code": "Trigger"},
           {"onset": 4.0, "code": "Trigger"}, {"onset": 5.0, "code": "Trigger"}]
    _cols, written = em.project(raw, rows)
    back = em.rows_from_events(written)
    assert [(r.stim_type, r.condition, r.trials) for r in back] == [
        ("Start Task", "start", (0,)),
        ("Trigger", "first", (0, 1)),
        ("Trigger", "last", (2, 3)),
    ]


def test_trials_are_numbered_within_their_stim_type():
    """A ConditionRow trial index is per stim type, not per file. Numbering
    across the file would reassign every trial on reopen."""
    written = [{"onset": 1.0, "trial_type": "A", "condition": "x"},
               {"onset": 2.0, "trial_type": "B", "condition": "y"},
               {"onset": 3.0, "trial_type": "A", "condition": "x"}]
    back = em.rows_from_events(written)
    by = {(r.stim_type, r.condition): r.trials for r in back}
    assert by[("A", "x")] == (0, 1)
    assert by[("B", "y")] == (0,)


def test_an_excluded_trial_still_consumes_its_index():
    """It occupies a position in its stim type. Skipping it silently shifts
    every later trial by one."""
    written = [{"onset": 1.0, "trial_type": "A", "condition": "keep"},
               {"onset": 2.0, "trial_type": "A", "condition": "n/a"},
               {"onset": 3.0, "trial_type": "A", "condition": "keep"}]
    back = em.rows_from_events(written)
    assert back[0].trials == (0, 2)


def test_an_ungrouped_file_yields_no_rows():
    """So the caller builds its usual one-row-per-stim-type table."""
    written = [{"onset": 1.0, "trial_type": "A", "condition": "n/a"},
               {"onset": 2.0, "trial_type": "A", "condition": ""}]
    assert em.rows_from_events(written) == []


def test_rows_come_back_in_the_order_they_were_written():
    written = [{"onset": 3.0, "trial_type": "A", "condition": "late"},
               {"onset": 1.0, "trial_type": "A", "condition": "early"}]
    back = em.rows_from_events(written)
    assert [r.condition for r in back] == ["early", "late"]


# ── the reader ───────────────────────────────────────────────────────────────

def test_the_reader_keeps_every_column(tmp_path):
    p = os.path.join(str(tmp_path), "x_events.tsv")
    _write(p, [{"onset": "1.0", "duration": "0", "trial_type": "A",
                "condition": "first", "nibs_event_id": "MEP",
                "nibs_position_id": "M1"}],
           ["onset", "duration", "trial_type", "condition",
            "nibs_event_id", "nibs_position_id"])
    rows = read_events_rows(p)
    assert rows[0]["condition"] == "first"
    assert rows[0]["nibs_event_id"] == "MEP"
    assert rows[0]["onset"] == 1.0


def test_a_missing_file_is_not_an_error(tmp_path):
    assert read_events_rows(os.path.join(str(tmp_path), "nope.tsv")) == []


def test_a_file_without_onset_is_ignored(tmp_path):
    p = os.path.join(str(tmp_path), "bad_events.tsv")
    _write(p, [{"a": "1"}], ["a"])
    assert read_events_rows(p) == []


# ── the destructive overwrite ────────────────────────────────────────────────

def test_rewriting_preserves_columns_it_does_not_own(tmp_path):
    """For a converted recording this file IS the BIDS events file. Rewriting
    it with four columns deleted the links to the stimulation description."""
    from mep_cmap.conditions_tab import write_events_tsv_beside

    rec = os.path.join(str(tmp_path), "sub-A_ses-01_emg.edf")
    open(rec, "wb").close()
    events = os.path.join(str(tmp_path), "sub-A_ses-01_events.tsv")
    _write(events, [
        {"onset": "1.0", "duration": "0", "trial_type": "A",
         "condition": "n/a", "nibs_event_id": "Mmax",
         "nibs_position_id": "tibial"},
        {"onset": "2.0", "duration": "0", "trial_type": "G",
         "condition": "n/a", "nibs_event_id": "MEP",
         "nibs_position_id": "M1"}],
        ["onset", "duration", "trial_type", "condition",
         "nibs_event_id", "nibs_position_id"])

    write_events_tsv_beside(rec, [
        {"onset": 1.0, "duration": 0, "trial_type": "A", "condition": "first"},
        {"onset": 2.0, "duration": 0, "trial_type": "G", "condition": "last"}])

    back = read_events_rows(events)
    assert [r["condition"] for r in back] == ["first", "last"]
    # The links survive, matched by onset.
    assert [r["nibs_event_id"] for r in back] == ["Mmax", "MEP"]
    assert [r["nibs_position_id"] for r in back] == ["tibial", "M1"]


def test_writing_where_no_file_existed_is_unchanged(tmp_path):
    from mep_cmap.conditions_tab import write_events_tsv_beside

    rec = os.path.join(str(tmp_path), "sub-B_ses-01_emg.edf")
    open(rec, "wb").close()
    write_events_tsv_beside(rec, [
        {"onset": 1.0, "duration": 0, "trial_type": "A", "condition": "x"}])
    rows = read_events_rows(os.path.join(str(tmp_path),
                                         "sub-B_ses-01_events.tsv"))
    assert set(rows[0]) == {"onset", "duration", "trial_type", "condition"}


def test_an_event_with_no_match_gets_na_not_a_wrong_link(tmp_path):
    """Better an honest gap than a delivery pointed at the wrong protocol."""
    from mep_cmap.conditions_tab import write_events_tsv_beside

    rec = os.path.join(str(tmp_path), "sub-C_ses-01_emg.edf")
    open(rec, "wb").close()
    events = os.path.join(str(tmp_path), "sub-C_ses-01_events.tsv")
    _write(events, [{"onset": "1.0", "duration": "0", "trial_type": "A",
                     "condition": "n/a", "nibs_event_id": "Mmax"}],
           ["onset", "duration", "trial_type", "condition", "nibs_event_id"])

    write_events_tsv_beside(rec, [
        {"onset": 1.0, "duration": 0, "trial_type": "A", "condition": "x"},
        {"onset": 99.0, "duration": 0, "trial_type": "A", "condition": "x"}])

    back = read_events_rows(events)
    assert back[0]["nibs_event_id"] == "Mmax"
    assert back[1]["nibs_event_id"] == "n/a"
