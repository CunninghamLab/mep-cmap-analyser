"""
Which source an EDF's events came from.

A sibling BIDS ``_events.tsv`` takes precedence over the annotations inside
the file, and nothing said so. A .tsv written from an earlier or cropped run
therefore replaced the recording's own markers silently: stimulus types went
missing from the setup table, and the events stopped where that run stopped,
on a file whose own annotations ran to the end. Two symptoms, one precedence
rule, and no way to tell from the interface which source was in use.
"""

import csv
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
APP = (PKG / "app.py").read_text(encoding="utf-8")


def _write_tsv(path, types):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["onset", "duration", "trial_type"])
        for i, t in enumerate(types):
            w.writerow([f"{100 + i:.3f}", "0", t])


@pytest.fixture
def pair(tmp_path, monkeypatch):
    """An EDF path with a sibling .tsv, and stubbed annotations."""
    edf = tmp_path / "sub-01_task-x_emg.edf"
    tsv = tmp_path / "sub-01_task-x_events.tsv"
    edf.write_bytes(b"")
    return edf, tsv


def test_nothing_is_reported_when_there_is_no_tsv(pair):
    from mep_cmap.formats import edf as E
    edf, _tsv = pair
    assert E.event_source_summary(str(edf)) == ""


def test_the_tsv_is_named_when_it_wins(pair, monkeypatch):
    from mep_cmap.formats import edf as E
    edf, tsv = pair
    _write_tsv(tsv, ["A"] * 3 + ["B"] * 2)
    monkeypatch.setattr(E, "_read_annotations", lambda _p: None)
    line = E.event_source_summary(str(edf))
    assert "events.tsv" in line
    assert "takes precedence" in line
    assert "5 event(s)" in line


def test_a_type_present_only_in_the_file_is_called_out(pair, monkeypatch):
    """The reported symptom: H exists in the recording and not on the tab."""
    from mep_cmap.formats import edf as E
    edf, tsv = pair
    _write_tsv(tsv, ["A", "B", "C"])
    monkeypatch.setattr(E, "_read_annotations",
                        lambda _p: {"A": [1.0], "B": [2.0], "C": [3.0],
                                    "H": [4.0, 5.0]})
    line = E.event_source_summary(str(edf))
    assert "H appear only there" in line
    assert "Delete or regenerate" in line, "the message must say what to do"


def test_a_differing_count_is_called_out_even_with_the_same_types(pair,
                                                                 monkeypatch):
    """The other symptom: events stopping early because the .tsv was cropped."""
    from mep_cmap.formats import edf as E
    edf, tsv = pair
    _write_tsv(tsv, ["A", "A"])
    monkeypatch.setattr(E, "_read_annotations", lambda _p: {"A": [1.0] * 40})
    line = E.event_source_summary(str(edf))
    assert "40 event(s)" in line


def test_agreement_is_not_reported_as_a_problem(pair, monkeypatch):
    from mep_cmap.formats import edf as E
    edf, tsv = pair
    _write_tsv(tsv, ["A", "B"])
    monkeypatch.setattr(E, "_read_annotations",
                        lambda _p: {"A": [1.0], "B": [2.0]})
    line = E.event_source_summary(str(edf))
    assert "appear only there" not in line
    assert "Delete or regenerate" not in line


def test_unreadable_annotations_do_not_break_the_report(pair, monkeypatch):
    """A stub .edf cannot be opened; the .tsv half must still be reported."""
    from mep_cmap.formats import edf as E
    edf, tsv = pair
    _write_tsv(tsv, ["A"])
    def _boom(_p):
        raise OSError("not a real EDF")
    monkeypatch.setattr(E, "_read_annotations", _boom)
    assert "events.tsv" in E.event_source_summary(str(edf))


def test_the_load_flow_reports_it():
    body = APP[APP.index("elif _fmt == 'edf':"):]
    body = body[:body.index("elif _fmt in (")]
    assert "event_source_summary" in body
    assert "self.log(" in body


def test_an_edf_defaults_to_analysing_every_type():
    """It set marker_choice to 'A', which since narrowing was added would
    restrict the analysis to one stimulus type on a file carrying eight."""
    body = APP[APP.index("elif _fmt == 'edf':"):]
    body = body[:body.index("elif _fmt in (")]
    assert "ALL_MARKERS" in body
    assert "self.marker_choice.set('A')" not in body
