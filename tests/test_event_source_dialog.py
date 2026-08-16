"""
The event-source dialogue.

A threshold level is not checkable by reading it: two volts is right or wrong
depending on the trigger's amplitude, its baseline, and whether the pulse rings,
none of which is visible from the box it was typed into. So the dialogue's real
content is the preview -- the trace with the level drawn across it and every
detected crossing marked -- and a count that changes as the level does.

Tk cannot run here, so the dialogue's construction is checked by reading it and
its decimation is exercised directly.
"""

import pathlib

import numpy as np
import pytest

from mep_cmap.event_sources import decimate_for_preview

DLG = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "event_source_dialog.py").read_text(encoding="utf-8")
APP = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "app.py").read_text(encoding="utf-8")

FS = 5000.0


# ── The preview must show what the detector sees ─────────────────────────────

def _recording_with_spikes(n_spikes=200, dur_s=2000.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.01, int(dur_s * FS))
    pos = sorted(rng.integers(0, x.size - 10, n_spikes))
    for i in pos:
        x[i:i + 3] = 5.0
    return x


def test_decimation_preserves_the_transients_the_detector_finds():
    """
    A stimulus trigger is a one-sample spike. Plain subsampling drew a flat
    line on a two-thousand-second recording while the detector found two
    hundred events -- a preview that would have led the analyst to set a level
    against a trace showing none of the pulses.
    """
    x = _recording_with_spikes()
    t, lo, hi = decimate_for_preview(x, FS, max_points=4000)

    kept = int((hi > 2.5).sum())
    assert kept >= 190, f"only {kept} of 200 pulses survived decimation"

    step = x.size // 4000
    naive = int((x[::step] > 2.5).sum())
    assert naive < 10, (
        "the fixture no longer demonstrates the problem: plain subsampling "
        f"kept {naive} pulses"
    )


def test_decimation_returns_a_band_not_a_line():
    x = _recording_with_spikes()
    t, lo, hi = decimate_for_preview(x, FS, max_points=1000)
    assert len(t) == len(lo) == len(hi)
    assert np.any(hi > lo), "min and max are identical; nothing was summarised"


def test_a_short_recording_is_not_decimated():
    x = np.arange(100, dtype=float)
    t, lo, hi = decimate_for_preview(x, FS, max_points=4000)
    assert np.allclose(lo, x) and np.allclose(hi, x)


def test_the_tail_is_not_dropped():
    """A recording whose length is not a multiple of the step still ends where
    it ends; losing the last block would hide events at the end of a session."""
    x = np.zeros(10_007)
    x[-3:] = 5.0
    t, lo, hi = decimate_for_preview(x, FS, max_points=100)
    assert hi.max() == 5.0, "the final partial block was discarded"


def test_an_empty_signal_does_not_raise():
    t, lo, hi = decimate_for_preview([], FS)
    assert len(t) == 0


# ── The dialogue ─────────────────────────────────────────────────────────────

def test_the_preview_draws_the_level_and_the_crossings():
    assert "axhline(src.level" in DLG
    assert "detect_threshold_crossings(" in DLG
    assert "fill_between" in DLG, "the trace must be a min/max band"


def test_the_count_updates_with_the_level():
    a = DLG.index("def _update_preview")
    b = DLG.index("\n    def ", a + 10)
    body = DLG[a:b]
    assert "count_var.set(" in body
    assert "event(s) detected" in body


def test_editing_a_field_refreshes_the_preview():
    assert 'trace_add("write"' in DLG
    a = DLG.index("def _on_edit")
    b = DLG.index("\n    def ", a + 10)
    assert "_update_preview()" in DLG[a:b]


def test_a_half_typed_number_is_not_rejected():
    """
    The analyst is mid-keystroke. Rejecting an unparseable value would fight
    the typing; the preview simply does not update until it parses.
    """
    a = DLG.index("def _apply_edits")
    b = DLG.index("\n    def ", a + 10)
    body = DLG[a:b]
    assert "except (TypeError, ValueError)" in body
    assert "pass" in body


def test_the_channel_is_read_once_not_per_keystroke():
    """Re-reading a long recording on every character would make it unusable."""
    a = DLG.index("def _channel_data")
    b = DLG.index("\n    def ", a + 10)
    assert "self._cache" in DLG[a:b]


def test_an_unreadable_channel_is_reported_in_the_preview():
    a = DLG.index("def _update_preview")
    b = DLG.index("\n    def ", a + 10)
    assert "Could not read" in DLG[a:b]


def test_the_interval_kind_says_nothing_is_detected():
    """The times are asserted; no part of the recording can confirm them."""
    assert "Nothing is detected" in DLG


# ── Both entry points ────────────────────────────────────────────────────────

def test_the_dialogue_is_reachable_from_tab_1a_and_channel_assignment():
    """
    One dialogue, two entry points: the first setup of a file is when the
    question arises, and it must be revisable afterwards without reopening the
    file.
    """
    assert APP.count("self._open_event_sources") >= 3, (
        "expected a definition plus a button in tab 1a and in each channel "
        "dialogue"
    )
    # Anchor on the BUTTON, not the phrase: the same words appear in the
    # run-gate warning, and searching from the start of the file finds that.
    a = APP.index('command=self._copy_setup_to_all_channels')
    assert "Event sources" in APP[a:a + 400], "no button on tab 1a"


def test_choosing_sources_rebuilds_the_stimulus_table():
    a = APP.index("def _apply_event_sources")
    b = APP.index("\n    def ", a + 10)
    body = APP[a:b]
    assert "extract_events(" in body
    assert "_build_labels_tab(" in body
    assert "warnings" in body, "merge warnings must reach the analyst"


def test_sources_are_saved_with_the_session_and_cleared_with_the_file():
    assert '"event_sources": [_s.to_dict()' in APP
    assert 'sess.get("event_sources")' in APP
    a = APP.index("def _reset_state_for_new_file")
    b = APP.index("\n    def ", a + 10)
    assert "self.event_sources = []" in APP[a:b], (
        "sources describe one recording's channels; they cannot carry over"
    )


# ── It must look alive the moment it opens ───────────────────────────────────

def test_the_dialogue_never_opens_with_nothing_selected():
    """
    An empty list meant an empty editor and an empty preview, so the dialogue
    looked broken until the analyst guessed that Add came first. A file with no
    configured sources uses its own markers; that is what the first row now
    describes -- the existing behaviour made visible and editable rather than
    implied by an empty list.
    """
    a = DLG.index("if not self._sources:")
    b = DLG.index("self._load_selected()", a)
    body = DLG[a:b]
    assert 'EventSource(' in body
    assert 'kind="embedded"' in body
    assert "self.listbox.selection_set(0)" in DLG


def test_there_is_an_explicit_preview_control():
    assert 'text="Preview"' in DLG
    assert "def _preview_now" in DLG
    a = DLG.index("def _preview_now")
    b = DLG.index("\n    def ", a + 10)
    body = DLG[a:b]
    assert "_apply_edits()" in body, "it must use what is currently typed"
    assert "force=True" in body


def test_selection_changes_force_the_draw_rather_than_scheduling_it():
    """
    draw_idle only schedules. After Add, the scheduled draw had not run by the
    time the analyst looked, so a newly added source appeared to have no
    preview until its row was clicked again.
    """
    assert "self.canvas.draw() if force else self.canvas.draw_idle()" in DLG
    for site in ("def _add", "def _remove"):
        a = DLG.index(site)
        b = DLG.index("\n    def ", a + 10)
        assert "force=True" in DLG[a:b], f"{site} does not force the redraw"
    a = DLG.index('self.listbox.bind("<<ListboxSelect>>"')
    assert "force=True" in DLG[a:a + 160]


def test_removing_the_last_source_leaves_a_valid_selection():
    """Otherwise the editor empties and the dialogue looks broken again."""
    a = DLG.index("def _remove")
    b = DLG.index("\n    def ", a + 10)
    body = DLG[a:b]
    assert "min(i, len(self._sources) - 1)" in body
