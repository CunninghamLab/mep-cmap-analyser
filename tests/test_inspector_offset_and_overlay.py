"""
Tests for the Inspector's MEP-offset marker, read-out and median overlay.

inspector.py cannot be imported here -- it pulls in matplotlib's Tk backend,
which is unavailable on a headless runner. So the decision logic lives in
detection/offset_detection.py where it can be exercised directly, and the Tk
wiring is checked by reading the source. Dragging itself has to be tested by
hand; these tests cover everything that can be established without a display.
"""

import ast
import pathlib

import numpy as np
import pytest

from mep_cmap.detection import offset_marker_field, resolve_mep_offset

SRC = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "inspector.py").read_text(encoding="utf-8")


# ── One marker, not two ───────────────────────────────────────────────────────

def test_csp_start_doubles_as_the_offset_marker():
    """
    During contraction the end of the MEP and the start of the silent period
    are the same physical event. Two draggable markers for one event can be
    moved apart, and the file then holds two different answers to the same
    question with nothing to say which is right.
    """
    assert offset_marker_field(True, True) == "silent_start_idx"


def test_a_dedicated_marker_is_used_when_there_is_no_silent_period():
    assert offset_marker_field(False, True) == "mep_offset_idx"
    assert offset_marker_field(False, False) == "mep_offset_idx"
    assert offset_marker_field(True, False) == "mep_offset_idx"


def test_the_marker_rule_matches_the_quantification_precedence():
    """
    The Inspector and the pipeline must agree about which marker is the offset,
    or what the analyst drags is not what gets reported.
    """
    rng = np.random.default_rng(0)
    fs, pre = 5000.0, 20.0
    n = int((pre + 300) * fs / 1000)
    sig = rng.normal(0, 0.01, n)

    # cSP present: quantification takes the cSP start, and the marker rule
    # points at the same field.
    r = resolve_mep_offset(sig, fs, onset_ms=22.0, csp_start_ms=55.0,
                           csp_enabled=True)
    assert r.source == "csp_start"
    assert offset_marker_field(True, True) == "silent_start_idx"

    # No cSP: quantification falls through to the envelope, and the marker is
    # the dedicated one.
    r2 = resolve_mep_offset(sig, fs, onset_ms=22.0, csp_start_ms=None,
                            csp_enabled=False, pre_ms=pre, search_end_ms=200)
    assert r2.source in ("envelope", "none")
    assert offset_marker_field(False, False) == "mep_offset_idx"


def test_a_manual_marker_outranks_both():
    rng = np.random.default_rng(1)
    sig = rng.normal(0, 0.01, 1600)
    r = resolve_mep_offset(sig, 5000.0, onset_ms=22.0, manual_offset_ms=41.0,
                           csp_start_ms=55.0, csp_enabled=True)
    assert r.source == "manual" and r.offset_ms == 41.0


# ── The Inspector wiring ──────────────────────────────────────────────────────

def test_inspector_uses_the_shared_marker_rule():
    """The rule must be stated once, not reimplemented in the interface."""
    assert "offset_marker_field(" in SRC
    assert "from .detection import" in SRC


def test_inspector_registers_a_colour_for_the_offset_marker():
    assert '"mep_offset_idx"' in SRC
    a = SRC.index("DOT_COLOURS = {")
    b = SRC.index("}", a)
    assert "mep_offset_idx" in SRC[a:b], "the marker has no colour assigned"


def test_offset_marker_is_included_in_the_stale_index_cleanup():
    """
    Stored indices persist across runs. One saved when the analysis window was
    longer would be reused against a shorter segment, and scatter() raises
    mid-draw so the Inspector never opens at all. The existing markers are
    already screened; a new one has to be screened too.
    """
    a = SRC.index("_stale = [f for f in (")
    b = SRC.index("]", a)
    assert "mep_offset_idx" in SRC[a:b]


def test_toggling_csp_clears_a_stale_offset_marker():
    """
    Turning the silent period on or off changes WHICH marker carries the
    offset, so a value stored under the other rule no longer applies.
    """
    assert "m.pop('mep_offset_idx',      None)" in SRC


def test_seeding_never_overwrites_a_dragged_value():
    """
    The envelope detector places the marker on first view only. Re-seeding on
    every redraw would silently undo a manual decision the next time the trial
    was displayed.
    """
    assert "if 'mep_offset_idx' not in m:" in SRC
    a = SRC.index("def _seed_offset_idx")
    b = SRC.index("\n    def ", a + 10)
    body = SRC[a:b]
    assert "Only ever used to place the marker the first time" in body


def test_readout_shows_offset_and_duration():
    a = SRC.index("def _refresh_status")
    body = SRC[a:]
    assert "Offset:" in body
    assert "Duration:" in body
    assert "offset_marker_field(" in body, (
        "the read-out must use the same rule as the marker and the pipeline"
    )


def test_the_offset_text_actually_reaches_the_status_bar():
    """
    Checking that the text is BUILT is not the same as checking it is SHOWN.

    The first version of this feature computed offset_txt correctly and then
    left it out of the status.config call, so the value was assembled on every
    redraw and discarded. The read-out looked untouched while every test
    passed, because the tests searched the method body for "Offset:" -- which
    the dead variable satisfied.

    This asserts the variable is interpolated into the string that is actually
    displayed.
    """
    a = SRC.index("self.status.config(")
    b = SRC.index("))", a)
    shown = SRC[a:b]
    assert "{offset_txt}" in shown, (
        "offset_txt is computed but never displayed"
    )
    # The same trap applies to every other field in that bar.
    for var in ("{silent_txt}", "{auc_txt}", "{csp_note}"):
        assert var in shown


def test_no_read_out_variable_is_computed_and_discarded():
    """Generalises the above: anything named *_txt must be displayed.

    This has now caught two separate omissions -- offset_txt built and left out
    of the status line, and lat_txt the same. Both looked correct in the source
    and showed nothing on screen.
    """
    import re

    a = SRC.index("def _refresh_status")
    # _refresh_status is the last method in the class, so there may be no
    # following "def" to slice against; fall back to end of file.
    nxt = SRC.find("\n    def ", a + 10)
    body = SRC[a:] if nxt == -1 else SRC[a:nxt]
    built = set(re.findall(r"^\s*(\w+_txt)\s*=", body, re.M))
    c = body.index("self.status.config(")
    d = body.index("))", c)
    shown = body[c:d]
    for name in sorted(built):
        assert "{" + name + "}" in shown, (
            f"{name} is built in _refresh_status but never displayed"
        )


# ── The median overlay ────────────────────────────────────────────────────────

def test_overlay_reuses_the_detection_template():
    """
    Drawing the same waveform the detector compared against means the analyst
    sees what the algorithm saw, rather than a second, subtly different
    average computed for display.
    """
    a = SRC.index("if self.show_median_var.get():")
    b = SRC.index("self.ax_raw.plot(self.t, emg", a)
    assert "self._condition_template()" in SRC[a:b]


def test_overlay_is_drawn_behind_the_trial():
    a = SRC.index("if self.show_median_var.get():")
    b = SRC.index("self.ax_raw.plot(self.t, emg", a)
    body = SRC[a:b]
    assert "zorder=1" in body
    assert "alpha=" in body
    assert SRC.index("show_median_var.get()") < SRC.index(
        "self.ax_raw.plot(self.t, emg"), "the median must be plotted first"


def test_overlay_has_a_checkbox_that_redraws():
    assert "Show event-type median" in SRC
    a = SRC.index("Show event-type median")
    b = SRC.index(")", SRC.index("command=", a))
    assert "_plot" in SRC[a:b], "the checkbox must trigger a redraw"


def test_overlay_defaults_to_off():
    a = SRC.index("self.show_median_var = tk.BooleanVar(")
    assert "value=False" in SRC[a:a + 80]


def test_inspector_still_parses_and_defines_the_new_methods():
    tree = ast.parse(SRC)
    names = {n.name for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef)}
    for fn in ("_seed_offset_idx", "_condition_template", "_refresh_status"):
        assert fn in names


# ── AUC is integrated from onset to offset ───────────────────────────────────

def test_auc_end_is_seeded_from_the_offset_not_a_fixed_width():
    """
    The AUC end used to default to onset + 50 ms, a fixed width unrelated to
    where the response actually ends. The pipeline integrates onset to offset,
    so the window shown during review and the window the results file used were
    different quantities. 50 ms survives only as a last resort.
    """
    a = SRC.index('if "auc_start_idx" not in m and "onset_idx" in m:')
    b = SRC.index('m["auc_end_idx"]   = int(a1)', a)
    body = SRC[a:b]
    assert "offset_marker_field(" in body, (
        "the AUC end must be seeded from whichever marker carries the offset"
    )
    # The fixed width is now a fallback, reached only when no offset exists.
    fallback = body.index("a0 + int(50 * fs / 1000)")
    guard = body.index("if a1 is None or int(a1) <= a0:")
    assert guard < fallback, "50 ms must be the fallback, not the default"


def test_moving_the_offset_marker_moves_the_auc_end():
    a = SRC.index("def _update_meta")
    b = SRC.index("\n    def ", a + 10)
    body = SRC[a:b]
    assert 'm["auc_end_idx"] = new_idx' in body
    assert "offset_marker_field(" in body, (
        "the link must follow the same rule as the marker and the pipeline"
    )


def test_moving_the_auc_end_moves_the_offset_marker():
    """
    While linked the two are one quantity, so the link has to work both ways.
    One-directional linking lets the analyst drag them apart, and the file then
    records two answers to the same question.
    """
    a = SRC.index("def _on_end_moved")
    b = SRC.index("self._auc_lines = [", a)
    body = SRC[a:b]
    assert "offset_marker_field(" in body
    assert "m[_f] = new_idx" in body
    assert "dp.role == _f" in body, "the marker on screen must move too"


def test_the_link_is_defeatable():
    """Unticking must leave both edges free, as before."""
    for fn in ("_update_meta", "_on_end_moved"):
        a = SRC.index(f"def {fn}")
        b = SRC.index("\n    def ", a + 10) if fn == "_update_meta" \
            else SRC.index("self._auc_lines = [", a)
        assert "link_onset_auc.get()" in SRC[a:b], (
            f"{fn} must honour the link checkbox"
        )


def test_the_checkbox_names_both_ends():
    assert "Link AUC to onset & offset" in SRC
    assert "Link onset → AUC" not in SRC, "the old one-sided label remains"


def test_background_csp_does_not_end_the_auc_for_non_csp_types():
    """
    The Inspector used to run a cortical-silent-period detector for EVERY
    stimulus type, purely to place the AUC end. On a resting M-wave that is
    asking a question the data cannot answer -- there is no contraction to
    silence -- and the pipeline never does it, computing a cSP only for the
    types assigned to it. The same trial therefore had one AUC on screen and a
    different one in the results file.
    """
    a = SRC.index("# ---------- auto AUC:")
    b = SRC.index("self.ax_raw.clear()", a)
    body = SRC[a:b]
    assert "if not self.enable_silent.get():" in body, (
        "the background cSP search must be limited to cSP stimulus types"
    )
    assert "_csp_end_for_auc = None" in body


def test_the_link_is_enforced_on_load_not_only_on_drag():
    """
    The analysis stores auc_start_idx and auc_end_idx in the segment metadata,
    so a reviewed file arrives with a window already set and the seeding path
    -- guarded on the key being absent -- never runs. The AUC end then sat tens
    of milliseconds from the offset marker while the checkbox said the two were
    linked.

    "Linked" has to mean they agree whenever they are shown, not merely that
    they move together once touched.
    """
    a = SRC.index("# Enforce the link on LOAD")
    b = SRC.index('if "auc_start_idx" in m and "auc_end_idx" in m:', a)
    body = SRC[a:b]
    assert "link_onset_auc.get()" in body
    assert 'm["auc_end_idx"] = int(m[_f])' in body
    assert 'm["auc_start_idx"] = int(m["onset_idx"])' in body


def test_enforcement_runs_after_the_offset_marker_is_seeded():
    """Otherwise the AUC end is reconciled against a marker that does not exist yet."""
    seed = SRC.index("m['mep_offset_idx'] = _seed")
    enforce = SRC.index("# Enforce the link on LOAD")
    # `self._auc_lines = []` also appears in __init__, so search for the
    # construction rather than the first assignment.
    lines = SRC.index("DraggableLine(self.ax_raw, self.t,")
    assert seed < enforce < lines, (
        "order must be: seed the offset marker, reconcile the AUC window, "
        "then build the draggable lines from the reconciled values"
    )


def test_toggling_the_link_redraws():
    """Ticking it must reconcile immediately, not at the next trial."""
    a = SRC.index('text="Link AUC to onset & offset"')
    b = SRC.index("pack(", a)
    assert "command=" in SRC[a:b]


# ── A failed detection must not become a measurement ─────────────────────────

def test_a_failed_detection_is_flagged_rather_than_shown_as_zero():
    """
    dispatch_onset returns None when it cannot find an onset. The marker then
    falls back to the stimulus index, which reads as "Latency: 0.0 ms" -- a
    number that looks like a measurement and is not one.
    """
    a = SRC.index("if onset_ms is None and 'onset_idx' not in m:")
    b = SRC.index("m.setdefault('onset_idx',   onset)", a)
    assert "m['onset_auto_failed'] = True" in SRC[a:b]


def test_the_readout_says_not_detected():
    a = SRC.index("def _refresh_status")
    body = SRC[a:]
    assert '"Latency: not detected"' in body
    assert "_no_onset" in body


def test_nothing_derived_from_a_non_detection_is_reported():
    """Offset and duration are measured FROM the onset; without one they mean nothing."""
    a = SRC.index("def _refresh_status")
    body = SRC[a:]
    assert "_fld = None if _no_onset else offset_marker_field(" in body


def test_dragging_the_onset_clears_the_flag():
    """A marker the analyst placed is a measurement, whatever the detector managed."""
    a = SRC.index("def _update_meta")
    b = SRC.index("\n    def ", a + 10)
    body = SRC[a:b]
    assert 'if field == "onset_idx":' in body
    assert 'm.pop("onset_auto_failed", None)' in body


def test_the_fallback_index_is_not_exported_on_save():
    """
    The analysis honours a stored onset_idx as a manual override. Exporting the
    fallback would turn a correctly blank latency into a measured 0.0 ms, and
    would carry the offset, duration and area window along with it.
    """
    a = SRC.index("def _close_and_save")
    b = SRC.index("self.top.destroy()", a)
    body = SRC[a:b]
    assert "onset_auto_failed" in body
    for field in ("onset_idx", "mep_offset_idx", "auc_start_idx", "auc_end_idx"):
        assert f"'{field}'" in body, f"{field} survives a non-detection"


def test_the_flag_itself_never_reaches_the_results():
    a = SRC.index("def _close_and_save")
    b = SRC.index("self.top.destroy()", a)
    assert "_m.pop('onset_auto_failed', None)" in SRC[a:b]


# ── Events queued before the window closed must not touch dead widgets ───────

def test_navigation_is_guarded_against_a_destroyed_window():
    """
    Tk delivers events that were already queued when a widget was destroyed.
    The Right and Left bindings step through trials and end in _plot, which
    touches the note box -- so pressing an arrow key as the window closes
    raised:

        _tkinter.TclError: invalid command name ".!toplevel3.!frame4.!scrolledtext"

    In a multi-channel run two Inspectors open in succession, so the window
    closes twice as often and the race is easy to hit.
    """
    for fn in ("_next", "_prev", "_plot", "_save_note_from_widget"):
        a = SRC.index(f"    def {fn}(self")
        b = SRC.index("\n", SRC.index("\n", a) + 1)
        head = SRC[a:b + 200]
        assert "_closed()" in head, (
            f"{fn} can run after the window is destroyed"
        )


def test_the_guard_checks_the_widgets_the_redraw_touches():
    """
    Checking the Toplevel alone was not enough. It reported itself as existing
    while a child had already been destroyed, and the redraw then failed on the
    child:

        _tkinter.TclError: invalid command name ".!toplevel3.!frame4.!scrolledtext"

    Reasoning about how a parent outlives its child during Tk teardown would
    need the exact answer to be right and would break again if it changed.
    Checking the widgets actually used is correct whatever the mechanism.
    """
    a = SRC.index("def _closed(self)")
    b = SRC.index("\n    def ", a + 10)
    body = SRC[a:b]
    assert "note_box" in body, "the widget that failed is not checked"
    assert "_widget_alive" in body


def test_the_guard_is_not_given_a_non_tk_object():
    """
    self.canvas is a FigureCanvasTkAgg and has no winfo_exists. Passing one to
    _widget_alive would raise, be caught, and report the window as dead --
    permanently, on every redraw, disabling the Inspector entirely.
    """
    a = SRC.index("def _closed(self)")
    b = SRC.index("\n    def ", a + 10)
    assert "self.canvas" not in SRC[a:b], (
        "a matplotlib canvas is not a Tk widget; use get_tk_widget()"
    )
    doc = SRC[SRC.index("def _widget_alive"):SRC.index("def _closed")]
    assert "FigureCanvasTkAgg" in doc, "the constraint should be recorded"


def test_the_note_box_write_is_guarded_at_the_point_of_use():
    """Belt and braces: the specific call in the traceback."""
    a = SRC.index('self.note_box.delete("1.0", "end")')
    assert "_widget_alive(self.note_box)" in SRC[a - 200:a]


def test_closing_sets_the_flag_before_anything_else():
    """
    winfo_exists() is still true while the close handler runs, so the flag has
    to be set first or a queued event slipping in during the save would pass
    the check.
    """
    a = SRC.index("def _close_and_save")
    b = SRC.index("self.top.destroy()", a)
    body = SRC[a:b]
    assert "self._is_closing = True" in body
    assert body.index("_is_closing = True") < body.index("note_box")


def test_the_flag_is_initialised_so_the_first_draw_works():
    a = SRC.index("self._is_closing = False")
    assert a < SRC.index("def _closed"), (
        "the flag must exist before any handler can consult it"
    )
