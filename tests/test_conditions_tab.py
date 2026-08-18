"""
Setup ▸ Conditions — the surface.

The rules are tested in test_conditions.py without a display. What is checked
here is the wiring: that the tab reaches the model rather than reimplementing
it, that Apply cannot write a table the model would refuse, and that the events
file it writes is the one the reader looks for.
"""

import ast
import csv
import json
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
TAB = (PKG / "conditions_tab.py").read_text(encoding="utf-8")
APP = (PKG / "app.py").read_text(encoding="utf-8")


def _method(name, src=TAB):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant):
                body = body[1:]
            return "\n".join(ast.unparse(n) for n in body)
    raise AssertionError(f"{name} not found")


# ── wiring ───────────────────────────────────────────────────────────────────

def test_the_tab_sits_between_dataset_and_bidsify():
    """Conditions describe the recording, so they are decided before it is
    converted or analysed."""
    i = APP.index('text="Dataset"')
    j = APP.index('text="Conditions"')
    k = APP.index('text="BIDS-ify"')
    assert i < j < k


def test_the_mixin_is_registered_where_attributes_are_checked():
    """A mixin absent from that list reads as undefined attributes -- the same
    failure David's fork still has for its own tab."""
    t = (pathlib.Path(__file__).parent / "test_no_missing_attributes.py"
         ).read_text(encoding="utf-8")
    assert "conditions_tab.py" in t


def test_the_tab_does_not_reimplement_the_rules():
    """Parsing, splitting and completeness live in conditions.py, which is
    tested without a display. A second copy here would be the copy that drifts.
    """
    for fn, call in (("_cond_split", "C.split_row("),
                     ("_cond_autofill", "C.autofill("),
                     ("_cond_rename", "C.sanitise_name("),
                     ("_cond_apply", "C.validate(")):
        assert call in _method(fn), f"{fn} should delegate to {call}"


def test_apply_validates_before_writing_anything():
    """A file written from a table the model would refuse is a file that
    disagrees with the analysis."""
    body = _method("_cond_apply")
    assert body.index("C.validate(") < body.index("to_event_rows")


def test_apply_composes_through_the_one_function():
    assert "C.group_events(event_rows)" in _method("_cond_apply")


def test_apply_hands_the_groups_to_the_analysis():
    body = _method("_cond_apply")
    assert "self.condition_event_rows = event_rows" in body
    assert "self.condition_map = decoded" in body


def test_a_failed_write_does_not_discard_the_assignment():
    """The conditions are still applied to the session; only the file is
    missing, and the message says so."""
    body = _method("_cond_apply")
    i = body.index("write_events_tsv_beside")
    j = body.index("self.condition_event_rows")
    assert i < j, "the assignment must survive a write failure"
    assert "could not be written" in body


# ── the events file ──────────────────────────────────────────────────────────

def test_the_written_path_matches_what_the_reader_looks_for():
    """What this writes must be what it reads on the next load."""
    from mep_cmap.conditions_tab import events_tsv_path
    from mep_cmap.formats.edf import _events_tsv_path
    for name in ("sub-01_ses-1_task-x_emg.edf", "sub-01_run-2_emg.edf"):
        assert events_tsv_path("/data/" + name) == _events_tsv_path("/data/" + name)


def test_the_file_round_trips_through_the_reader(tmp_path):
    from mep_cmap import conditions as C
    from mep_cmap.conditions_tab import write_events_tsv_beside
    from mep_cmap.formats.edf import _read_events_tsv

    rec = tmp_path / "sub-01_task-x_emg.edf"
    rec.write_bytes(b"")
    stim = {"A": [1.0, 2.0, 3.0, 4.0]}
    rows = C.validate(C.split_row(C.rows_from_events(stim), 0, [0, 1],
                                  new_condition="pre",
                                  keep_condition="post"), stim)
    ev = C.to_event_rows(rows, stim)
    path = write_events_tsv_beside(str(rec), ev)

    back = _read_events_tsv(path)
    assert back == {"A": [1.0, 2.0, 3.0, 4.0]}, \
        "the reader groups by trial_type; conditions come through group_events"

    with open(path, encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh, delimiter="\t"))
    assert [r["condition"] for r in got] == ["pre", "pre", "post", "post"]
    groups, decoded = C.group_events(got)
    assert sorted(groups) == ["A\u00b7post", "A\u00b7pre"]
    assert decoded["A\u00b7pre"] == ("A", "pre")


def test_an_excluded_trial_is_written_as_na(tmp_path):
    from mep_cmap import conditions as C
    from mep_cmap.conditions_tab import write_events_tsv_beside

    rec = tmp_path / "sub-01_task-x_emg.edf"
    rec.write_bytes(b"")
    stim = {"A": [1.0, 2.0]}
    rows = [C.ConditionRow("A", "keep", (0,)),
            C.ConditionRow("A", "", (1,), excluded=True)]
    path = write_events_tsv_beside(str(rec), C.to_event_rows(rows, stim))
    with open(path, encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh, delimiter="\t"))
    assert [r["trial_type"] for r in got] == ["A", "n/a"]


def test_the_sidecar_documents_the_extra_column(tmp_path):
    """condition is not a BIDS-defined column, and an undocumented extra column
    is one only its author can interpret."""
    from mep_cmap import conditions as C
    from mep_cmap.conditions_tab import write_events_tsv_beside

    rec = tmp_path / "sub-01_task-x_emg.edf"
    rec.write_bytes(b"")
    path = write_events_tsv_beside(
        str(rec), C.to_event_rows([C.ConditionRow("A", "pre", (0,))],
                                  {"A": [1.0]}))
    side = pathlib.Path(path).with_suffix(".json")
    meta = json.loads(side.read_text(encoding="utf-8"))
    assert set(meta) == {"onset", "duration", "trial_type", "condition"}
    assert "Description" in meta["condition"]
    assert "n/a" in meta["trial_type"]["Description"], \
        "the sidecar should say what n/a means, since it marks an exclusion"


# ── the review pane ──────────────────────────────────────────────────────────

def test_no_detection_markers_are_drawn():
    """These are raw epochs before any analysis.

    Onset and amplitude belong to Preview detection, which shows one trial at a
    time because a dozen sets of markers on one axes cannot be read.
    """
    body = _method("_cond_draw")
    for marker in ("onset", "PTP", "offset", "dispatch_onset"):
        assert marker not in body


def test_several_conditions_are_drawn_in_distinct_colours():
    """Comparing two is how a split is judged before it is applied."""
    body = _method("_cond_draw")
    assert "_CYCLE[n % len(_CYCLE)]" in body
    assert "legend" in body


def test_a_large_selection_falls_back_to_the_average():
    """Two hundred translucent lines is neither readable nor quick."""
    from mep_cmap.conditions_tab import OVERLAY_LIMIT
    assert 50 <= OVERLAY_LIMIT <= 300
    body = _method("_cond_draw")
    assert "len(usable) <= OVERLAY_LIMIT" in body
    assert "len(usable) > OVERLAY_LIMIT" in body


def test_the_review_window_matches_the_analysis_window():
    """What is judged here is what will be measured."""
    body = _method("_cond_load_segments")
    assert "window_samples(" in body
    # ast.unparse normalises quotes, so match on the key rather than the call
    assert "window_map" in body


def test_the_trial_list_keeps_its_selection():
    """Without exportselection=False, selecting in the table clears the list:
    Tk hands the X selection to whichever widget was touched last."""
    assert "exportselection=False" in TAB


def test_reloading_is_explicit():
    """Returning to check a waveform should not silently undo an assignment."""
    assert 'text="⟳ Reload from file"' in TAB
    assert "_cond_reload" in _method("_build_conditions_tab")


@pytest.mark.parametrize("fn", ["_cond_split", "_cond_set_from_selection",
                                "_cond_autofill"])
def test_edits_that_need_one_row_say_so(fn):
    body = _method(fn)
    assert "showinfo" in body or "showwarning" in body


# ── populating ───────────────────────────────────────────────────────────────

def test_the_tab_reads_the_active_file_variable():
    """It read txt_file_path, which does not exist.

    The attribute holding the open recording is file_path. Reading a name that
    is never set is not an error -- getattr guards it -- so the tab simply
    reported that no file was loaded, for every file.
    """
    assert "txt_file_path" not in TAB
    assert "self.file_path.get()" in TAB


def test_the_tab_populates_when_it_is_shown():
    """The table is built from the file's events, which do not exist when the
    tab is constructed."""
    s2 = (PKG / "stage2.py").read_text(encoding="utf-8")
    assert "tab_conditions" in s2
    assert "_cond_tab_shown()" in s2


def test_an_assignment_survives_leaving_the_tab_and_returning():
    """Rebuilding on every visit would discard the work silently.

    Which is why Reload is a button: going away to check a waveform is not a
    request to start again.
    """
    body = _method("_cond_tab_shown")
    assert "_cond_source_path" in body
    assert "if self._cond_rows and" in body


def test_the_table_is_rebuilt_when_the_recording_changes():
    body = _method("_cond_tab_shown")
    assert "!= path" in body or "== path" in body
    assert "_cond_reload()" in body


# ── the order of the workflow ────────────────────────────────────────────────

def test_loading_a_file_lands_on_conditions():
    """What a stimulus type is FOR is decided before how it is detected.

    Twenty pulses labelled A may be two timepoints; configuring latency
    windows for "A" before saying so means configuring them again for each
    half afterwards.
    """
    body = _method("_build_labels_tab", src=APP)
    assert "self.nb_setup.select(self.tab_conditions)" in body


def test_confirming_goes_on_to_the_labels_tab():
    body = _method("_cond_apply")
    assert "self._cond_confirming = True" in body
    assert "_build_labels_tab(sorted(groups))" in body


def test_confirming_does_not_bounce_back_to_conditions():
    """Confirm rebuilds the labels tab, which is also what a fresh file does;
    without distinguishing the two callers that would be a loop."""
    body = _method("_build_labels_tab", src=APP)
    assert "_cond_confirming" in body
    i = body.index("_cond_confirming")
    assert "nb_stage1.select" in body[i:i + 300]


def test_the_flag_is_cleared_even_if_the_rebuild_fails():
    """Left set, every later file would skip the Conditions tab silently."""
    body = _method("_cond_apply")
    assert "finally:" in body
    assert "self._cond_confirming = False" in body


def test_the_button_says_what_it_does_next():
    assert "Confirm events & continue" in TAB


def test_navigation_falls_back_if_the_tab_is_missing():
    """A layout change should not strand a loaded file on no tab at all."""
    body = _method("_build_labels_tab", src=APP)
    assert "except Exception:" in body
    i = body.index("nb_setup.select(self.tab_conditions)")
    assert "nb_stage1.select" in body[i:]


# ── undo ─────────────────────────────────────────────────────────────────────

def test_every_edit_goes_through_the_history():
    """An edit that bypassed it would be the one that cannot be taken back,
    and the analyst would not know which until they tried."""
    for fn in ("_cond_split", "_cond_set_from_selection", "_cond_rename",
               "_cond_autofill", "_cond_toggle_exclude", "_cond_delete"):
        body = _method(fn)
        assert "_cond_commit(" in body, f"{fn} edits without recording it"
        assert "self._cond_rows =" not in body, \
            f"{fn} assigns the table directly, bypassing the history"


def test_a_refused_edit_leaves_nothing_on_the_stack():
    """An undo that does nothing visible teaches the analyst the button is
    unreliable, and the next press is the one that goes too far."""
    body = _method("_cond_split")
    i = body.index("except C.ConditionError")
    j = body.index("_cond_commit(")
    assert i < j, "the failure path must return before anything is recorded"
    assert "return" in body[i:j]


def test_undo_and_redo_are_symmetrical():
    u, r = _method("_cond_undo"), _method("_cond_redo")
    assert "_cond_redo_stack.append" in u
    assert "_cond_undo_stack.append" in r


def test_a_new_edit_discards_the_redo_stack():
    """Redoing onto a table that has since changed would apply an edit to rows
    it was never made against."""
    assert "_cond_redo_stack.clear()" in _method("_cond_commit")


def test_the_history_is_bounded():
    from mep_cmap.conditions_tab import UNDO_DEPTH
    assert 10 <= UNDO_DEPTH <= 200
    assert "del self._cond_undo_stack[:-UNDO_DEPTH]" in _method("_cond_commit")


def test_reloading_clears_the_history():
    """Undoing past a reload would restore rows belonging to a recording that
    is no longer open."""
    body = _method("_cond_reload")
    assert "_cond_undo_stack.clear()" in body
    assert "_cond_redo_stack.clear()" in body


def test_the_button_names_what_it_will_undo():
    """'Undo split' is a different proposition from 'Undo delete'."""
    body = _method("_cond_refresh_history_buttons")
    assert "_cond_undo_stack[-1][1]" in body
    # ast.unparse normalises quotes, so match the value not the literal
    assert "disabled" in body


def test_the_shortcuts_are_bound_to_the_widgets_not_the_window():
    """Ctrl+Z in a dialogue's entry field should still be the entry's own."""
    body = _method("_build_conditions_tab")
    assert "self._cond_tree, self._cond_list" in body
    assert "self.root.bind" not in body


def test_the_shortcuts_are_bound_after_both_widgets_exist():
    """Bound beside the table, the listbox did not yet exist."""
    i = TAB.index("for _w in (self._cond_tree, self._cond_list):")
    j = TAB.index("self._cond_list = tk.Listbox")
    assert j < i


def test_the_history_holds_frozen_rows():
    """A snapshot is the list itself rather than a copy that has to be kept
    honest, because ConditionRow is frozen and every helper returns a new list.
    """
    from mep_cmap.conditions import ConditionRow
    assert getattr(ConditionRow, "__dataclass_params__").frozen
