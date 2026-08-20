"""
One session per recording, in derivatives.

Save Session opened a dialogue defaulting beside the raw data, while the
automatic save wrote a BIDS-named file under derivatives. A recording could
therefore carry two sessions that knew nothing of each other, and which one
took effect depended on what the analyst happened to pick on the way back in.
"""

import ast
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = (ROOT / "mep_cmap" / "app.py").read_text(encoding="utf-8")
S2 = (ROOT / "mep_cmap" / "stage2.py").read_text(encoding="utf-8")


def _body(name, src=APP):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found")


def test_there_is_one_rule_for_where_a_session_lives():
    """Asserted by calling it, not by reading it.

    The rule moved out of the method into session_path_for so that code working
    over a LIST of recordings could use it too. Source-text assertions on the
    method broke on that move while the behaviour was unchanged, which is a
    test reporting on where code sits rather than what it does.
    """
    from mep_cmap.bids import session_path_for
    p = session_path_for(os.path.join("study", "raw", "rec.smr"),
                         None, os.path.join("study", "deriv"))
    assert p.endswith("_session.json")
    assert "derivatives" in p


def test_the_session_goes_to_derivatives_not_beside_the_raw_data():
    """Raw data is what the amplifier and the stimulator wrote; a session is
    something this tool produced."""
    from mep_cmap.bids import session_path_for
    raw = os.path.join("study", "raw")
    p = session_path_for(os.path.join(raw, "rec.smr"), None,
                         os.path.join("study", "deriv"))
    assert os.path.dirname(p) != raw
    # Falls back to the recording's own folder only when none is configured.
    q = session_path_for(os.path.join(raw, "rec.smr"), None, "")
    assert q.startswith(os.path.join(raw, "derivatives"))


def test_derivatives_is_not_nested_twice():
    from mep_cmap.bids import session_path_for
    p = session_path_for(os.path.join("study", "rec.smr"), None,
                         os.path.join("study", "derivatives"))
    assert "derivatives" + os.sep + "derivatives" not in p


def test_every_recording_gets_its_own_session():
    """Two recordings in one session folder must not share a file, or the
    second silently overwrites the first."""
    from mep_cmap.bids import session_path_for
    a = session_path_for(os.path.join("s", "sub-004_run-1.smr"), None, "d")
    b = session_path_for(os.path.join("s", "sub-004_run-2.smr"), None, "d")
    assert a != b


def test_the_method_delegates_rather_than_repeating_the_rule():
    """Two builders drift, and the one that drifts deletes or fails to find
    files silently -- which is exactly what the reset was doing."""
    assert "session_path_for(" in _body("session_path")


def test_both_writers_use_it():
    """Two rules for one filename is how the two files came about."""
    assert "self.session_path()" in _body("save_session")
    assert "self.session_path()" in _body("_autosave_session")


def test_saving_does_not_ask_where():
    """A recording has one session, and Save Session writes it."""
    body = _body("save_session")
    assert "asksaveasfilename" not in body


def test_a_named_copy_is_still_possible():
    """A variant set aside before changing something is a different intention
    from recording where the work has got to."""
    body = _body("save_session_copy")
    assert "asksaveasfilename" in body
    assert "copyfile" in body


def test_the_copy_command_is_reachable():
    assert "save_session_copy()" in APP


def test_saving_without_a_recording_says_so():
    assert "Open a recording first" in _body("save_session")


def test_the_session_goes_to_derivatives_not_beside_the_raw_data_source():
    """Kept as a source check only for the reset path, which must use the
    shared rule rather than the pre-derivatives filename it used to delete."""
    body = _body("_dataset_reset_selected") if "_dataset_reset_selected" in APP else APP
    assert "session_path_for(" in body


def test_derivatives_is_not_nested_twice_in_the_rule():
    from mep_cmap.bids import session_path_for
    p = session_path_for("rec.smr", None, os.path.join("a", "DERIVATIVES"))
    assert p.lower().count("derivatives") == 1


# ── reachable from anywhere in First Level ───────────────────────────────────

def test_the_footer_belongs_to_first_level_not_one_tab():
    """Preparing a recording without running it is a workflow this tool
    supports, and it required navigating to 1c to record the work whichever
    tab that work had been done on."""
    assert 'self.footer_frame = tk.Frame(self.stage1_outer' in APP
    assert 'self.footer_frame = tk.Frame(self.tab_detect' not in APP


def test_the_footer_is_packed_before_the_notebook():
    """So Tk gives it its height first and the scrolling bodies take what is
    left, rather than the footer being squeezed out."""
    i = APP.index("self.footer_frame.pack(side='bottom'"
                  if "self.footer_frame.pack(side='bottom'" in APP
                  else 'self.footer_frame.pack(side="bottom"')
    j = APP.index("self.nb_stage1 = ttk.Notebook")
    assert i < j


def test_moving_between_tabs_saves():
    """A recording set up and then left for the next file used to keep its
    labels, conditions and windows nowhere but the session not yet written."""
    assert "_autosave_session()" in S2
    assert "_session_dirty" in S2


def test_the_autosave_is_guarded():
    """A save that fails must not stop the analyst changing tab."""
    i = S2.index("_autosave_session()")
    assert "except Exception" in S2[i:i + 300]


# ── Run is gated on having seen the detection settings ───────────────────────

def test_run_starts_disabled():
    """The footer used to belong to 1c, so reaching the button meant having
    passed the detection settings. Moving it to the whole of First Level made
    Run clickable from the labels tab."""
    i = APP.index('text="\u25b6  Run Analysis"')
    assert 'state="disabled"' in APP[i:i + 200]


def test_run_is_enabled_once_the_detection_tab_is_seen():
    body = _body("_refresh_run_button")
    assert "_seen_detection_tab" in body
    # ast.unparse normalises quotes; match the values, not the literals
    assert "normal" in body and "disabled" in body
    assert "_seen_detection_tab = True" in S2


def test_the_gate_resets_for_each_recording():
    """Left set from the previous file it would apply to the first recording
    of a session and to no other -- worse than not having it, because it would
    look like it was working."""
    body = _body("_browse_file_path")
    assert "self._seen_detection_tab = False" in body


def test_it_stays_enabled_once_seen():
    """Re-disabling on a trip back to the filter tab would be pedantry rather
    than protection."""
    s2i = S2.index("_seen_detection_tab = True")
    assert "= False" not in S2[s2i:s2i + 200]


def test_preview_is_not_gated():
    """Trying the settings is how one finds out whether they need looking at,
    and it writes nothing."""
    i = APP.index('text="\U0001f50e Preview detection"')
    assert 'state="disabled"' not in APP[i:i + 200]


def test_the_disabled_button_says_why():
    """A greyed control with no explanation reads as a broken one."""
    i = APP.index("self._run_btn = tk.Button")
    assert "Tooltip(self._run_btn" in APP[i:i + 900]
    assert "1c" in APP[i:i + 900]


# ── conditions belong to the session ─────────────────────────────────────────

def test_the_session_carries_the_conditions():
    """None of this was saved, so reopening a session lost every condition
    assigned in it -- and silently, because the analysis still ran on the
    stimulus types underneath.
    """
    body = _body("_build_session_payload") if "_build_session_payload" in APP \
        else APP
    for key in ("condition_event_rows", "condition_map", "condition_rows",
                "condition_epochs"):
        assert f'"{key}"' in APP, f"{key} is not saved"


def test_the_rows_are_saved_field_by_field():
    """ConditionRow is frozen and not JSON-serialisable; storing it whole would
    fail at save time rather than at load."""
    i = APP.index('"condition_rows"')
    block = APP[i:i + 500]
    for field in ("stim_type", "condition", "trials", "excluded",
                  "pre_ms", "post_ms"):
        assert field in block, f"{field} is dropped on save"


def test_loading_reconstructs_the_rows():
    assert "ConditionRow(stim_type=r.get" in APP


def test_a_session_written_before_conditions_still_loads():
    """It loads as a recording with none assigned, which is the state every
    session had until now."""
    assert 'sess.get("condition_rows") or []' in APP
    assert 'sess.get("condition_event_rows") or []' in APP


def test_the_epoch_book_survives_with_integer_channels():
    """JSON keys are strings; a channel index that came back as '0' would
    never match the integer the tab looks up."""
    assert "int(_c): {k: tuple(v)" in APP


def test_the_restored_table_is_not_rebuilt_over():
    """Rebuilding from the file the moment the tab opened is what made the
    conditions look as though they had never been saved."""
    tab = (PKG / "conditions_tab.py").read_text(encoding="utf-8") \
        if 'PKG' in dir() else \
        (ROOT / "mep_cmap" / "conditions_tab.py").read_text(encoding="utf-8")
    assert "Conditions restored from the session" in tab
    assert "_cond_source_path" in APP


# ── loading a session must put it on screen ──────────────────────────────────

def test_loading_reopens_the_recording():
    """Everything restored into memory and nothing was redrawn: the setup
    table still showed whatever was there before, the channel dropdown was not
    repopulated, and the Conditions tab held rows with no recording behind
    them -- which reads as "none of my settings saved" when in fact none of
    them had been displayed.
    """
    body = _body("load_session")
    assert "_browse_file_path(fp)" in body


def test_the_restored_state_survives_reopening_the_file():
    """Opening a recording resets exactly the maps that were just loaded."""
    body = _body("load_session")
    assert "_keep" in body
    for attr in ("latency_map", "window_map", "_chan_settings", "_cond_rows"):
        assert attr in body, f"{attr} is not preserved across the reopen"


def test_the_setup_table_is_rebuilt():
    body = _body("load_session")
    assert "_build_labels_tab" in body


def test_the_channels_own_settings_are_reapplied():
    """The flat maps are a view of one channel's snapshot; without this the
    table shows whichever channel was last active."""
    body = _body("load_session")
    assert "_restore_chan_settings(self.channel_idx)" in body


def test_a_missing_file_stops_before_reopening():
    """Warning and then trying to open it anyway would raise on top of the
    warning."""
    body = _body("load_session")
    i = body.index("File not found")
    j = body.index("_browse_file_path(fp)")
    assert "return" in body[i:j]


def test_a_session_with_no_file_does_not_try_to_open_one():
    body = _body("load_session")
    assert "if not fp:" in body


def test_reopening_failure_is_reported_not_raised():
    body = _body("load_session")
    assert "Could not reopen" in body
