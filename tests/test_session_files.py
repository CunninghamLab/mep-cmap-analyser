"""
One session per recording, in derivatives.

Save Session opened a dialogue defaulting beside the raw data, while the
automatic save wrote a BIDS-named file under derivatives. A recording could
therefore carry two sessions that knew nothing of each other, and which one
took effect depended on what the analyst happened to pick on the way back in.
"""

import ast
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
    body = _body("session_path")
    assert "derivatives" in body
    assert "_session.json" in body


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


def test_the_session_goes_to_derivatives_not_beside_the_raw_data():
    """Raw data is what the scanner and the stimulator wrote; a session is
    something this tool produced."""
    body = _body("session_path")
    assert "derivatives_path" in body
    assert "os.path.dirname(fp)" in body, "falls back only when none is set"


def test_derivatives_is_not_nested_twice():
    body = _body("session_path")
    assert "== 'derivatives'" in body or '== "derivatives"' in body


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
