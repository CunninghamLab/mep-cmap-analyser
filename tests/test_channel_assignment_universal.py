"""
Channel assignment reaches every format.

It used to run for Spike2 text exports alone, on the reasoning that other
formats "use the Format Wizard or have no event marker concept". A LabChart
text export of six named channels -- handgrip, MEP, stim, two EMG -- was
therefore analysed on whichever channel came first, with no route to Event
sources at all.

The premise was wrong twice over. Channels need choosing wherever there is
more than one, whatever wrote the file; and a recording whose embedded markers
are wrong needs a threshold source configured against a trigger channel, which
is only reachable through this dialogue. "There is nothing to choose" is a
statement about the file, not about what the analyst may need to do with it.
"""

import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
APP = (PKG / "app.py").read_text(encoding="utf-8")


def _method(name, src=APP):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant):
                body = body[1:]
            return "\n".join(ast.unparse(n) for n in body)
    raise AssertionError(f"{name} not found")


def test_the_dialogue_is_not_restricted_by_format():
    """The old gate named four formats to exclude; a fifth would have been
    excluded by being forgotten rather than by being considered."""
    assert "_needs_assign_dlg = (_fmt != 'spike2_smr')" in APP
    assert "'generic_tsv', 'labchart', 'cfwb'" not in \
        APP[APP.index("_needs_assign_dlg"):APP.index("_needs_assign_dlg") + 200]


def test_it_is_not_skipped_when_the_file_offers_no_choice():
    """A single embedded marker is exactly the case where a threshold source
    may be needed, because the embedded one may be the thing that is wrong."""
    i = APP.index("_needs_assign_dlg = ")
    line = APP[i:APP.index("\n", i)]
    assert "len(chan_list)" not in line
    assert "len(markers)" not in line


def test_spike2_smr_keeps_its_own_dialogue():
    """It does the same job with the extra marker-channel step that format
    requires; two dialogues for one file would be worse than one."""
    assert "_fmt != 'spike2_smr'" in APP


def test_the_channel_list_is_read_for_every_format():
    """Opening the gate would only move the failure if a branch never set it."""
    body = _method("_browse_file_path")
    assert "chan_list = list_waveform_channels(fpath)" in body


def test_the_assignment_can_be_reopened():
    """A remembered answer that can only be changed by opening the file again
    is a remembered answer the analyst has to work around."""
    assert "_reopen_assignment = lambda" in APP
    body = _method("reopen_channel_assignment")
    assert "fn()" in body


def test_reopening_rebuilds_the_labels_tab():
    """The channel and marker it sets are what that table is built from."""
    assert "_build_labels_tab" in _method("reopen_channel_assignment")


def test_the_stored_dialogue_is_dropped_when_a_file_is_opened():
    """Reopening the assignment for a recording that is no longer loaded would
    set the channel from one file's names against another's data."""
    body = _method("_browse_file_path")
    i = body.index("self._reopen_assignment = None")
    j = body.index("chan_list = list_waveform_channels(fpath)")
    assert i < j


def test_reopening_before_a_file_is_loaded_explains_itself():
    body = _method("reopen_channel_assignment")
    assert "showinfo" in body
    assert "Open a recording first" in body


def test_there_is_a_way_in_from_the_setup_table():
    """Beside Event sources, which is the other half of the same decision."""
    assert "Channel assignment\u2026" in APP
    assert "command=self.reopen_channel_assignment" in APP


# ── the menu route ───────────────────────────────────────────────────────────

def test_the_file_menu_reaches_every_format():
    """Two routes to the same decision -- the menu where you look for it, the
    button where you are working -- both landing in the same dialogue."""
    body = _method("_reassign_channels")
    assert "self.reopen_channel_assignment()" in body


def test_the_sidecar_path_is_taken_first_where_one_exists():
    """Reopening the dialogue without discarding a saved answer would leave it
    to reappear on the next load."""
    body = _method("_reassign_channels")
    i = body.index("reopen_channel_assignment()")
    j = body.index("os.remove(str(side))")
    assert i < j, "the no-sidecar case returns before the sidecar case runs"
