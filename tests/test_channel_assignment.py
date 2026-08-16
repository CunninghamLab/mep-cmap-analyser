"""
Choosing the channels to analyse when a Spike2 file is first opened.

The dialogue asked for one EMG channel. Analysing several meant finding a
button beside the channel dropdown that most people would never look for. The
dialogue is where the recording's channels are already being declared, so it is
the natural place to say which of them matter.

There is deliberately no "primary" channel. Every ticked channel is analysed
identically with its own setup on tab 1a, so a primary would imply a hierarchy
that does not exist. The first ticked is simply where configuration starts.
"""

import json
import pathlib
import tempfile

from mep_cmap.formats.spike2_smr import analysis_channels_from_config

APP = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "app.py").read_text(encoding="utf-8")
SMR = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "formats" / "spike2_smr.py").read_text(encoding="utf-8")

HAVE = ["EMG 1", "EMG 2", "EMG 3", "Torque"]


# ── The saved assignment ─────────────────────────────────────────────────────

def test_a_new_sidecar_lists_every_selected_channel():
    names, dropped = analysis_channels_from_config(
        {"emg_channel": "EMG 1", "analysis_channels": ["EMG 1", "EMG 2"]}, HAVE)
    assert names == ["EMG 1", "EMG 2"]
    assert not dropped


def test_a_sidecar_written_before_this_loads_as_one_channel():
    """No migration, no prompt: it names one channel and that is what runs."""
    names, dropped = analysis_channels_from_config({"emg_channel": "EMG 2"}, HAVE)
    assert names == ["EMG 2"]
    assert not dropped


def test_a_channel_that_is_no_longer_in_the_file_is_dropped_not_substituted():
    """
    The recording has been re-exported or renamed. Silently analysing a
    different channel is worse than analysing fewer, and the caller is told so
    it can say which went missing.
    """
    names, dropped = analysis_channels_from_config(
        {"emg_channel": "EMG 1", "analysis_channels": ["EMG 1", "EMG 9"]}, HAVE)
    assert names == ["EMG 1"]
    assert dropped == ["EMG 9"]


def test_an_older_build_can_still_read_a_new_sidecar():
    """
    emg_channel stays the first channel to be analysed, so a build without
    multi-channel support reads the file unchanged rather than failing on an
    unfamiliar key.
    """
    from mep_cmap.formats.spike2_smr import save_config

    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "rec.smr"
        f.write_bytes(b"x")
        save_config(str(f), "EMG 1", "DigMark",
                    analysis_channels=["EMG 1", "EMG 2"])
        side = next(pathlib.Path(d).glob("*.json"))
        cfg = json.loads(side.read_text())
        assert cfg["emg_channel"] == "EMG 1"
        assert cfg["stim_channel"] == "DigMark"
        assert cfg["analysis_channels"] == ["EMG 1", "EMG 2"]


def test_channels_are_stored_by_name_not_index():
    """An index means nothing if the file is re-exported in a different order."""
    a = SMR.index("def save_config")
    b = SMR.index("def analysis_channels_from_config")
    body = SMR[a:b]
    assert "str(c)" in body, "names, not positions"


# ── The dialogue ─────────────────────────────────────────────────────────────

def test_the_dialogue_offers_a_tick_list():
    assert "EMG channels:" in APP
    assert "_chan_vars[_cn] = _v" in APP


def test_there_is_no_primary_selector():
    """
    Every ticked channel is analysed identically, so a primary would imply a
    hierarchy that does not exist -- and invite the question of what to do when
    all channels matter equally.
    """
    a = APP.index("EMG channels:")
    seg = APP[a - 900:a + 1500]
    assert "Radiobutton" not in seg
    assert "primary" in seg.lower(), "the reasoning should be recorded"


def test_saving_with_nothing_ticked_is_refused():
    a = APP.index("picked = [c for c in _analogue")
    # Wide enough to reach the return; the block is deeply indented, so the
    # character count runs well ahead of the line count.
    seg = APP[a:a + 700]
    assert "if not picked:" in seg
    assert "showwarning" in seg
    assert "return" in seg


def test_the_first_ticked_channel_is_where_configuration_starts():
    a = APP.index('_chosen["emg"]      = picked[0]')
    assert '_chosen["channels"] = picked' in APP[a:a + 200]


def test_a_long_channel_list_scrolls_rather_than_growing_the_window():
    a = APP.index("if len(_analogue) > 10:")
    assert "Scrollbar" in APP[a:a + 600]


# ── Reaching the analysis selection ──────────────────────────────────────────

def test_the_selection_is_resolved_from_names_after_the_dropdown_exists():
    """An index is defined by the dropdown's list, which does not exist earlier."""
    a = APP.index('_pending = getattr(self, "_pending_analysis_channel_names", None)')
    seg = APP[a:a + 700]
    assert "enumerate(chan_list)" in seg
    assert "self.analyse_channels = _idx" in seg


def test_the_analyse_button_still_overrides_the_dialogue():
    """The dialogue sets the initial answer, not the only answer."""
    assert "_choose_analysis_channels" in APP
    a = APP.index("self._pending_analysis_channel_names = list(_picked)")
    assert "does not become the only way" in APP[a - 400:a]


# ── Reopening the dialogue ───────────────────────────────────────────────────

def test_there_is_a_way_to_reassign_channels():
    """
    The dialogue appears only when no sidecar exists, so a file set up before
    multi-channel support would keep its single channel forever. The only other
    route is deleting a JSON by hand.
    """
    assert "Reassign channels" in APP
    assert "def _reassign_channels(self)" in APP


def test_reassigning_confirms_before_discarding():
    a = APP.index("def _reassign_channels")
    b = APP.index("\n    def ", a + 10)
    body = APP[a:b]
    assert "askyesno" in body
    assert "os.remove" in body
    assert body.index("askyesno") < body.index("os.remove")


def test_reassigning_explains_when_the_format_has_no_assignment():
    a = APP.index("def _reassign_channels")
    b = APP.index("\n    def ", a + 10)
    assert "Nothing to reassign" in APP[a:b]


# ── The dialogue appears every time ──────────────────────────────────────────

def test_the_dialogue_is_not_skipped_once_a_sidecar_exists():
    """
    It used to appear only when no sidecar existed, so after the first open the
    channel and marker choices became invisible and unchangeable -- the only
    way back was deleting the derivatives folder and the sidecars by hand.
    Those two decisions determine what the whole analysis measures.
    """
    a = APP.index("# Always ask, pre-filled with whatever was saved.")
    b = APP.index("info = _smr_info(fpath)", a)
    assert "if not _smr_has_cfg(fpath):" not in APP[a:b], (
        "the dialogue is still gated on the sidecar being absent"
    )


def test_saved_channels_are_preselected():
    """Remembering still does its job; confirming is one click."""
    a = APP.index("_prev_sel = []")
    b = APP.index("_chan_vars = {}", a)
    body = APP[a:b]
    assert "analysis_channels_from_config" in body
    assert "_smr_has_cfg(fpath)" in body


def test_the_saved_trigger_source_is_preselected():
    """
    Now that the dialogue opens every time, defaulting to DigMark would quietly
    undo a deliberate choice of something else on every reopen.
    """
    a = APP.index("_saved_stim = \"\"")
    b = APP.index("stim_var = tk.StringVar(", a)
    body = APP[a:b]
    assert 'get("stim_channel"' in body
    assert "DigMark" in body, "DigMark should remain the fallback"
    assert body.index("_saved_stim") < body.index("DigMark"), (
        "the saved value must be tried before the default"
    )


def test_the_wording_matches_the_behaviour():
    assert "will not be asked again" not in APP, (
        "the dialogue now appears every time; the note must say so"
    )
    assert "remembered and shown again next time" in APP


# ── Both dialogues, not just the Spike2 one ──────────────────────────────────

def _dialog_bodies():
    """Source of each channel-assignment dialogue, by AST rather than offsets.

    Both are nested functions of several hundred lines; slicing on a fixed
    character window or a guessed end marker finds the wrong region and the
    assertions then examine unrelated code.
    """
    import ast

    lines = APP.splitlines(keepends=True)
    starts = [0]
    for ln in lines[:-1]:
        starts.append(starts[-1] + len(ln))

    out = []
    tree = ast.parse(APP)
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name in ("_show_smr_dialog", "_show_assign_dlg")):
            a = starts[node.lineno - 1]
            b = starts[node.end_lineno - 1] + len(lines[node.end_lineno - 1])
            out.append((node.name, APP[a:b]))
    assert len(out) == 2, f"expected two dialogues, found {[n for n, _ in out]}"
    return out


def test_every_channel_dialogue_offers_a_tick_list():
    """
    Multi-channel analysis is not a Spike2 feature, so the way into it must not
    be either. The generic dialogue serves LabChart MATLAB, BrainVision, EDF,
    AcqKnowledge, epoched MATLAB and Brainsight; leaving it as a single choice
    made the capability reachable for one format out of ten.
    """
    for name, body in _dialog_bodies():
        assert "Checkbutton" in body, f"{name} still offers a single choice"
        assert "_chan_vars" in body, f"{name} does not collect a set"


def test_every_channel_dialogue_refuses_an_empty_selection():
    for name, body in _dialog_bodies():
        assert "showwarning" in body, f"{name} accepts no channels"


def test_every_channel_dialogue_preselects_the_previous_choice():
    """Otherwise reopening a file quietly resets it to the first channel."""
    for name, body in _dialog_bodies():
        assert ("_prev_sel" in body) or ("_prev" in body), \
            f"{name} does not restore the previous selection"


def test_the_generic_dialogue_populates_the_analysis_selection():
    """
    Anchored AFTER the generic dialogue: the same line appears in the Spike2
    path, and searching from the start of the file finds that one instead.
    """
    a = APP.index("def _show_assign_dlg")
    seg = APP[a:]
    b = seg.index('_picked = _chosen.get("channels") or [_chosen["emg"]]')
    block = seg[b:b + 600]
    assert "self.analyse_channels = {" in block
    assert "_refresh_analyse_button" in block
