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
