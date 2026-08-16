"""
Guards against the failure that recurred three times in one working session:
something existing in two places, one of them updated.

    * the Inspector payload chain -- an argument added at both ends and missed
      in the middle hop
    * the session writers -- a key added to one of two, so it was saved by
      whichever path happened to run
    * the channel assignment dialogues -- one converted to a tick list, the
      other left as a single choice, so multi-channel was reachable for one
      format out of ten

Each was found in testing, by the analyst, after a round trip. Each was
invisible to the compiler, to pyflakes and to every behavioural test, because
nothing raised: the code that ran was correct, and the code that did not run
was the problem.

These tests enumerate rather than sample. A test that checks the instance its
author knew about reproduces the fault it was written to prevent.
"""

import ast
import pathlib
import re

APP = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "app.py").read_text(encoding="utf-8")
TREE = ast.parse(APP)


def _fn(name):
    for n in ast.walk(TREE):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"{name} not found")


def _assigned_attrs(fn):
    out = set()
    for n in ast.walk(fn):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == "self" and isinstance(n.ctx, ast.Store)):
            out.add(n.attr)
    return out


def test_only_one_place_builds_a_session_payload():
    """Two builders drifted by thirteen settings before this was noticed."""
    a = APP.index("def _session_payload(self")
    b = APP.index("\n    def ", a + 10)
    outside = APP[:a] + APP[b:]
    assert not re.findall(r"^\s*session\s*=\s*\{", outside, re.M)


def test_per_file_state_is_cleared_in_exactly_one_place():
    """
    Clearing spread across the load paths means a new one inherits the previous
    file's state, and adding a new piece of state means finding every site.
    """
    for attr in ("segments_metadata", "_chan_segment_meta", "_chan_settings",
                 "_chan_confirmed"):
        assert attr in _assigned_attrs(_fn("_reset_state_for_new_file")), (
            f"self.{attr} is not cleared in _reset_state_for_new_file"
        )


def test_every_channel_dialogue_is_a_tick_list():
    """
    Multi-channel analysis must not be reachable from only one of them. There
    are two: one for Spike2 SMR, one for every other format with channels.
    """
    n_dialogues = APP.count('dlg.title("Channel Assignment")')
    assert n_dialogues >= 2, "the test assumes more than one dialogue"
    n_ticklists = len(re.findall(r"_chan_vars\s*=\s*\{\}", APP))
    assert n_ticklists >= n_dialogues, (
        f"{n_dialogues} channel dialogues but only {n_ticklists} tick lists"
    )


def test_the_inspector_payload_hops_agree():
    """Covered in detail in test_inspector_payload_chain; asserted here too."""
    def params(name):
        return [a.arg for a in _fn(name).args.args if a.arg != "self"]

    assert params("_show_inspector_cb") == params("_open_inspector_gui")


def test_every_per_stimulus_map_is_stored_per_channel():
    """
    Storing some of tab 1a's maps but not others let the latency numbers drift
    from the dropdowns that produce them.
    """
    a = APP.index("def _harvest_labels_tab")
    b = APP.index("def _set_confirm_state")
    harvested = set(re.findall(r"self\.(\w+)\s*=", APP[a:b]))
    a2 = APP.index("_chan_settings_keys = (")
    stored = set(re.findall(r'"(\w+)"', APP[a2:APP.index(")", a2)]))
    assert not (harvested - stored)
