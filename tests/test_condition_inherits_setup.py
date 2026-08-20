"""
A condition inherits its stimulus type's setup.

Splitting a stimulus type rebuilds the setup table with composite keys -- A
becomes A-pre and A-post -- and every per-type map is keyed by the row. A
muscle group, latency profile, gap or silent-period tick set against A
therefore vanished the moment conditions were applied, and the new rows took
defaults.

A condition is a property of the trial, not of the response: A-pre and A-post
are the same stimulus recorded at different times and want the same detection
settings. Inheriting them is a starting point, not a constraint -- either row
can be edited, and its own entry then wins.
"""

import ast
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
APP = (PKG / "app.py").read_text(encoding="utf-8")
PIPE = (PKG / "pipeline.py").read_text(encoding="utf-8")


def _body(name, src=APP):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found")


def _inherited(mapping, stim, default=None):
    """The shipped lookup, re-expressed.

    app.py needs a working matplotlib Tk backend and cannot be imported by the
    suite. The behaviour is asserted here and the SOURCE is checked below, so a
    change to one without the other fails.
    """
    from mep_cmap.conditions import decompose
    if stim in (mapping or {}):
        return mapping[stim]
    base, cond = decompose(stim)
    if cond and base in (mapping or {}):
        return mapping[base]
    return default


def test_the_shipped_method_matches_what_is_asserted_here():
    """The re-expression above must not drift from the code it stands in for."""
    body = _body("_inherited")
    assert "if stim in (mapping or {}):" in body
    assert "if cond and base in (mapping or {}):" in body
    assert "return default" in body


def test_a_row_with_its_own_entry_uses_it():
    m = {"A\u00b7pre": (4.0, 18.0), "A": (13.0, 30.0)}
    assert _inherited(m, "A\u00b7pre") == (4.0, 18.0)


def test_a_condition_falls_back_to_its_stimulus_type():
    assert _inherited({"A": (3.0, 12.0)}, "A\u00b7post") == (3.0, 12.0)


def test_a_plain_stimulus_type_does_not_inherit_from_anywhere():
    assert _inherited({"A": (3.0, 12.0)}, "B", "fallback") == "fallback"


def test_an_unknown_condition_returns_the_default():
    assert _inherited({}, "A\u00b7pre", "d") == "d"


def test_an_empty_map_is_tolerated():
    assert _inherited(None, "A\u00b7pre", "d") == "d"


@pytest.mark.parametrize("mapping", [
    "latency_map", "latency_stim_map", "latency_muscle_map",
    "gap_ms_map", "delay_ms_map",
])
def test_every_stimulus_property_inherits(mapping):
    """Each of these describes the stimulus or the response, not the trial."""
    body = _body("_build_labels_tab")
    assert f"self.{mapping}, stim" in body, f"{mapping} does not inherit"


def test_the_silent_period_tick_inherits():
    """Whether a silent period applies is a property of the stimulus."""
    body = _body("_build_labels_tab")
    assert "_b in self.csp_types" in body


def test_the_epoch_window_does_not_inherit():
    """It is set per condition deliberately -- that is the point of the
    Conditions tab -- so a blank box means the file-wide default, not the
    stimulus type's window."""
    body = _body("_build_labels_tab")
    assert "self._inherited(self.window_map" not in body


# ── the missing profile is reported ──────────────────────────────────────────

def test_a_missing_latency_profile_is_reported_not_invented():
    """The fallback was a hardcoded 10-50 ms, and every detector bounds its
    result by the minimum -- so a type absent from the map returned exactly
    10.00 ms on every trial, with a between-trial SD of zero."""
    assert "cfg.latency_map.get(stim_type, (10.0, 50.0))" not in PIPE
    assert "has no latency profile" in PIPE


def test_the_report_says_what_to_do():
    i = PIPE.index("has no latency profile")
    window = PIPE[i:i + 500]
    assert "tab 1a" in window
    assert "pinned" in window


def test_the_substitute_is_the_amplitude_window_not_a_constant():
    """If a bound must be invented, it should at least be one the analyst
    chose rather than a number from the source."""
    i = PIPE.index("has no latency profile")
    assert "cfg.ptp_start" in PIPE[i - 400:i]


# ── every path that resolves a latency profile ───────────────────────────────

def test_no_hardcoded_ten_to_fifty_survives_anywhere():
    """There were THREE copies of the same fallback: the detection path, the
    agreement path, and the inspector. Each turned a missing profile into an
    onset of about 10 ms that read as a real latency, and the inspector's copy
    is why the preview and the analysis disagreed with each other.
    """
    import pathlib
    pkg = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
    offenders = []
    for path in (pkg / "pipeline.py", pkg / "inspector.py"):
        src = path.read_text(encoding="utf-8")
        if "(10.0, 50.0)" in src:
            offenders.append(path.name)
    assert not offenders, (
        "a hardcoded latency fallback remains in: " + ", ".join(offenders))


def test_the_inspector_reports_a_missing_profile():
    """It happens most easily on a channel that was never set up: the maps are
    per channel, so previewing EMG 2 with EMG 1 configured finds nothing."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent / "mep_cmap" /
           "inspector.py").read_text(encoding="utf-8")
    assert "has no latency profile on" in src
    assert "tab 1a" in src


def test_the_inspector_warns_once_per_type():
    """A message rewritten on every redraw is noise, not a warning."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent / "mep_cmap" /
           "inspector.py").read_text(encoding="utf-8")
    assert "_warned_no_latency" in src


def test_the_inspector_falls_back_to_the_chosen_window():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent / "mep_cmap" /
           "inspector.py").read_text(encoding="utf-8")
    i = src.index("has no latency profile on")
    assert "self.ptp_start_ms" in src[i - 500:i]
