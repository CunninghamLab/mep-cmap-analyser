"""
A delay measured on a stim code must reach the rows that code became.

The scan measures against the configured events, keyed by the recording's own
stim code. The setup table is keyed by its rows, and applying conditions splits
one code into several: 'A' becomes 'A·first' and 'A·last'. The lookup asked for
a row called 'A', found none, and dropped a delay it had just measured -- while
the summary line reported that no delay had been proposed.

The symptom was a log that contradicted itself:

    A: artefact at +17.5 ms (SD 0.95, n=21) - proposing delay +17.5 ms
    No delays proposed; the markers line up with the stimulus artefact...
"""

import ast
import pathlib

import pytest

from mep_cmap.conditions import SEPARATOR, compose, decompose

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = (ROOT / "mep_cmap" / "app.py").read_text(encoding="utf-8")


def _method(name, src=APP):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found")


def _scan_method():
    """The delay-scan handler, found by what it does rather than its name."""
    for node in ast.walk(ast.parse(APP)):
        if isinstance(node, ast.FunctionDef):
            body = ast.unparse(node)
            if "scan_event_delays(" in body and "_lab_entry_delay" in body:
                return body
    raise AssertionError("delay scan handler not found")


# ── the mapping ──────────────────────────────────────────────────────────────

def test_a_split_row_still_maps_back_to_its_code():
    """The property the fix relies on."""
    assert decompose(compose("A", "first"))[0] == "A"
    assert decompose("A")[0] == "A"


def test_the_scan_resolves_rows_by_code_not_by_exact_key():
    """Looking the row up by the scanned key alone is what failed: after
    conditions there is no row called 'A'."""
    body = _scan_method()
    assert "decompose" in body, "rows must be resolved back to their stim code"
    assert "_lab_entry_delay.get(stim)" not in body, (
        "the exact-key lookup is the bug: no row is called 'A' once conditions "
        "are applied")


def test_every_row_sharing_a_code_is_filled_in():
    """A delay is a property of the trigger path, not of the condition, so all
    of a code's rows take it."""
    body = _scan_method()
    assert "_rows_for_code" in body
    # Applied inside a loop over that code's rows, not to a single entry.
    assert body.count("var.set(") == 1
    i_rows = body.index("_rows_for_code.get(")
    assert body.index("var.set(") > i_rows


def test_the_summary_counts_what_was_actually_applied():
    """'applied' drives both the message and the status line, so a delay that
    is measured but not placed must not be reported as placed -- and one that
    IS placed must not be reported as absent."""
    body = _scan_method()
    assert "applied += 1" in body
    assert "if applied:" in body


def test_the_source_is_recorded_against_the_row(): 
    """delay_source_map is read per row elsewhere; keying it by the scanned
    code would leave the rows looking hand-entered."""
    body = _scan_method()
    assert "self.delay_source_map[_key]" in body


# ── the separator is what makes this work ────────────────────────────────────

def test_a_condition_name_cannot_contain_the_separator():
    """decompose() splits on the first separator, so a condition containing one
    would resolve to the wrong code — and the delay would land on a row that
    does not exist. Refused outright rather than silently rewritten, so the
    analyst names it rather than discovering later what it was renamed to."""
    from mep_cmap.conditions import ConditionError, sanitise_name
    with pytest.raises(ConditionError):
        sanitise_name(f"pre{SEPARATOR}post")
