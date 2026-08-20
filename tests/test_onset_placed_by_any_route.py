"""
Every route that places an onset must count as placing one.

A trial whose onset the detector could not find carries `onset_auto_failed`, so
the status says "not detected" rather than reporting the fallback index as a
measured 0.0 ms latency. The flag is cleared when the analyst places a marker,
because a placed marker is a measurement whatever the detector managed.

_update_meta cleared it. Dragging the AUC START with "Link AUC to onset &
offset" ticked did not: it wrote onset_idx directly and moved the onset dot to
match, so the marker moved, the latency became computable, and the status still
read "not detected".
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
INSPECTOR = (ROOT / "mep_cmap" / "inspector.py").read_text(encoding="utf-8")


def _function(name, src=INSPECTOR):
    """A function by name, including ones nested inside methods."""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found")


def _onset_writers():
    """Every place that assigns onset_idx, by enclosing function."""
    tree = ast.parse(INSPECTOR)
    out = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        body = ast.unparse(fn)
        for line in body.splitlines():
            s = line.strip()
            if s.startswith("m['onset_idx'] =") or s.startswith('m["onset_idx"] ='):
                out.append(fn.name)
                break
    return out


# ── the rule ─────────────────────────────────────────────────────────────────

def test_the_onset_marker_clears_the_flag():
    body = _function("_update_meta")
    assert "onset_auto_failed" in body
    assert "_refresh_status()" in body, "a cleared flag with a stale label " \
                                        "looks exactly like a flag that stayed"


def test_dragging_the_auc_start_clears_it_too():
    """The bug: this writes onset_idx directly rather than through
    _update_meta, so the rule attached to that function was skipped."""
    body = _function("_on_start_moved")
    # Quote style is whatever ast.unparse chooses.
    assert "onset_idx'] = new_idx" in body or 'onset_idx"] = new_idx' in body
    assert "onset_auto_failed" in body, (
        "moving the onset by this route must count as placing it")


def test_the_auc_route_refreshes_the_status():
    assert "_refresh_status()" in _function("_on_start_moved")


# ── no third route ───────────────────────────────────────────────────────────

def test_every_writer_of_onset_idx_clears_the_flag():
    """The guard against a third one appearing. A route that sets an onset
    without clearing the flag reports a latency it has as absent."""
    offenders = []
    for name in _onset_writers():
        body = _function(name)
        if "onset_auto_failed" not in body:
            offenders.append(name)
    assert offenders == [], (
        f"these place an onset without clearing onset_auto_failed: {offenders}")


# ── what the flag is for ─────────────────────────────────────────────────────

def test_a_non_detection_is_not_reported_as_zero():
    """The fallback index is the stimulus, which reads as 0.0 ms -- and the
    analysis honours a stored onset_idx as a manual override, so exporting it
    turned a blank latency into a measured one."""
    body = _function("_refresh_status")
    assert "not detected" in body
    assert "onset_auto_failed" in body


def test_the_fallback_is_not_exported():
    body = _function("_on_close") if "_on_close" in INSPECTOR else INSPECTOR
    assert "onset_auto_failed" in body
    assert "'onset_idx'" in body or '"onset_idx"' in body
