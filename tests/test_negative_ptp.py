"""
A negative peak-to-peak amplitude cannot exist, so it is reported as a fault.

ptp_max below ptp_min means the two landmarks are not the maximum and minimum
of one response: they have swapped over, or they are on something that is not
the response. The value has a sign only because of the order they are
subtracted in, and "PTP: -0.04 mV" presents an impossibility as a measurement.

Seen when stored landmarks outlived the geometry that positioned them -- an
event delay moved the response 17.5 ms while the indices stayed put, leaving
PTP min at +0.013 mV and PTP max at -0.028 mV. That cause is fixed upstream;
this check is independent of it and holds for any future reason the markers end
up the wrong way round, including an analyst dragging them there.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = (ROOT / "mep_cmap" / "inspector.py").read_text(encoding="utf-8")


def _function(name, src=SRC):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found")


def test_an_inverted_pair_is_detected():
    body = _function("_refresh_status")
    assert "_ptp_inverted" in body
    assert "ptp_amp < 0" in body


def test_it_says_what_is_wrong_rather_than_printing_the_number():
    """A reader who sees '-0.04 mV' has to work out for themselves that it is
    impossible. The status says which way round the markers are."""
    body = _function("_refresh_status")
    assert "wrong way round" in body


def test_the_number_is_still_shown():
    """Hiding it would leave nothing to reconcile against the markers on
    screen, which is how the fault is recognised."""
    body = _function("_refresh_status")
    i = body.index("wrong way round")
    assert "ptp_amp:.2f" in body[max(0, i - 300):i + 60]


def test_a_normal_trial_reads_normally():
    body = _function("_refresh_status")
    assert "f'PTP:{ptp_amp:.2f}" in body or 'f"PTP:{ptp_amp:.2f}' in body


def test_the_status_line_is_coloured_for_the_fault():
    """A sentence among numbers is easy to slide past."""
    body = _function("_refresh_status")
    assert "fg=" in body
    assert "_ptp_inverted" in body.split("fg=")[1][:120]


def test_the_colour_is_restored_when_it_is_fine():
    """Otherwise a fault on one trial leaves every later one looking wrong."""
    body = _function("_refresh_status")
    assert "_status_fg_default" in body


def test_the_default_colour_is_captured_from_the_theme():
    """Hardcoding one would be wrong on another platform."""
    assert 'self.status.cget("fg")' in SRC
    assert "_status_fg_default" in SRC


# ── the arithmetic the check relies on ───────────────────────────────────────

@pytest.mark.parametrize("mx,mn,inverted", [
    (0.05, -0.06, False),      # ordinary MEP
    (0.0, 0.0, False),         # flat, degenerate but not inverted
    (-0.028, 0.013, True),     # the real case
    (-0.001, 0.0, True),       # marginal
])
def test_inversion_is_max_minus_min_below_zero(mx, mn, inverted):
    assert ((mx - mn) < 0) is inverted
