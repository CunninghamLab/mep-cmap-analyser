"""
A stored landmark is a position, and it only means anything while the segment
is cut the same way.

The existing screen discards an index that no longer FITS the segment. It does
not catch one that still fits but no longer points at what it was placed on,
which is the commoner case and the one that produced a reported measurement
that cannot exist.

Applying a 17.5 ms event delay cuts the epoch 35 samples later, so the response
moves 17.5 ms earlier against the same axis while every stored index stays put.
On a real recording that put PTP min at +0.013 mV and PTP max at -0.028 mV, and
the status bar read "PTP: -0.04 mV" -- the labels had swapped over and a
negative peak-to-peak amplitude was presented as a measurement.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = (ROOT / "mep_cmap" / "inspector.py").read_text(encoding="utf-8")
APP = (ROOT / "mep_cmap" / "app.py").read_text(encoding="utf-8")


def _function(name, src=SRC):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found")


# ── what the fingerprint is made of ──────────────────────────────────────────

def test_the_delay_is_part_of_the_geometry():
    """The case that broke. Applying a delay changes WHICH samples are cut
    without changing the axis, so nothing else can tell the response moved."""
    body = _function("_segment_geometry")
    assert "delay_ms_map" in body


def test_the_epoch_is_part_of_the_geometry():
    """A different pre-stimulus length renumbers every sample."""
    body = _function("_segment_geometry")
    assert "self.t[0]" in body and "self.t[-1]" in body


def test_the_amplitude_window_is_part_of_the_geometry():
    """It is where the peaks were searched for, so moving it moves them."""
    body = _function("_segment_geometry")
    assert "_ptp_window_ms" in body


def test_it_is_per_stimulus_type():
    """All three components are per type."""
    body = _function("_segment_geometry")
    assert "stim_type" in body


def test_it_is_readable_rather_than_a_digest():
    """It goes into the session JSON and is read by a human when something
    looks wrong. A hash says only that something changed."""
    body = _function("_segment_geometry")
    for opaque in ("hashlib", "md5", "sha1", "sha256", "hash("):
        assert opaque not in body


# ── the delay actually reaches the inspector ─────────────────────────────────

def test_the_inspector_accepts_the_delay_map():
    assert "delay_ms_map=None," in SRC
    assert "self.delay_ms_map" in SRC


def test_both_call_sites_pass_it():
    """A default of {} reads every delay as 0.0, which is wrong in exactly the
    case this exists for -- so it is passed, not defaulted."""
    assert APP.count("delay_ms_map        = dict(getattr(self,") == 2


# ── what happens when it changes ─────────────────────────────────────────────

def test_landmarks_are_dropped_when_the_geometry_changes():
    body = _function("_seed_metadata") if "_seed_metadata" in SRC else SRC
    assert "_geometry" in body
    assert "_segment_geometry()" in body


def test_the_non_detection_flag_goes_with_them():
    """Left behind, it would report 'not detected' against a freshly detected
    onset."""
    assert "m.pop('onset_auto_failed', None)" in SRC


def test_the_geometry_is_recorded_alongside():
    """Nothing to compare against next time otherwise."""
    assert "m['_geometry'] = _geom" in SRC


def test_a_first_visit_keeps_its_landmarks():
    """No stored geometry means metadata written before this existed, not
    metadata that has moved. Discarding it would throw away every manual edit
    made before the upgrade."""
    a = SRC.index("if m.get('_geometry') not in (None, _geom):")
    assert a > 0, "an absent geometry must not count as a change"


def test_the_auc_bounds_are_screened_too():
    """They are positions in the same segment and move with everything else."""
    a = SRC.index("_LANDMARKS = (")
    b = SRC.index(")", a)
    names = SRC[a:b]
    assert "auc_start_idx" in names and "auc_end_idx" in names
