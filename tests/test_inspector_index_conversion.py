"""
Inspector indices must be converted using the stim type's OWN epoch.

The Inspector cuts its segments at -prestim_ms; the analysis cuts at the epoch
window, which since 1.4.0 is per stimulus type. Converting between the two used
the FILE-WIDE pre_ms, which was right while every type shared one window and
wrong the moment they did not.

On a real recording: conditions set a 100 ms epoch against a file-wide 20 ms,
so every index shifted by 160 samples at 2 kHz and the peak markers landed
80 ms early, in the pre-stimulus baseline. Peak-to-peak across noise read as
about zero and, the two indices being arbitrary points rather than a max and a
min, sometimes negative -- -0.01 mV reached trials.csv.

Latency hid it. man_lat is computed from the inspector offset alone, so
latencies stayed correct at 29-35 ms while amplitudes collapsed, and a results
file with plausible latencies and impossible amplitudes reads as an amplitude
problem rather than a coordinate one.
"""

import numpy as np
import pytest

from mep_cmap.pipeline import (PipelineConfig, pipeline_quantify_segments,
                               window_samples)

FS = 2000.0


def _cfg(pre_ms, prestim_ms, window_map=None):
    cfg = PipelineConfig()
    cfg.pre_ms = pre_ms
    cfg.post_ms = 300.0
    cfg.prestim_ms = prestim_ms
    cfg.window_map = dict(window_map or {})
    return cfg


def _segment(cfg, stim_type, peak_ms, trough_ms, amp=0.05):
    """A flat trace with one peak and one trough at known latencies."""
    before, after = window_samples(cfg, stim_type, FS)
    seg = np.zeros(before + after, dtype=float)
    seg[before + int(peak_ms * FS / 1000)] = +amp
    seg[before + int(trough_ms * FS / 1000)] = -amp
    return seg


def test_the_conversion_uses_the_types_own_epoch():
    """The property the fix turns on: a type with its own window must not be
    converted through the file-wide one."""
    cfg = _cfg(pre_ms=20.0, prestim_ms=100.0,
               window_map={"A": (100.0, 300.0)})
    assert window_samples(cfg, "A", FS)[0] == 200      # the type's own
    assert int(cfg.pre_ms * FS / 1000) == 40           # the file-wide one


def _run(cfg, stim_type, seg, meta):
    """One trial through the quantifier, returning its PTP in mV."""
    before = window_samples(cfg, stim_type, FS)[0]
    rows = pipeline_quantify_segments(
        stim_type, np.array([seg]),
        np.zeros((1, int(cfg.prestim_ms * FS / 1000))),
        out_set=set(), excluded_set=set(), segments_metadata=meta,
        ptp_start_idx=before + int(10 * FS / 1000),
        ptp_end_idx=before + int(50 * FS / 1000),
        fs=FS, cfg=cfg, custom_labels={}, name="test",
        auto_onsets={0: 35.0})
    # (auto_rows, manual_rows, summary_row, with_out_row, ptps_array).
    # ptps_array is the per-trial value after the manual override, which is
    # exactly the number that reaches PTP(mV) in trials.csv.
    return float(rows[4][0])


def test_a_per_type_epoch_gives_the_right_amplitude():
    """The real case: 100 ms epoch, 20 ms file-wide, markers on the response."""
    cfg = _cfg(pre_ms=20.0, prestim_ms=100.0,
               window_map={"A": (100.0, 300.0)})
    seg = _segment(cfg, "A", peak_ms=45.5, trough_ms=38.0, amp=0.05)
    insp_sb = int(cfg.prestim_ms * FS / 1000)
    meta = {("A", 0): {"ptp_max_idx": insp_sb + int(45.5 * FS / 1000),
                       "ptp_min_idx": insp_sb + int(38.0 * FS / 1000)}}
    ptp = _run(cfg, "A", seg, meta)
    assert ptp == pytest.approx(0.10, abs=1e-6)


def test_a_shared_epoch_is_unchanged():
    """The case that always worked must keep working."""
    cfg = _cfg(pre_ms=100.0, prestim_ms=100.0)
    seg = _segment(cfg, "A", peak_ms=45.5, trough_ms=38.0, amp=0.05)
    insp_sb = int(cfg.prestim_ms * FS / 1000)
    meta = {("A", 0): {"ptp_max_idx": insp_sb + int(45.5 * FS / 1000),
                       "ptp_min_idx": insp_sb + int(38.0 * FS / 1000)}}
    ptp = _run(cfg, "A", seg, meta)
    assert ptp == pytest.approx(0.10, abs=1e-6)


def test_a_negative_amplitude_is_never_published():
    """A floor, not a repair: it stops an impossible number reaching the
    results file, where it would propagate into z-scores and normalisation."""
    cfg = _cfg(pre_ms=100.0, prestim_ms=100.0)
    seg = _segment(cfg, "A", peak_ms=45.5, trough_ms=38.0, amp=0.05)
    insp_sb = int(cfg.prestim_ms * FS / 1000)
    # Markers deliberately the wrong way round.
    meta = {("A", 0): {"ptp_max_idx": insp_sb + int(38.0 * FS / 1000),
                       "ptp_min_idx": insp_sb + int(45.5 * FS / 1000)}}
    ptp = _run(cfg, "A", seg, meta)
    assert ptp >= 0


# ── every helper gets the type's own epoch, not the file-wide one ────────────
#
# The index conversion was one instance. Three more handed cfg.pre_ms and
# cfg.post_ms to functions receiving a per-type segment: onset agreement, MEP
# offset, and the CSP auto-detect branch. The offset one was visible in the
# results -- the pipeline reported 82.0 ms for a trial the Inspector put at
# 53.0 ms, because it had been told the epoch began at -20 ms when it began at
# -100 ms.

import ast
import pathlib


def _quantify_source():
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "pipeline.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "pipeline_quantify_segments"):
            return ast.unparse(node)
    raise AssertionError("pipeline_quantify_segments not found")


def test_no_helper_is_handed_the_file_wide_epoch():
    """The guard against a fifth instance. Every helper in this function is
    given a segment cut to the type's own window, so the file-wide pair
    describes a segment that was never passed."""
    body = _quantify_source()
    assert "cfg.pre_ms" not in body
    assert "cfg.post_ms" not in body


def test_the_per_type_epoch_is_resolved_once_per_trial():
    body = _quantify_source()
    assert "resolve_window(cfg, stim_type)" in body
    assert "_pre_type_ms" in body and "_post_type_ms" in body


def test_the_offset_detector_gets_it():
    """The one the Inspector disagreed with."""
    body = _quantify_source()
    i = body.index("resolve_mep_offset(")
    call = body[i:i + 700]
    assert "pre_ms=_pre_type_ms" in call
    assert "search_end_ms=_post_type_ms" in call
