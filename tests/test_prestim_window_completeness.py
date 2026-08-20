"""
Baselines are equal length within a stimulus type, and no trial is lost to a
few missing samples.

The analysis window was checked for completeness and the pre-stimulus one was
not, so max(0, ...) clamped it at the start of the recording and produced a
SHORT array beside full-length ones. np.array() over the mixture raised
"inhomogeneous shape" three stages later, where nothing pointed back here.

Only the BASELINE is short in that case: the epoch is complete and the response
is perfectly measurable, because the baseline additionally clears the artefact
gap. Reproduced from a real file -- 21 stimuli, the first at 0.1 s, 2000 Hz, a
100 ms baseline and a 5 ms gap -- where the first trial's baseline is short by
5% and its MEP is fine.
"""

import numpy as np
import pytest

from mep_cmap.pipeline import (MIN_BASELINE_FRACTION, PipelineConfig,
                               pipeline_extract_segments)

FS = 2000.0


def _cfg(**kw):
    cfg = PipelineConfig()
    cfg.prestim_ms = 100.0
    cfg.rms_guard_ms = getattr(cfg, "rms_guard_ms", 0.0) or 0.0
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _recording(n_seconds=12.0):
    n = int(n_seconds * FS)
    return np.arange(n) / FS, np.zeros(n, dtype=float)


def _onsets():
    """The real file's timing: first stimulus at 0.1 s, then every ~0.5 s."""
    return [0.1] + [0.6005 + 0.5005 * i for i in range(20)]


def _extract(onsets, cfg, log=None):
    time, emg = _recording()
    return pipeline_extract_segments(time, emg, {"A": onsets}, ["A"], FS, cfg,
                                     log)


# ── the crash ────────────────────────────────────────────────────────────────

def test_a_stimulus_too_early_for_a_full_baseline_does_not_crash():
    out = _extract(_onsets(), _cfg(gap_ms_map={"A": 5.0}))
    pre = np.array([s[1] for s in out["A"]])        # raised ValueError before
    assert pre.ndim == 2


def test_every_baseline_in_a_type_is_the_same_length():
    """Equal length is the point: the outlier test compares one trial's
    baseline RMS against the others'."""
    out = _extract(_onsets(), _cfg(gap_ms_map={"A": 5.0}))
    assert len({len(s[1]) for s in out["A"]}) == 1


# ── trimming, not dropping ───────────────────────────────────────────────────

def test_no_trial_is_lost_to_a_few_missing_samples():
    """All 21 survive. The trial that would have been dropped is systematically
    the first one, which is not random with respect to anything measured across
    a session."""
    out = _extract(_onsets(), _cfg(gap_ms_map={"A": 5.0}))
    assert len(out["A"]) == 21


def test_the_trim_is_only_as_deep_as_it_needs_to_be():
    """5 ms of gap costs 5 ms of baseline, not the whole window."""
    cfg = _cfg(gap_ms_map={"A": 5.0})
    out = _extract(_onsets(), cfg)
    n = len(out["A"][0][1])
    assert n == int(cfg.prestim_ms * FS / 1000) - int(5.0 * FS / 1000)


def test_the_baseline_is_trimmed_from_the_front():
    """Its end is fixed relative to the stimulus; the far end is what is
    missing on the clamped trial."""
    time, emg = _recording()
    emg[:] = np.arange(len(emg))                  # position is recoverable
    cfg = _cfg(gap_ms_map={"A": 5.0})
    out = pipeline_extract_segments(time, emg, {"A": _onsets()}, ["A"], FS, cfg)
    # Second trial: baseline must still END where the gap says, not start there.
    _emg, pre, t = out["A"][1]
    idx = int(round(t * FS))
    assert pre[-1] == pytest.approx(idx - int(5.0 * FS / 1000) - 1)


def test_the_trim_is_reported():
    said = []
    _extract(_onsets(), _cfg(gap_ms_map={"A": 5.0}), said.append)
    assert any("trimmed to" in m for m in said)
    assert any("comparable" in m for m in said)


def test_a_recording_with_room_for_every_baseline_is_untouched():
    cfg = _cfg(gap_ms_map={"A": 5.0})
    out = _extract([t + 1.0 for t in _onsets()], cfg)
    assert len(out["A"]) == 21
    assert len(out["A"][0][1]) == int(cfg.prestim_ms * FS / 1000)


def test_nothing_is_reported_when_nothing_is_trimmed():
    said = []
    _extract([t + 1.0 for t in _onsets()], _cfg(gap_ms_map={"A": 5.0}),
             said.append)
    assert said == []


# ── the floor ────────────────────────────────────────────────────────────────
#
# Reachable only where the epoch is SHORTER than the baseline, which is the
# ordinary case: a 20 ms epoch either side of the pulse against a 100 ms
# baseline. A stimulus early enough to lose most of its baseline but keep its
# epoch is the situation the floor exists for. Where the epoch is the longer of
# the two, the analysis-window check drops the trial first and the floor never
# runs.

def _short_epoch_cfg():
    return _cfg(pre_ms=20.0, post_ms=100.0, gap_ms_map={"A": 5.0})


def test_a_baseline_in_name_only_drops_the_trial_instead():
    """Trimming every trial to a quarter of the requested window to save one is
    worse than losing that one."""
    cfg = _short_epoch_cfg()
    out = _extract([0.03] + [1.0 + 0.5 * i for i in range(10)], cfg)
    assert len(out["A"]) == 10                     # the 0.03 s trial went
    assert len(out["A"][0][1]) == int(cfg.prestim_ms * FS / 1000)


def test_the_dropped_trial_is_reported():
    said = []
    _extract([0.03] + [1.0 + 0.5 * i for i in range(10)],
             _short_epoch_cfg(), said.append)
    assert any("skipped" in m for m in said)


def test_the_floor_is_half_the_requested_window():
    assert MIN_BASELINE_FRACTION == 0.5


def test_exactly_at_the_floor_the_trial_is_kept():
    """A boundary that drops what it says it keeps is a boundary nobody can
    reason about."""
    cfg = _cfg(pre_ms=20.0, post_ms=100.0, gap_ms_map={"A": 0.0},
               rms_guard_ms=0.0)
    half = cfg.prestim_ms / 2.0 / 1000.0           # baseline exactly 50%
    out = _extract([half] + [1.0 + 0.5 * i for i in range(5)], cfg)
    assert len(out["A"]) == 6


# ── unchanged behaviour ──────────────────────────────────────────────────────

def test_the_analysis_window_check_still_applies():
    """A stimulus at the very end has no post-stimulus window."""
    time, emg = _recording(n_seconds=2.0)
    out = pipeline_extract_segments(time, emg, {"A": [1.999]}, ["A"], FS,
                                    _cfg(gap_ms_map={"A": 5.0}))
    assert "A" not in out or out["A"] == []


def test_no_log_callback_is_not_an_error():
    out = _extract(_onsets(), _cfg(gap_ms_map={"A": 5.0}))
    assert len(out["A"]) == 21
