"""
Guard: the PTP window must not govern where onsets can be found.

The PTP window is one setting for the whole file. The latency profile is per
stimulus type. Until v1.3.2 the onset search window was taken from the PTP
window, so an amplitude-measurement setting silently overrode the physiological
bounds configured in Stage 1b.

The failure was not a refusal to detect. On a deltoid-like case -- true onset
8.9 ms, profile 8-16 ms, PTP window 10-50 ms -- every trial returned exactly
10.00 ms, the window edge, with a between-trial SD of zero. Implausibly
consistent latencies are the signature, which makes this considerably more
dangerous than a detector returning None: it looks like unusually clean data.

The invariant asserted here is simple and strong: for a fixed latency profile,
moving the PTP window must not move the detected onset.
"""

import numpy as np
import pytest

from mep_cmap.detection import ONSET_METHOD_LABELS
from mep_cmap.pipeline import (ARTEFACT_FLOOR_MS, PipelineConfig,
                               _detect_onset_dispatch, onset_search_window)

FS = 5000.0
PRE_MS = 100
POST_MS = 200

# Deltoid / trapezius: the shipped profile starts at 8 ms, below the 10 ms
# default PTP window start. This combination is reachable from stock settings.
EARLY_PROFILE = (8.0, 16.0)
EARLY_ONSET = 9.0


def make_trial(seed=0, onset_ms=EARLY_ONSET, dur_ms=8.0, noise=0.012, amp=1.0):
    rng = np.random.default_rng(seed)
    n = int((PRE_MS + POST_MS) * FS / 1000)
    stim = int(PRE_MS * FS / 1000)
    x = rng.normal(0.0, noise, n)
    i0 = stim + int(onset_ms * FS / 1000)
    i1 = stim + int((onset_ms + dur_ms) * FS / 1000)
    k = np.arange(i1 - i0) / float(i1 - i0)
    x[i0:i1] += amp * np.sin(2 * np.pi * k) * np.sin(np.pi * k) ** 0.5
    return x


def _cfg(method, ptp_start, ptp_end=50):
    return PipelineConfig(pre_ms=PRE_MS, post_ms=POST_MS,
                          ptp_start=ptp_start, ptp_end=ptp_end,
                          onset_method=method,
                          latency_map={"TMS": EARLY_PROFILE},
                          onset_env_n_boot=200)


def _onsets(method, ptp_start, n=10):
    cfg = _cfg(method, ptp_start)
    out = [_detect_onset_dispatch(make_trial(s), FS, cfg, *EARLY_PROFILE)
           for s in range(n)]
    return [v for v in out if v is not None]


# ── The invariant ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method", sorted(ONSET_METHOD_LABELS))
def test_ptp_window_start_does_not_move_the_detected_onset(method):
    if method == "bootstrap":
        pytest.skip("legacy detector is frozen; its own bias is documented")
    a = _onsets(method, ptp_start=10)   # clips the profile
    b = _onsets(method, ptp_start=8)    # matches the profile
    c = _onsets(method, ptp_start=5)    # wider than the profile
    assert a and b and c
    assert a == b == c, (
        f"'{method}' returns different onsets depending on the PTP window. "
        f"The PTP window must not constrain onset detection."
    )


@pytest.mark.parametrize("method", ["bigoni", "rms_envelope", "cusum",
                                    "peak_fraction"])
def test_onsets_are_not_pinned_to_the_ptp_window_edge(method):
    """
    The specific artefact: every trial landing on exactly the window start.
    Zero variance across trials with genuinely different noise is impossible
    for a real detector and is what the bug produced.
    """
    vals = _onsets(method, ptp_start=10)
    assert len(vals) >= 8
    assert np.std(vals, ddof=1) > 0.0, (
        f"'{method}' returned an identical latency on every trial, which "
        f"means it is reporting a window boundary rather than a measurement."
    )
    assert not all(abs(v - 10.0) < 1e-9 for v in vals)


@pytest.mark.parametrize("method", ["bigoni", "rms_envelope", "cusum",
                                    "peak_fraction"])
def test_detected_onset_is_near_truth_despite_a_clipping_ptp_window(method):
    vals = _onsets(method, ptp_start=10)
    assert abs(float(np.mean(vals)) - EARLY_ONSET) < 1.5


# ── The window calculation ────────────────────────────────────────────────────

def test_search_window_widens_to_cover_the_profile():
    cfg = _cfg("bigoni", ptp_start=10, ptp_end=50)
    lo, hi = onset_search_window(cfg, 8.0, 16.0)
    assert lo == 8.0
    assert hi == 50.0


def test_search_window_widens_the_end_for_a_late_profile():
    cfg = _cfg("bigoni", ptp_start=10, ptp_end=30)
    lo, hi = onset_search_window(cfg, 28.0, 45.0)
    assert hi == 45.0


def test_search_window_never_narrows_the_ptp_window():
    """Widening only: a tight profile must not shrink the peak search."""
    cfg = _cfg("bigoni", ptp_start=5, ptp_end=60)
    lo, hi = onset_search_window(cfg, 18.0, 28.0)
    assert lo == 5.0
    assert hi == 60.0


def test_search_window_is_floored_at_the_artefact_blank():
    """
    A profile reaching towards the stimulus must not drag the peak search onto
    the artefact.
    """
    cfg = _cfg("bigoni", ptp_start=10, ptp_end=50)
    lo, _ = onset_search_window(cfg, 0.0, 12.0)
    assert lo == ARTEFACT_FLOOR_MS


def test_search_window_tolerates_a_missing_profile():
    cfg = _cfg("bigoni", ptp_start=10, ptp_end=50)
    assert onset_search_window(cfg, None, None) == (10.0, 50.0)


def test_search_window_stays_ordered_for_a_degenerate_profile():
    cfg = _cfg("bigoni", ptp_start=10, ptp_end=50)
    lo, hi = onset_search_window(cfg, 40.0, 5.0)
    assert hi > lo


# ── No regression on the ordinary case ────────────────────────────────────────

@pytest.mark.parametrize("method", ["bigoni", "bigoni_walkback",
                                    "rms_envelope", "cusum", "peak_fraction"])
def test_ordinary_hand_muscle_case_is_unaffected(method):
    """
    Profile 18-28 ms inside a 5-60 ms PTP window: nothing is clipped, so the
    widening must be a no-op and detection must still land near truth.
    """
    cfg = PipelineConfig(pre_ms=PRE_MS, post_ms=POST_MS,
                         ptp_start=5, ptp_end=60,
                         onset_method=method,
                         latency_map={"TMS": (15.0, 35.0)},
                         onset_env_n_boot=200)
    vals = [_detect_onset_dispatch(
                make_trial(s, onset_ms=22.0, dur_ms=18.0), FS, cfg, 15.0, 35.0)
            for s in range(10)]
    vals = [v for v in vals if v is not None]
    assert len(vals) >= 8
    assert abs(float(np.mean(vals)) - 22.0) < 2.0


# ── The pinning diagnostic ────────────────────────────────────────────────────

def _collect_log():
    msgs = []
    return msgs, lambda *a: msgs.append(" ".join(str(x) for x in a))


def test_pinned_onsets_are_reported_to_the_analyst():
    """
    The legacy bootstrap detector falls back to the physiological floor when
    its backward scan finds nothing, so it returns the bound itself on every
    trial. The value is plausible and perfectly consistent, which is exactly
    what makes it dangerous. It must not pass silently.
    """
    from mep_cmap.pipeline import pipeline_detect_onsets

    segs = np.vstack([make_trial(s, onset_ms=22.0, dur_ms=18.0)
                      for s in range(12)])
    cfg = PipelineConfig(pre_ms=PRE_MS, post_ms=POST_MS,
                         ptp_start=5, ptp_end=60,
                         onset_method="bootstrap",
                         latency_map={"TMS": (20.0, 35.0)})
    msgs, log = _collect_log()
    s0 = int((PRE_MS + 5) * FS / 1000)
    s1 = int((PRE_MS + 60) * FS / 1000)
    onsets = pipeline_detect_onsets("TMS", segs, set(), s0, s1, FS, cfg,
                                    log_callback=log)
    vals = [v for v in onsets.values() if v is not None]
    assert vals and len(set(vals)) == 1        # genuinely pinned
    assert any("latency bound" in m for m in msgs), (
        "onsets collapsed onto the search bound without any warning"
    )


def test_no_warning_when_onsets_are_genuine_measurements():
    from mep_cmap.pipeline import pipeline_detect_onsets

    segs = np.vstack([make_trial(s, onset_ms=22.0, dur_ms=18.0)
                      for s in range(12)])
    cfg = PipelineConfig(pre_ms=PRE_MS, post_ms=POST_MS,
                         ptp_start=5, ptp_end=60,
                         onset_method="bigoni",
                         latency_map={"TMS": (15.0, 35.0)})
    msgs, log = _collect_log()
    s0 = int((PRE_MS + 5) * FS / 1000)
    s1 = int((PRE_MS + 60) * FS / 1000)
    pipeline_detect_onsets("TMS", segs, set(), s0, s1, FS, cfg,
                           log_callback=log)
    assert not any("latency bound" in m for m in msgs)


def test_diagnostic_fires_from_three_trials():
    """
    Three is the threshold. It was four until a real peripheral condition with
    exactly three detected trials sat 3/3 on its lower bound and passed
    silently -- the conditions with fewest usable trials are the ones where a
    wrong latency profile does most damage.
    """
    from mep_cmap.pipeline import _warn_if_onsets_pinned_to_a_bound

    msgs, log = _collect_log()
    _warn_if_onsets_pinned_to_a_bound(
        "TMS", {0: 20.0, 1: 20.0, 2: 20.0}, FS, 20.0, 35.0, log)
    assert any("latency bound" in m for m in msgs)


def test_diagnostic_stays_quiet_on_fewer_than_three_trials():
    from mep_cmap.pipeline import _warn_if_onsets_pinned_to_a_bound

    msgs, log = _collect_log()
    _warn_if_onsets_pinned_to_a_bound(
        "TMS", {0: 20.0, 1: 20.0}, FS, 20.0, 35.0, log)
    assert not msgs


def test_widening_cannot_produce_an_onset_outside_the_profile():
    """
    Widening the SEARCH window must not widen the accepted RESULT: the
    physiological bounds are still enforced by every detector.
    """
    cfg = PipelineConfig(pre_ms=PRE_MS, post_ms=POST_MS,
                         ptp_start=2, ptp_end=90,
                         onset_method="bigoni",
                         latency_map={"TMS": (18.0, 28.0)})
    for seed in range(10):
        sig = make_trial(seed, onset_ms=9.0)      # well outside 18-28
        got = _detect_onset_dispatch(sig, FS, cfg, 18.0, 28.0)
        assert got is None or 18.0 <= got <= 28.0
