"""
Tests for the envelope, CUSUM, consensus and offset detectors.

Ground truth is synthetic: every trial is built here with a known onset and
offset, so assertions are on measured bias and detection rate rather than on
agreement with a reference implementation. No participant data is required.

Tolerances are deliberately loose relative to measured performance. They exist
to catch regressions -- a detector silently losing its bias correction, or a
threshold change admitting noise -- not to pin down exact values, which depend
on the RNG stream and would make the suite brittle across numpy versions.
"""

import numpy as np
import pytest

from mep_cmap.detection.envelope_stats import (
    bootstrap_runlength_criterion,
    compute_envelope_baseline,
    compute_rms_envelope,
    find_sustained_run,
    passes_width_guard,
)
from mep_cmap.detection.offset_detection import (
    OFFSET_SOURCES,
    detect_mep_offset,
    resolve_mep_offset,
)
from mep_cmap.detection.onset_consensus import (
    CONSENSUS_DEFAULT_METHODS,
    compute_onset_agreement,
    detect_mep_onset_consensus,
)
from mep_cmap.detection.onset_cusum import detect_mep_onset_cusum
from mep_cmap.detection.onset_rms_envelope import detect_mep_onset_rms_envelope
from mep_cmap.detection.tkeo import apply_tkeo

FS = 5000.0
PRE_MS = 100.0
POST_MS = 150.0
TRUE_ONSET = 22.0
TRUE_DUR = 18.0
TRUE_OFFSET = TRUE_ONSET + TRUE_DUR

COMMON = dict(pre_ms=PRE_MS, search_start_ms=5, search_end_ms=60,
              min_latency_ms=15.0, max_latency_ms=35.0,
              min_peak_amplitude=0.05)


def make_trial(seed=0, noise=0.012, amp=1.0, onset_ms=TRUE_ONSET,
               dur_ms=TRUE_DUR, phases=2, baseline_emg=0.0):
    """Synthetic MEP on Gaussian noise with an exactly known onset/offset."""
    rng = np.random.default_rng(seed)
    n = int((PRE_MS + POST_MS) * FS / 1000.0)
    stim = int(PRE_MS * FS / 1000.0)
    x = rng.normal(0.0, noise, n)
    if baseline_emg > 0:
        x += rng.normal(0.0, baseline_emg, n)
    i0 = stim + int(onset_ms * FS / 1000.0)
    i1 = stim + int((onset_ms + dur_ms) * FS / 1000.0)
    k = np.arange(i1 - i0) / float(i1 - i0)
    x[i0:i1] += amp * np.sin(phases * np.pi * k) * np.sin(np.pi * k) ** 0.5
    return x


def onset_errors(fn, n=25, detector_kwargs=None, **trial_kwargs):
    detector_kwargs = detector_kwargs or {}
    errs = []
    for s in range(n):
        lat = fn(make_trial(seed=s, **trial_kwargs), FS,
                 **COMMON, **detector_kwargs)
        if lat is not None:
            errs.append(lat - trial_kwargs.get("onset_ms", TRUE_ONSET))
    return np.asarray(errs)


# ── TKEO ──────────────────────────────────────────────────────────────────────

def test_tkeo_preserves_length_and_avoids_zero_edges():
    x = np.sin(2 * np.pi * 50 * np.arange(1000) / FS)
    psi = apply_tkeo(x)
    assert psi.size == x.size
    assert psi[0] == psi[1] and psi[-1] == psi[-2]


def test_tkeo_amplifies_higher_frequencies_at_equal_amplitude():
    t = np.arange(2000) / FS
    slow = np.mean(np.abs(apply_tkeo(np.sin(2 * np.pi * 50 * t))))
    fast = np.mean(np.abs(apply_tkeo(np.sin(2 * np.pi * 200 * t))))
    assert fast > 5 * slow


def test_tkeo_handles_degenerate_input():
    assert apply_tkeo(np.array([1.0, 2.0])).size == 2
    assert apply_tkeo(np.array([])).size == 0


# ── Envelope statistics ───────────────────────────────────────────────────────

def test_envelope_is_far_less_dispersed_than_rectified_signal():
    """The premise of the whole approach: envelope SD is much smaller."""
    noise = np.random.default_rng(1).normal(0, 0.01, 20000)
    env = compute_rms_envelope(noise, FS, window_ms=5.0)
    raw_ratio = np.std(np.abs(noise), ddof=1) / np.mean(np.abs(noise))
    env_ratio = np.std(env, ddof=1) / np.mean(env)
    assert env_ratio < raw_ratio / 3.0


def test_envelope_length_and_edges():
    noise = np.random.default_rng(2).normal(0, 0.01, 5000)
    env = compute_rms_envelope(noise, FS, window_ms=5.0)
    assert env.size == noise.size
    # Reflection padding: no roll-off at the segment start.
    assert abs(env[:100].mean() - env[500:].mean()) / env[500:].mean() < 0.25


def test_envelope_threshold_is_unclipped():
    """Regression guard: onset_bootstrap clips to [1.5mu, 5mu]; this must not."""
    env = compute_rms_envelope(
        np.random.default_rng(3).normal(0, 0.01, 5000), FS, 5.0)
    b = compute_envelope_baseline(env, criterion=2.5)
    assert b.threshold == pytest.approx(b.mu + 2.5 * b.sd)


def test_envelope_baseline_rejects_unusable_input():
    assert compute_envelope_baseline(np.zeros(100)) is None
    assert compute_envelope_baseline(np.array([1.0, 2.0])) is None


def test_block_bootstrap_demands_much_longer_runs_than_iid():
    """
    The i.i.d. bootstrap ignores envelope autocorrelation and reports a
    criterion of ~1-2 samples, which admits pure noise as a response.
    """
    env = compute_rms_envelope(
        np.random.default_rng(4).normal(0, 0.01, 20000), FS, 5.0)
    iid = bootstrap_runlength_criterion(env, n_boot=300, block_samples=1)
    block = bootstrap_runlength_criterion(env, n_boot=300, block_samples=25)
    assert block > 3 * iid


def test_runlength_criterion_is_reproducible():
    env = compute_rms_envelope(
        np.random.default_rng(5).normal(0, 0.01, 10000), FS, 5.0)
    kw = dict(n_boot=200, seed=42, block_samples=25)
    assert (bootstrap_runlength_criterion(env, **kw)
            == bootstrap_runlength_criterion(env, **kw))


def test_find_sustained_run_honours_duration_and_bounds():
    v = np.zeros(100)
    v[40:44] = 10.0
    assert find_sustained_run(v, 1.0, 4) == 40
    assert find_sustained_run(v, 1.0, 6) is None
    assert find_sustained_run(v, 1.0, 4, lo=50) is None


def test_width_guard_rejects_a_spike_and_accepts_a_response():
    fine = np.zeros(200)
    fine[100] = 10.0                      # single-sample artefact
    assert not passes_width_guard(fine, 1.0, 95, 15)
    fine2 = np.zeros(200)
    fine2[100:140] = 10.0                 # sustained response
    assert passes_width_guard(fine2, 1.0, 95, 15)


# ── RMS envelope onset ────────────────────────────────────────────────────────

def test_rms_envelope_onset_is_accurate_and_complete():
    e = onset_errors(detect_mep_onset_rms_envelope,
                     detector_kwargs=dict(n_boot=200))
    assert e.size == 25
    assert abs(e.mean()) < 1.5
    assert e.std(ddof=1) < 1.5


@pytest.mark.parametrize("window_ms", [2.0, 5.0, 10.0, 20.0])
def test_refinement_makes_latency_insensitive_to_window_width(window_ms):
    """
    The central claim for the refinement stage. Unrefined, mean bias runs from
    -0.4 ms at W=2 to -7 ms at W=20 because a centred window leads the signal
    by ~W/2. Refined, every width must land near truth.
    """
    e = onset_errors(detect_mep_onset_rms_envelope, n=20,
                     detector_kwargs=dict(n_boot=200, env_window_ms=window_ms,
                                          refine_on_raw=True))
    assert e.size >= 18
    assert abs(e.mean()) < 1.5


def test_unrefined_envelope_shows_the_expected_early_bias():
    """Guards the diagnosis, not just the fix: without refinement, wide
    windows must still detect early."""
    e = onset_errors(detect_mep_onset_rms_envelope, n=20,
                     detector_kwargs=dict(n_boot=200, env_window_ms=10.0,
                                          refine_on_raw=False,
                                          min_response_ms=0))
    assert e.size > 0
    assert e.mean() < -2.0


def test_rms_envelope_respects_amplitude_gate_and_latency_bounds():
    tiny = make_trial(seed=99, amp=0.01)
    assert detect_mep_onset_rms_envelope(tiny, FS, n_boot=100, **COMMON) is None
    rejected = sum(
        detect_mep_onset_rms_envelope(make_trial(seed=s, onset_ms=45.0), FS,
                                      n_boot=200, **COMMON) is None
        for s in range(20))
    assert rejected >= 18


def test_rms_envelope_rejects_single_sample_artefact():
    """Window smearing widens a lone spike into an apparent response; the
    width guard is what stops it being reported as an onset."""
    called = 0
    for s in range(20):
        x = np.random.default_rng(7000 + s).normal(
            0, 0.012, int((PRE_MS + POST_MS) * FS / 1000.0))
        x[int(PRE_MS * FS / 1000.0) + 110] += 0.6
        if detect_mep_onset_rms_envelope(x, FS, n_boot=200, **COMMON) is not None:
            called += 1
    assert called <= 2


def test_rms_envelope_false_positive_rate_on_pure_noise():
    fp = 0
    for s in range(60):
        x = np.random.default_rng(9000 + s).normal(
            0, 0.012, int((PRE_MS + POST_MS) * FS / 1000.0))
        kw = dict(COMMON)
        kw["min_peak_amplitude"] = 0.0     # isolate the run-length criterion
        if detect_mep_onset_rms_envelope(x, FS, n_boot=200, **kw) is not None:
            fp += 1
    assert fp <= 6


# ── CUSUM onset ───────────────────────────────────────────────────────────────

def test_cusum_onset_is_accurate():
    e = onset_errors(detect_mep_onset_cusum)
    assert e.size >= 23
    assert abs(e.mean()) < 1.5
    assert e.std(ddof=1) < 2.0


def test_cusum_windowed_reset_prevents_early_anchoring():
    """
    Unbounded CUSUM lets a favourable noise run hold the accumulator
    marginally positive, anchoring the change point several ms early. The
    bounded window must keep gross early errors out.
    """
    e = onset_errors(detect_mep_onset_cusum, n=40,
                     detector_kwargs=dict(max_accum_ms=10.0))
    assert e.size >= 35
    assert (e < -3.0).sum() == 0


def test_cusum_h_controls_false_alarms():
    n = int((PRE_MS + POST_MS) * FS / 1000.0)
    fp = 0
    for s in range(60):
        x = np.random.default_rng(9000 + s).normal(0, 0.012, n)
        kw = dict(COMMON)
        kw["min_peak_amplitude"] = 0.0
        if detect_mep_onset_cusum(x, FS, **kw) is not None:
            fp += 1
    assert fp <= 3


def test_cusum_retains_sensitivity_to_small_responses():
    e = onset_errors(detect_mep_onset_cusum, n=20, amp=0.15, noise=0.03)
    assert e.size >= 17


# ── Consensus ─────────────────────────────────────────────────────────────────

def test_consensus_runs_all_default_members_and_agrees_with_truth():
    ag = compute_onset_agreement(make_trial(seed=3), FS,
                                 params={"onset_env_n_boot": 200}, **COMMON)
    assert ag.n_attempted == len(CONSENSUS_DEFAULT_METHODS)
    assert set(ag.per_method) == set(CONSENSUS_DEFAULT_METHODS)
    assert abs(ag.consensus_ms - TRUE_ONSET) < 2.0
    assert ag.spread_ms is not None and ag.iqr_ms is not None


def test_consensus_scalar_wrapper_matches_agreement_median():
    kw = dict(params={"onset_env_n_boot": 200}, **COMMON)
    ag = compute_onset_agreement(make_trial(seed=4), FS, **kw)
    assert detect_mep_onset_consensus(make_trial(seed=4), FS, **kw) \
        == ag.consensus_ms


def test_consensus_tolerates_unknown_and_failing_members():
    ag = compute_onset_agreement(make_trial(seed=5), FS,
                                 methods=("bigoni", "no_such_method"),
                                 **COMMON)
    assert ag.n_attempted == 1


def test_consensus_returns_none_when_nothing_detects():
    flat = np.zeros(int((PRE_MS + POST_MS) * FS / 1000.0))
    ag = compute_onset_agreement(flat, FS, **COMMON)
    assert ag.consensus_ms is None
    assert ag.n_detected == 0


def test_agreement_spread_widens_as_snr_falls():
    def median_spread(noise):
        vals = [compute_onset_agreement(
                    make_trial(seed=s, noise=noise), FS,
                    params={"onset_env_n_boot": 150}, **COMMON).spread_ms
                for s in range(20)]
        vals = [v for v in vals if v is not None]
        return float(np.median(vals))
    # Only asserted where all members still detect; at very low SNR members
    # drop out entirely and spread can fall again.
    assert median_spread(0.01) <= median_spread(0.03)


# ── Offset ────────────────────────────────────────────────────────────────────

def test_offset_is_accurate_and_complete():
    errs = []
    for s in range(25):
        o = detect_mep_offset(make_trial(seed=s), FS, onset_ms=TRUE_ONSET,
                              pre_ms=PRE_MS, search_end_ms=120, n_boot=200)
        if o is not None:
            errs.append(o - TRUE_OFFSET)
    e = np.asarray(errs)
    assert e.size == 25
    assert abs(e.mean()) < 3.0


@pytest.mark.parametrize("window_ms", [2.0, 5.0, 10.0, 20.0])
def test_offset_refinement_makes_latency_insensitive_to_window_width(window_ms):
    errs = []
    for s in range(15):
        o = detect_mep_offset(make_trial(seed=s), FS, onset_ms=TRUE_ONSET,
                              pre_ms=PRE_MS, search_end_ms=120,
                              env_window_ms=window_ms, refine_forward=True,
                              n_boot=200)
        if o is not None:
            errs.append(o - TRUE_OFFSET)
    e = np.asarray(errs)
    assert e.size >= 13
    assert abs(e.mean()) < 2.5


def test_offset_not_truncated_at_an_internal_trough_of_a_polyphasic_mep():
    """A polyphasic response dips towards baseline mid-way; the sustained
    return requirement must not end the response there."""
    x = make_trial(seed=11, phases=4)
    o = detect_mep_offset(x, FS, onset_ms=TRUE_ONSET, pre_ms=PRE_MS,
                          search_end_ms=120, n_boot=200)
    assert o is not None
    assert o > TRUE_ONSET + 0.6 * TRUE_DUR


def test_offset_requires_an_onset():
    assert detect_mep_offset(make_trial(seed=7), FS, onset_ms=None,
                             pre_ms=PRE_MS) is None


def test_offset_precedence_manual_beats_csp():
    r = resolve_mep_offset(make_trial(seed=7), FS, onset_ms=TRUE_ONSET,
                           manual_offset_ms=41.0, csp_start_ms=55.0,
                           csp_enabled=True)
    assert r.source == "manual" and r.offset_ms == 41.0
    assert r.duration_ms == pytest.approx(41.0 - TRUE_ONSET)


def test_offset_precedence_csp_beats_envelope():
    r = resolve_mep_offset(make_trial(seed=7), FS, onset_ms=TRUE_ONSET,
                           csp_start_ms=55.0, csp_enabled=True)
    assert r.source == "csp_start" and r.offset_ms == 55.0


def test_offset_precedence_falls_through_to_envelope():
    # cSP present but disabled for this stimulus type.
    r = resolve_mep_offset(make_trial(seed=7), FS, onset_ms=TRUE_ONSET,
                           csp_start_ms=55.0, csp_enabled=False,
                           pre_ms=PRE_MS, search_end_ms=120, n_boot=200)
    assert r.source == "envelope"
    # cSP enabled but not detected on this trial.
    r2 = resolve_mep_offset(make_trial(seed=7), FS, onset_ms=TRUE_ONSET,
                            csp_start_ms=None, csp_enabled=True,
                            pre_ms=PRE_MS, search_end_ms=120, n_boot=200)
    assert r2.source == "envelope"


def test_offset_source_none_without_onset():
    r = resolve_mep_offset(make_trial(seed=7), FS, onset_ms=None,
                           csp_enabled=False, pre_ms=PRE_MS)
    assert r.source == "none"
    assert r.offset_ms is None and r.duration_ms is None


def test_every_reported_source_is_declared():
    for src in ("manual", "csp_start", "envelope", "none"):
        assert src in OFFSET_SOURCES


# ── Cross-cutting ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fn,kw", [
    (detect_mep_onset_rms_envelope, dict(n_boot=100)),
    (detect_mep_onset_cusum, {}),
])
def test_detectors_survive_degenerate_input(fn, kw):
    assert fn(np.zeros(50), FS, **COMMON, **kw) is None
    assert fn(np.zeros(3000), FS, **COMMON, **kw) is None
    assert fn(np.full(3000, 5.0), FS, **COMMON, **kw) is None


@pytest.mark.parametrize("fn,kw", [
    (detect_mep_onset_rms_envelope, dict(n_boot=100)),
    (detect_mep_onset_cusum, {}),
])
def test_detectors_are_deterministic(fn, kw):
    x = make_trial(seed=12)
    assert fn(x, FS, **COMMON, **kw) == fn(x, FS, **COMMON, **kw)


def test_envelope_and_cusum_tolerate_elevated_background_emg():
    """Active-contraction paradigms: a raised, noisy baseline must not stop
    detection outright, even though accuracy is expected to degrade."""
    for fn, kw in ((detect_mep_onset_rms_envelope, dict(n_boot=200)),
                   (detect_mep_onset_cusum, {})):
        e = onset_errors(fn, n=20, detector_kwargs=kw, baseline_emg=0.10)
        assert e.size >= 15
        assert abs(e.mean()) < 3.0
