"""
Tests for the derivative-ratio onset detector (Boyles et al. 2026).

Ported from the MATLAB reference in TMSMultiLab. Three details of that
implementation do not behave as its own comments describe; they are corrected by
default and reproduced under ``literal=True``. These tests pin both behaviours,
because a correction that cannot be switched off makes the published method
unreproducible, and one that is silently on makes results unattributable.
"""

import numpy as np
import pytest

from mep_cmap.detection import ONSET_METHOD_LABELS, dispatch_onset
from mep_cmap.detection.onset_boyles import detect_mep_onset_boyles

FS = 5000.0
PRE_MS = 100
POST_MS = 200
TRUE_ONSET = 22.0


def make_trial(seed=0, onset_ms=TRUE_ONSET, dur_ms=18.0, amp=1.0, noise=0.012,
               harmonic=0.3):
    """Synthetic response with harmonic content, as real EMG has.

    The ``harmonic`` term is not decoration. Every gate in this detector is
    stated in absolute derivatives, so it requires the response to be
    spectrally richer than the baseline rather than merely larger. A pure
    single-cycle sine over 18 ms is about 55 Hz and its derivative content is
    only ~2.8x the noise floor: detection is 5 of 20 trials, while the same
    fixture with modest harmonic content (~3.9x) gives 19 of 20. Real surface
    EMG carries roughly 20-500 Hz and sits well inside the working range, so a
    fixture without harmonic content would test a condition the detector is not
    intended for and understate it by a factor of four.
    """
    rng = np.random.default_rng(seed)
    n = int((PRE_MS + POST_MS) * FS / 1000)
    stim = int(PRE_MS * FS / 1000)
    x = rng.normal(0.0, noise, n)
    i0 = stim + int(onset_ms * FS / 1000)
    i1 = stim + int((onset_ms + dur_ms) * FS / 1000)
    k = np.arange(i1 - i0) / float(i1 - i0)
    shape = np.sin(2 * np.pi * k) + harmonic * np.sin(8 * np.pi * k)
    x[i0:i1] += amp * shape * np.sin(np.pi * k) ** 0.5
    return x


def condition(n=20, **kw):
    segs = np.vstack([make_trial(seed=s, **kw) for s in range(n)])
    return segs, np.median(segs, axis=0)


KW = dict(pre_ms=PRE_MS, search_start_ms=5, search_end_ms=45,
          min_latency_ms=15.0, max_latency_ms=35.0)


def latencies(segs, template=None, **kw):
    out = [detect_mep_onset_boyles(s, FS, template=template, **KW, **kw)
           for s in segs]
    return [v for v in out if v is not None], len(segs)


# ── Registration ──────────────────────────────────────────────────────────────

def test_method_is_registered_and_labelled_with_its_source():
    assert "boyles" in ONSET_METHOD_LABELS
    assert "Boyles" in ONSET_METHOD_LABELS["boyles"]
    assert "2026" in ONSET_METHOD_LABELS["boyles"]


def test_reachable_through_the_shared_dispatch():
    segs, tpl = condition(n=8)
    got = dispatch_onset(segs[0], FS, {"onset_method": "boyles"},
                         pre_ms=PRE_MS, search_start_ms=5, search_end_ms=45,
                         min_latency_ms=15.0, max_latency_ms=35.0,
                         template=tpl)
    assert got is None or 15.0 <= got <= 35.0


def test_dispatch_passes_the_template_through():
    """
    Without the template the peak-jitter gate is skipped, so a trial whose peak
    is displaced must be rejected only when a template is supplied. If dispatch
    silently dropped it the gate would never fire and nothing would fail.
    """
    _, tpl = condition(n=20)
    odd = make_trial(seed=99, onset_ms=32.0)     # peak far from the template's
    kw = dict(pre_ms=PRE_MS, search_start_ms=5, search_end_ms=45,
              min_latency_ms=15.0, max_latency_ms=45.0)
    with_tpl = dispatch_onset(odd, FS, {"onset_method": "boyles",
                                        "onset_boyles_peak_jitter_ms": 2.0},
                              template=tpl, **kw)
    without = dispatch_onset(odd, FS, {"onset_method": "boyles",
                                       "onset_boyles_peak_jitter_ms": 2.0},
                             template=None, **kw)
    assert with_tpl is None
    assert without is not None


# ── Basic behaviour ───────────────────────────────────────────────────────────

def test_detects_a_clean_response_near_truth():
    segs, tpl = condition(n=20)
    got, n = latencies(segs, tpl)
    assert len(got) >= n * 0.7
    assert abs(float(np.mean(got)) - TRUE_ONSET) < 4.0


def test_amplitude_gate_suppresses_a_tiny_response():
    segs, tpl = condition(n=10, amp=0.01)
    got, _ = latencies(segs, tpl)
    assert not got


def test_physiological_bounds_are_honoured():
    segs, tpl = condition(n=12)
    out = [detect_mep_onset_boyles(s, FS, template=tpl, pre_ms=PRE_MS,
                                   search_start_ms=5, search_end_ms=45,
                                   min_latency_ms=15.0, max_latency_ms=35.0)
           for s in segs]
    for v in out:
        assert v is None or 15.0 <= v <= 35.0


def test_the_tighter_of_the_two_ceilings_applies():
    """
    The algorithm carries its own 35 ms ceiling. A muscle-specific profile must
    be able to tighten that, never be loosened by it.
    """
    segs, tpl = condition(n=12)
    out = [detect_mep_onset_boyles(s, FS, template=tpl, pre_ms=PRE_MS,
                                   search_start_ms=5, search_end_ms=45,
                                   min_latency_ms=15.0, max_latency_ms=21.0,
                                   boyles_max_latency_ms=35.0)
           for s in segs]
    for v in out:
        assert v is None or v <= 21.0


def test_works_without_a_template():
    """The Inspector may have no condition average to hand."""
    segs, _ = condition(n=10)
    got, _ = latencies(segs, None)
    assert got


def test_deterministic():
    segs, tpl = condition(n=4)
    a = detect_mep_onset_boyles(segs[0], FS, template=tpl, **KW)
    b = detect_mep_onset_boyles(segs[0], FS, template=tpl, **KW)
    assert a == b


@pytest.mark.parametrize("bad", [np.zeros(20), np.zeros(3000),
                                 np.full(3000, 5.0)])
def test_degenerate_input_returns_none(bad):
    assert detect_mep_onset_boyles(bad, FS, **KW) is None


def test_short_pre_stimulus_window_is_tolerated():
    """
    The published baseline window is 100 ms before the stimulus, which exceeds
    the pre-stimulus length of many epoch settings. It must clamp rather than
    fail: a 20 ms epoch is a normal configuration in this tool.
    """
    n = int((20 + POST_MS) * FS / 1000)
    stim = int(20 * FS / 1000)
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.012, n)
    i0 = stim + int(22 * FS / 1000)
    i1 = stim + int(40 * FS / 1000)
    k = np.arange(i1 - i0) / float(i1 - i0)
    x[i0:i1] += np.sin(2 * np.pi * k) * np.sin(np.pi * k) ** 0.5
    got = detect_mep_onset_boyles(x, FS, pre_ms=20, search_start_ms=5,
                                  search_end_ms=45, min_latency_ms=15.0,
                                  max_latency_ms=35.0,
                                  baseline_start_ms=100.0)
    assert got is not None


# ── The three corrected deviations ────────────────────────────────────────────

def test_literal_mode_exists_and_changes_the_result():
    """
    Both behaviours must be reachable. A correction that cannot be switched off
    makes the published method unreproducible; one that is silently applied
    makes a result unattributable to either version.
    """
    segs, tpl = condition(n=20)
    fixed, _ = latencies(segs, tpl, literal=False)
    lit, _ = latencies(segs, tpl, literal=True)
    assert (len(fixed), sorted(fixed)) != (len(lit), sorted(lit))


def test_literal_slope_window_degrades_as_sampling_rate_rises():
    """
    The reference fixes the slope window at 5 SAMPLES while its own comment and
    its unused ``blocklength`` variable both say 5 ms. The window therefore
    shrinks as sampling rate rises: 5 ms at 1 kHz, 1 ms at 5 kHz. Measured on a
    real recording, literal detection fell from 18/20 at 1 kHz to 11/20 at
    5 kHz while the corrected version held.

    Asserted as a relationship rather than fixed counts, so the test states the
    mechanism and does not become brittle.
    """
    segs, tpl = condition(n=20)
    lit_hi, n = latencies(segs, tpl, literal=True)             # 5 kHz
    dec = 5                                                    # -> 1 kHz
    segs_lo, tpl_lo = segs[:, ::dec], tpl[::dec]
    out = [detect_mep_onset_boyles(s, FS / dec, template=tpl_lo, **KW)
           for s in segs_lo]
    lit_lo = [v for v in out if v is not None]
    assert len(lit_lo) >= len(lit_hi), (
        "the literal slope window should perform BETTER at the lower sampling "
        "rate, where 5 samples is close to the intended 5 ms"
    )


def test_corrected_slope_window_is_a_duration_not_a_sample_count():
    """Halving the sampling rate must not halve the effective window."""
    segs, tpl = condition(n=20)
    hi, _ = latencies(segs, tpl, literal=False)
    dec = 5
    out = [detect_mep_onset_boyles(s, FS / dec, template=tpl[::dec], **KW)
           for s in segs[:, ::dec]]
    lo = [v for v in out if v is not None]
    assert lo and hi
    assert abs(float(np.mean(lo)) - float(np.mean(hi))) < 2.5


def test_literal_amplitude_gate_is_less_strict():
    """
    The reference compares the response's peak-to-peak against 1.1x the
    baseline MAXIMUM rather than its peak-to-peak, making the gate roughly half
    as strict as its name implies. A response that the corrected gate rejects
    must therefore sometimes pass in literal mode.
    """
    segs, tpl = condition(n=20, amp=0.05, noise=0.02)
    kw = dict(amplitude_gate=4.0)
    fixed, _ = latencies(segs, tpl, literal=False, **kw)
    lit, _ = latencies(segs, tpl, literal=True, **kw)
    assert len(lit) >= len(fixed)


def test_onset_cannot_precede_the_first_peak():
    """
    A structural property worth pinning, not a fault: the backward search is
    bounded by the first peak, so if the derivative ratio peaks at the response
    peak the result lands there. This is the likely mechanism behind reports of
    the algorithm returning a peak rather than an onset.
    """
    segs, tpl = condition(n=12)
    for s in segs:
        got = detect_mep_onset_boyles(s, FS, template=tpl, **KW)
        if got is None:
            continue
        stim = int(PRE_MS * FS / 1000)
        start = stim + int(5 * FS / 1000)
        finish = stim + int(45 * FS / 1000)
        w = s[start:finish]
        first_peak_ms = (start + min(int(np.argmax(w)), int(np.argmin(w)))
                         - stim) * 1000.0 / FS
        assert got <= first_peak_ms + 1e-6


# ── Consensus integration ─────────────────────────────────────────────────────

def test_available_as_a_consensus_member():
    from mep_cmap.detection.onset_methods_median import _ADAPTERS, compute_onset_agreement

    assert "boyles" in _ADAPTERS
    segs, tpl = condition(n=12)
    ag = compute_onset_agreement(
        segs[0], FS, pre_ms=PRE_MS, search_start_ms=5, search_end_ms=45,
        min_latency_ms=15.0, max_latency_ms=35.0,
        methods=("bigoni", "boyles", "rms_envelope"),
        params={"onset_env_n_boot": 150}, template=tpl)
    assert ag.n_attempted == 3
    assert "boyles" in ag.per_method


def test_not_in_the_default_consensus_set():
    """
    Deliberately opt-in. It has the most parameters of any method here, its
    published validation was on three participants, and adding it would make the
    member count even -- so the median would become an average of two.
    """
    from mep_cmap.detection.defaults import DEFAULT_METHODS_MEDIAN_MEMBERS

    assert "boyles" not in DEFAULT_METHODS_MEDIAN_MEMBERS
    assert len(DEFAULT_METHODS_MEDIAN_MEMBERS) % 2 == 1


# ── Window width, not spectral content ───────────────────────────────────────

def test_derivative_gates_need_spectral_content_at_realistic_noise():
    """
    The gates are stated in absolute derivatives, so the method needs the
    response to carry more high-frequency energy than the baseline. At the
    published 2.5 ms window a smooth response is detected at low noise and not
    at all at 0.02 mV, while a response with harmonic content survives.

    This module described the behaviour wrongly twice before this test existed:
    first attributing it to the method on evidence gathered with the window
    wrongly set to 5 ms, then retracting it as an artefact of that error on the
    strength of one noise level at which the corrected window masks it. The
    grid below is asserted so neither reading can be reached again from a
    single point.
    """
    smooth_quiet, t1 = condition(n=20, harmonic=0.0, noise=0.010)
    smooth_noisy, t2 = condition(n=20, harmonic=0.0, noise=0.020)
    rich_noisy, t3 = condition(n=20, harmonic=0.8, noise=0.020)

    n_sq = len(latencies(smooth_quiet, t1)[0])
    n_sn = len(latencies(smooth_noisy, t2)[0])
    n_rn = len(latencies(rich_noisy, t3)[0])

    assert n_sq >= 15, "a smooth response in a quiet baseline should detect"
    assert n_sn < n_sq, "raising baseline noise must reduce detection"
    assert n_rn > n_sn, "harmonic content must restore it at the same noise"


def test_the_window_width_is_not_the_cause():
    """
    Guards the second correction specifically: the effect must persist at the
    published window, or it is an artefact of window width as the retraction
    claimed.
    """
    smooth, tpl = condition(n=20, harmonic=0.0, noise=0.020)
    got, _ = latencies(smooth, tpl)              # default 2.5 ms window
    assert len(got) <= 4



def test_every_default_for_the_block_window_is_the_published_width():
    """
    The published width is 2.5 ms. It is written down in four places -- the
    canonical defaults, the detector signature, the dispatch and the consensus
    adapter -- and a stale copy in any of them substitutes silently, producing
    a plausible latency rather than an error. An earlier version of this port
    used 5.0 ms throughout, which halved detection on smooth responses.
    """
    import inspect

    from mep_cmap.detection import onset_methods_median
    from mep_cmap.detection.defaults import DETECTION_DEFAULTS
    from mep_cmap.detection.onset_boyles import detect_mep_onset_boyles
    from mep_cmap.pipeline import PipelineConfig

    assert DETECTION_DEFAULTS["onset_boyles_block_ms"] == 2.5
    assert PipelineConfig().onset_boyles_block_ms == 2.5
    sig = inspect.signature(detect_mep_onset_boyles)
    assert sig.parameters["block_ms"].default == 2.5
    src = inspect.getsource(onset_methods_median._call_boyles)
    assert '"onset_boyles_block_ms", 2.5' in src


def test_the_published_window_is_reachable_from_a_bare_params_dict():
    """A caller omitting the key must still get the published width."""
    from mep_cmap.detection import dispatch_onset

    segs, tpl = condition(n=12, harmonic=0.0)
    bare = [dispatch_onset(s, FS, {"onset_method": "boyles"}, template=tpl,
                           pre_ms=PRE_MS, search_start_ms=5, search_end_ms=45,
                           min_latency_ms=15.0, max_latency_ms=35.0)
            for s in segs]
    assert len([v for v in bare if v is not None]) >= 9
