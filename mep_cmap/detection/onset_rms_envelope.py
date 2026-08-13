"""
mep_cmap.detection.onset_rms_envelope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Onset detection by moving-window RMS envelope against an SD-scaled baseline
threshold, with optional TKEO preconditioning and short-window refinement.

This is the classical threshold-based detector of Hodges & Bui
(Electroencephalogr Clin Neurophysiol 1996;101:511-519), implemented with
three corrections to the usual form:

1.  The threshold is derived from the SD of the *envelope*, not of the raw
    rectified samples. See ``envelope_stats`` for why this matters.

2.  The minimum duration is calibrated against the chance distribution of
    above-threshold run lengths in the pre-stimulus envelope, so the
    false-positive rate is a stated significance level rather than a
    hard-coded sample count.

3.  The envelope crossing is treated as a coarse ANCHOR, not as the onset.
    A centred moving window of width W smears the transition symmetrically
    and crosses threshold roughly W/2 EARLY; a causal window crosses roughly
    W/2 LATE. Rather than subtracting a fixed offset — which assumes a
    step-like transition, and has the wrong sign depending on the window mode
    — the onset is RE-DETECTED on a much shorter envelope
    (``refine_window_ms``, default 1 ms) within a neighbourhood of the anchor.
    The coarse stage establishes that a response exists and where it roughly
    is; the fine stage sets the latency. Detected latency is then largely
    insensitive to the choice of W, which is the main criticism of this method
    class.

    This is a bidirectional local re-detection, not the one-directional
    walkback of ``onset_bigoni_walkback``. A backward-only walk cannot correct
    a centred-window anchor, because that anchor is already too early and a
    backward walk can only move it earlier still.

Known limitation
----------------
Every threshold-based method inherits its threshold from the pre-stimulus
baseline, so accuracy degrades when that baseline is not quiet and stationary.
In active-contraction paradigms the background EMG is large and varies
trial-to-trial, and a low-amplitude MEP may never exceed ``mu + k*sd``. For
those designs the derivative-based detectors (``onset_bigoni``) make no
baseline assumption and remain preferable. Enabling ``use_tkeo`` mitigates but
does not remove this limitation.

  * detect_mep_onset_rms_envelope
"""

import numpy as np

from .envelope_stats import (
    bootstrap_runlength_criterion,
    compute_envelope_baseline,
    compute_rms_envelope,
    find_sustained_run,
    passes_width_guard,
)
from .tkeo import apply_tkeo


def detect_mep_onset_rms_envelope(
        signal, fs, *,
        pre_ms=100,
        search_start_ms=5,
        search_end_ms=60,
        min_latency_ms=None,
        max_latency_ms=None,
        min_peak_amplitude=0.05,
        env_window_ms=5.0,
        criterion=2.5,
        significance=0.99,
        n_boot=500,
        min_run_ms=1.0,
        min_response_ms=3.0,
        use_tkeo=False,
        causal_window=False,
        refine_on_raw=True,
        refine_window_ms=1.0,
        refine_sd_mult=2.5,
        refine_sustain_ms=1.0,
        artefact_blank_ms=2.0,
        seed=42):
    """
    RMS-envelope onset detector with SD-scaled baseline threshold.

    Parameters
    ----------
    signal             : 1-D np.ndarray  EMG segment (pre-stim + post-stim)
    fs                 : float  sampling frequency in Hz
    pre_ms             : float  ms of pre-stimulus data in the segment
    search_start_ms    : float  ms post-stim to begin searching
    search_end_ms      : float  ms post-stim to stop searching
    min_latency_ms     : float or None  physiological floor (ms post-stim);
                         defaults to ``artefact_blank_ms``
    max_latency_ms     : float or None  physiological ceiling (ms post-stim);
                         defaults to ``search_end_ms``
    min_peak_amplitude : float  amplitude gate on the raw signal, in mV
    env_window_ms      : float  RMS window width (default 5.0)
    criterion          : float  SD multiplier for the threshold (default 2.5)
    significance       : float  percentile for the run-length criterion
    n_boot             : int    bootstrap resamples for the run-length criterion
    min_run_ms         : float  hard floor on the required above-threshold
                         duration (default 1.0). The bootstrap criterion can
                         legitimately fall below a physiologically meaningful
                         duration on a very clean baseline; this floor stops a
                         two-sample noise blip qualifying as a response.
    min_response_ms    : float  width guard (default 3.0). The response must be
                         continuously elevated for this long on the short
                         envelope, which rejects single-sample artefacts that
                         window smearing would otherwise widen into an
                         apparent response. Set to 0 to disable.
    use_tkeo           : bool   apply the Teager-Kaiser operator first. The
                         threshold is then derived from the TKEO baseline, so
                         ``criterion`` keeps its meaning but ``min_peak_amplitude``
                         still gates on the raw signal in mV.
    causal_window      : bool   causal rather than centred RMS window
    refine_on_raw      : bool   refine the coarse anchor (default True)
    refine_window_ms   : float  short envelope width used for refinement
    refine_sd_mult     : float  SD multiplier for the refinement baseline
                         (default 2.5). Should normally track ``criterion``:
                         the refinement is a detection on a shorter window,
                         not a walk through known-elevated signal, so a lax
                         value lets baseline noise satisfy it. At 1.0 roughly
                         one baseline sample in six exceeds the threshold,
                         which in testing misrouted the refinement and
                         reintroduced several ms of early bias at wide W.
    refine_sustain_ms  : float  sustained return-to-baseline required to stop
                         the walkback
    artefact_blank_ms  : float  hard floor — onset never placed before this
    seed               : int    RNG seed; detection is reproducible

    Returns
    -------
    latency_ms : float, or None if no confident onset was found
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    if n < 16 or fs <= 0:
        return None

    ms_per_samp = 1000.0 / fs
    stim_idx = int(pre_ms * fs / 1000.0)
    if stim_idx < 5 or stim_idx >= n:
        return None

    # ── Physiological bounds ─────────────────────────────────────────────────
    _min_lat = artefact_blank_ms if min_latency_ms is None else min_latency_ms
    _max_lat = search_end_ms if max_latency_ms is None else max_latency_ms
    _min_lat = max(float(_min_lat), float(artefact_blank_ms))
    if _max_lat <= _min_lat:
        return None

    win_start = int((pre_ms + search_start_ms) * fs / 1000.0)
    win_end = min(int((pre_ms + search_end_ms) * fs / 1000.0), n)
    if win_start >= win_end:
        return None

    # ── Amplitude gate (always on the raw signal, in mV) ─────────────────────
    raw_window = signal[win_start:win_end]
    if raw_window.size == 0:
        return None
    if float(raw_window.max() - raw_window.min()) < min_peak_amplitude:
        return None

    # ── Detector signal and envelope ─────────────────────────────────────────
    det = apply_tkeo(signal) if use_tkeo else signal
    env = compute_rms_envelope(det, fs, window_ms=env_window_ms,
                               causal=causal_window)

    # Trim the baseline by the window half-width so no baseline sample is
    # contaminated by the stimulus artefact bleeding backward through a
    # centred window.
    env_win = max(1, int(round(env_window_ms * fs / 1000.0)))
    guard = 0 if causal_window else (env_win // 2 + 1)
    pre_hi = max(0, stim_idx - guard)
    env_pre = env[:pre_hi]

    base = compute_envelope_baseline(env_pre, criterion=criterion)
    if base is None:
        return None

    # Block length must match the envelope width, or the chance run-length
    # distribution collapses to ~1 sample and noise passes as signal.
    min_run_samples = max(2, int(round(min_run_ms * fs / 1000.0)))
    crit_samples = bootstrap_runlength_criterion(
        env_pre, criterion=criterion, significance=significance,
        n_boot=n_boot, seed=seed, tail="upper",
        min_samples=min_run_samples, block_samples=env_win)

    # ── Coarse anchor: first sustained excursion above threshold ─────────────
    scan_lo = max(win_start, int((pre_ms + _min_lat) * fs / 1000.0))
    scan_hi = min(win_end, int((pre_ms + _max_lat) * fs / 1000.0) + crit_samples)
    anchor = find_sustained_run(env, base.threshold, crit_samples,
                                lo=scan_lo, hi=scan_hi, above=True)
    if anchor is None:
        return None

    onset_idx = anchor

    # ── Refinement on a short envelope ───────────────────────────────────────
    if refine_on_raw:
        refined = _refine_anchor(
            signal, fs, anchor_idx=anchor, floor_idx=scan_lo,
            ceiling_idx=win_end, stim_idx=stim_idx,
            search_radius=env_win,
            refine_window_ms=refine_window_ms,
            refine_sd_mult=refine_sd_mult,
            refine_sustain_ms=refine_sustain_ms)
        if refined is not None:
            onset_idx = refined

    onset_idx = max(onset_idx, scan_lo)

    # ── Width guard against single-sample artefacts ──────────────────────────
    if min_response_ms and min_response_ms > 0:
        fine = compute_rms_envelope(signal, fs, window_ms=refine_window_ms,
                                    causal=False)
        fine_win = max(1, int(round(refine_window_ms * fs / 1000.0)))
        fine_base = compute_envelope_baseline(
            fine[:max(0, stim_idx - (fine_win // 2 + 1))],
            criterion=refine_sd_mult)
        if fine_base is not None:
            need = max(1, int(round(min_response_ms * fs / 1000.0)))
            if not passes_width_guard(fine, fine_base.threshold,
                                      onset_idx, need):
                return None

    latency_ms = (onset_idx - stim_idx) * ms_per_samp

    if latency_ms < _min_lat or latency_ms > _max_lat:
        return None
    return round(float(latency_ms), 2)


def _refine_anchor(signal, fs, *, anchor_idx, floor_idx, ceiling_idx, stim_idx,
                   search_radius, refine_window_ms, refine_sd_mult,
                   refine_sustain_ms):
    """
    Re-detect the onset on a short envelope within a neighbourhood of the anchor.

    The short window adds only ~refine_window_ms/2 of smearing while remaining
    robust to the zero crossings of raw rectified EMG, which defeat any test
    applied to individual raw samples.

    The search spans ``anchor_idx +/- search_radius``, so the correction works
    in either direction: a centred coarse window (anchor early) is moved later,
    a causal one (anchor late) is moved earlier. ``refine_sustain_ms`` guards
    against a noise blip inside the neighbourhood capturing the onset.

    Returns an index into ``signal``, or None if refinement is not possible.
    """
    fine = compute_rms_envelope(signal, fs, window_ms=refine_window_ms,
                               causal=False)
    fine_win = max(1, int(round(refine_window_ms * fs / 1000.0)))
    pre_hi = max(0, stim_idx - (fine_win // 2 + 1))
    fine_base = compute_envelope_baseline(fine[:pre_hi],
                                         criterion=refine_sd_mult)
    if fine_base is None:
        return None

    sustain = max(1, int(round(refine_sustain_ms * fs / 1000.0)))
    lo = max(int(floor_idx), int(anchor_idx) - int(search_radius))
    hi = min(int(ceiling_idx), int(anchor_idx) + int(search_radius) + 1)
    anchor_idx = int(anchor_idx)
    if lo >= hi or anchor_idx >= fine.size:
        return None

    thr = fine_base.threshold

    # Locate the onset of the excursion CONTAINING the anchor, not merely the
    # first excursion in the neighbourhood. Searching the neighbourhood
    # indiscriminately lets an unrelated noise blip a few ms earlier capture
    # the onset, which in testing produced occasional 7 ms early outliers and
    # tripled the latency SD.
    # Test the anchor against a short stretch rather than a single sample: one
    # sample of a 1 ms envelope is far too noisy to decide which branch to take.
    look = fine[anchor_idx:min(anchor_idx + sustain, fine.size)]
    anchor_is_elevated = look.size > 0 and float(np.mean(look > thr)) > 0.5

    if anchor_is_elevated:
        # Anchor already sits inside the response (typical of a causal coarse
        # window, or a coarse window narrow enough not to lead the signal).
        # Walk back to where the elevated region began, stopping once the
        # envelope has been quiet for `sustain` samples.
        quiet = 0
        for i in range(anchor_idx, lo - 1, -1):
            if fine[i] <= thr:
                quiet += 1
                if quiet >= sustain:
                    return min(i + sustain, anchor_idx)
            else:
                quiet = 0
        return lo
    # Anchor precedes the response — the usual case for a centred coarse
    # window, which leads the signal by roughly half its width. Advance to the
    # first sustained excursion at or after the anchor.
    return find_sustained_run(fine, thr, sustain,
                              lo=anchor_idx, hi=hi, above=True)
