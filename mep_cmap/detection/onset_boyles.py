"""
mep_cmap.detection.onset_boyles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Derivative-ratio onset detection, after Boyles et al. (2026).

Reference: https://doi.org/10.1038/s41598-026-61560-0
Ported from the MATLAB implementation in the TMSMultiLab library
(``Analysis/Matlab/MEP_onset_Boyles_2026.m``, N. Holmes), which is the
authoritative statement of the algorithm this module follows.

How it works
------------
Walking backwards from the first peak of the response, each candidate sample is
scored by the ratio of the mean absolute derivative AHEAD of it to the mean
absolute derivative BEHIND it. The onset is the earliest sample whose ratio is
still at least ``ratio_cutoff`` of the maximum ratio found. Three gates then
apply: a latency ceiling, a requirement that most of the first few forward
derivatives exceed the baseline mean derivative, and a requirement that the mean
derivative over a following window exceeds baseline mean + k SD.

This is methodologically distinct from every other detector here. It is not a
threshold crossing (peak_fraction, bootstrap, rms_envelope), not a run length in
the derivative (bigoni), and not a cumulative change point (cusum) -- it is a
local before/after contrast in slope. That independence is its main value as a
member method.

It needs a TEMPLATE
-------------------
The published algorithm takes a grand-mean waveform across epochs of the same
muscle, participant and condition, and uses its first peak to reject trials
whose peak falls too far from the expected latency. ``template`` supplies it;
the pipeline passes the per-stimulus-type median waveform it already computes
for onset anchoring. With no template the peak-jitter gate is skipped and
everything else proceeds, so the detector remains usable from the Inspector
where a condition average may not be to hand.

Deviations from the reference implementation
-------------------------------------------
Three details of the MATLAB port do not behave as its own comments describe.
They are corrected here by default, and ``literal=True`` restores the reference
behaviour exactly for anyone reproducing the paper. Every deviation is a
correction of an evident slip, not a reinterpretation of the method:

1.  **The slope window is fixed in SAMPLES, not milliseconds.** The paper gives
    it both ways: "B : block length (5 samples - 2.5 ms given a sampling rate of
    2 kHz)", while the Figure 1 legend describes "a 2.5 ms window before the
    point and after the point". Those agree only at the 2 kHz used in that
    study. The reference implementation indexes with ``options.blocklength``,
    whose default is the literal ``5``, so the window is 5 SAMPLES however the
    data were sampled -- 2.5 ms at 2 kHz, 1 ms at 5 kHz, 0.5 ms at 10 kHz.
    (Its own ``blocklength = round(5*(samplehz/1000))`` line, which is computed
    and never used, would give 5 ms, twice the published width.) Here the
    window is a duration, defaulting to the published 2.5 ms.

    The paper's own Limitations section anticipates this: it notes the block
    length is a heuristic that risks being "partly tuned to the acquisition
    characteristics ... including sampling rate". Its sensitivity analysis also
    upsampled 2 kHz data to 5 kHz specifically because the comparator method is
    sampling-rate sensitive, without applying the same reasoning to this
    parameter.

2.  **The amplitude gate compared a peak-to-peak against a single peak.** The
    reference sets ``base.p2p = max(data(base.on:base.off))`` -- a maximum, with
    no ``- min(...)`` -- and compares the response's true peak-to-peak against
    ``1.1 x`` that. Since the data are baseline-subtracted, the gate is roughly
    half as strict as its name and comment imply. Here both sides are
    peak-to-peak.

3.  **The peak-jitter gate compared two different quantities.** The reference
    takes the trial's LARGEST absolute peak but the template's FIRST peak. On a
    biphasic response whose second phase is larger, those differ by the
    peak-to-trough interval on every trial, so the tolerance is spent on a
    definitional mismatch rather than on jitter. Here both sides use the first
    peak.

It needs the response to be spectrally richer than the baseline
--------------------------------------------------------------
Every gate is stated in ABSOLUTE derivatives -- the ratio itself, the count
against the baseline mean derivative, and the comparison against baseline mean
plus k SD. What the method therefore requires is that the response carry more
high-frequency energy than the baseline, not merely more amplitude. Detection
collapses when it does not, and it fails by returning None rather than by
returning a wrong latency.

Measured at the published 2.5 ms window, so this is not an artefact of window
width (detection out of 20 synthetic trials, response 1 mV):

    baseline noise    smooth    some harmonics    rich
    0.006 mV            20            16           18
    0.012 mV            19            16           17
    0.020 mV             0             9           16
    0.030 mV             0             0           14

The RMS envelope detector scores 20 of 20 across that entire grid, so this is
specific to the derivative-based gates rather than a general SNR limit.

This module twice described that behaviour wrongly before arriving here. It was
first attributed to the method as such, on evidence gathered with the window
mistakenly set to 5 ms; then retracted as an artefact of that error, on the
basis of a single noise level (0.012 mV) at which the corrected window happens
to mask it. Both readings were reached from too narrow a slice of the parameter
space. The effect is real, and it is governed by baseline noise and spectral
content together.

In practice, surface EMG from a quiet resting recording sits in the top rows of
that grid -- on a real recording with a 0.0034 mV pre-stimulus RMS this detector
found 17 to 18 of 20 trials per condition. Aggressive low-pass filtering, or a
noisy baseline, moves a dataset toward the bottom rows.

A note on robustness
--------------------
This detector has more interacting parameters than any other here, two of them
scaled by the trial's own peak-to-trough interval, and the published validation
was on three participants. The backward search is also bounded by the first
peak, so the returned onset can never precede it: if the derivative ratio peaks
at the response peak, the result lands on the peak. Treat it as a consensus
member and a comparison point rather than a default, and read
``Onset_Disagreement(ms)`` when it is in the set.

  * detect_mep_onset_boyles
"""

import numpy as np

_EPS_DEFAULT = 1e-6


def detect_mep_onset_boyles(
        signal, fs, *,
        pre_ms=100,
        search_start_ms=5.0,
        search_end_ms=45.0,
        min_latency_ms=None,
        max_latency_ms=None,
        min_peak_amplitude=0.05,
        template=None,
        block_ms=2.5,
        baseline_start_ms=100.0,
        baseline_end_ms=1.0,
        amplitude_gate=1.1,
        peak_jitter_ms=15.0,
        peak_window_length=1.75,
        ratio_cutoff=0.85,
        boyles_max_latency_ms=35.0,
        deriv_check_ms=2.0,
        deriv_check_duty=0.75,
        base_deriv_sds=1.5,
        deriv_check_window_length=2.0,
        ratio_epsilon=_EPS_DEFAULT,
        literal=False):
    """
    Derivative-ratio onset detector (Boyles et al. 2026).

    Parameters
    ----------
    signal              : 1-D np.ndarray  EMG segment (pre-stim + post-stim)
    fs                  : float  sampling frequency in Hz
    pre_ms              : float  ms of pre-stimulus data in ``signal``. Must be
                          the pre-stimulus length of the array passed, not a
                          nominal baseline setting.
    search_start_ms     : float  ms post-stim to begin searching (paper: 5)
    search_end_ms       : float  ms post-stim to stop searching (paper: 45)
    min_latency_ms      : float or None  physiological floor from the latency
                          profile
    max_latency_ms      : float or None  physiological ceiling from the profile.
                          The TIGHTER of this and ``boyles_max_latency_ms``
                          applies, so a muscle-specific profile is never
                          loosened by the algorithm's own default.
    min_peak_amplitude  : float  shared amplitude gate, in mV. Applied in
                          addition to the algorithm's relative gate so this
                          detector honours the same setting as the others.
    template            : 1-D np.ndarray or None  condition average/median,
                          same length as ``signal``. Enables the peak-jitter
                          gate; skipped when None.
    block_ms            : float  slope comparison window either side of a
                          candidate. Default 2.5 ms, the width stated in the
                          paper; see the module notes on sample-vs-duration.
    baseline_start_ms   : float  baseline window start, ms BEFORE the stimulus
                          (paper: 100). Clamped to the pre-stimulus data
                          actually present, so a 20 ms epoch still works.
    baseline_end_ms     : float  baseline window end, ms before the stimulus
    amplitude_gate      : float  required ratio of response to baseline
                          peak-to-peak (paper: 1.1)
    peak_jitter_ms      : float  maximum peak displacement from the template
                          (paper: 15)
    peak_window_length  : float  multiplies the peak-to-trough interval to set
                          how far back the search may run (paper: 1.75)
    ratio_cutoff        : float  fraction of the maximum derivative ratio a
                          candidate must reach (paper: 0.85)
    boyles_max_latency_ms : float  the algorithm's own latency ceiling
                          (paper: 35)
    deriv_check_ms      : float  window just after the candidate in which
                          forward derivatives must exceed the baseline mean
    deriv_check_duty    : float  fraction of that window that must exceed it
                          (paper: 3 of the first 4 samples, i.e. 0.75)
    base_deriv_sds      : float  SD multiplier for the overall-slope gate
                          (paper: 1.5)
    deriv_check_window_length : float  multiplies the peak-to-trough interval to
                          set the overall-slope window (paper: 2)
    ratio_epsilon       : float  added to the denominator to avoid division by
                          zero
    literal             : bool  reproduce the reference implementation exactly,
                          including the three slips described in the module
                          docstring. For reproducing the published method;
                          not recommended for new analyses.

    Returns
    -------
    latency_ms : float, or None if no confident onset was found
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    if n < 32 or fs <= 0:
        return None

    ms = 1000.0 / fs
    stim = int(round(pre_ms * fs / 1000.0))
    if stim < 5 or stim >= n:
        return None

    # ── Windows ──────────────────────────────────────────────────────────────
    start = stim + int(round(search_start_ms * fs / 1000.0))
    finish = min(stim + int(round(search_end_ms * fs / 1000.0)), n)
    if start >= finish - 2:
        return None

    # Baseline, clamped to the pre-stimulus data present. The paper's 100 ms
    # window exceeds the pre-stimulus length of many epoch settings; clamping
    # degrades the estimate rather than failing outright.
    b_lo = max(0, stim - int(round(baseline_start_ms * fs / 1000.0)))
    b_hi = max(b_lo + 4, stim - int(round(baseline_end_ms * fs / 1000.0)))
    b_hi = min(b_hi, stim)
    if b_hi - b_lo < 4:
        return None

    base = signal[b_lo:b_hi]
    data = signal - float(np.mean(base))
    base = data[b_lo:b_hi]

    # ── Shared amplitude gate ────────────────────────────────────────────────
    win = data[start:finish]
    if win.size == 0:
        return None
    p2p = float(win.max() - win.min())
    if p2p < min_peak_amplitude:
        return None

    # ── Peaks ────────────────────────────────────────────────────────────────
    i_max = int(np.argmax(win))
    i_min = int(np.argmin(win))
    anchor = start + min(i_max, i_min)          # first peak, either polarity
    delta_p2p = abs(i_max - i_min)
    if delta_p2p < 2:
        return None

    # ── Relative amplitude gate ──────────────────────────────────────────────
    if literal:
        base_amp = float(base.max())            # reference: max, not p2p
    else:
        base_amp = float(base.max() - base.min())
    if not (p2p > amplitude_gate * base_amp):
        return None

    # ── Peak-jitter gate (needs a template) ──────────────────────────────────
    if template is not None:
        tpl = np.asarray(template, dtype=float)
        if tpl.size == n:
            tpl = tpl - float(np.mean(tpl[b_lo:b_hi]))
            tw = tpl[start:finish]
            if tw.size:
                tpl_anchor = start + min(int(np.argmax(tw)), int(np.argmin(tw)))
                if literal:
                    # Reference: the trial's LARGEST peak against the
                    # template's FIRST peak.
                    trial_ref = start + int(np.argmax(np.abs(win)))
                else:
                    trial_ref = anchor
                if abs(trial_ref - tpl_anchor) > \
                        int(round(peak_jitter_ms * fs / 1000.0)):
                    return None

    # ── Baseline derivative statistics ───────────────────────────────────────
    bd = np.abs(np.diff(base))
    if bd.size < 2:
        return None
    bd_mean = float(bd.mean())
    bd_sd = float(bd.std(ddof=1))
    bd_cutoff = bd_mean + base_deriv_sds * bd_sd

    # ── Candidate window and window widths ───────────────────────────────────
    k_min = int(max(anchor - peak_window_length * delta_p2p, start))
    if k_min >= anchor:
        return None

    block = 5 if literal else max(2, int(round(block_ms * fs / 1000.0)))
    check = 4 if literal else max(2, int(round(deriv_check_ms * fs / 1000.0)))
    long_win = max(4, int(round(deriv_check_window_length * delta_p2p)))

    lo_needed = k_min - block - 1
    hi_needed = anchor + max(block, check, long_win) + 1
    if lo_needed < 0 or hi_needed >= n:
        # Trim the candidate range rather than refusing outright.
        k_min = max(k_min, block + 1)
        anchor = min(anchor, n - max(block, check, long_win) - 2)
        if k_min >= anchor:
            return None

    # ── Derivative ratio at each candidate ───────────────────────────────────
    idx = np.arange(k_min, anchor + 1)
    ratio = np.full(n, np.nan)
    n_above = np.zeros(n)
    long_mean = np.zeros(n)
    for l in idx:
        fwd = np.abs(np.diff(data[l + 1:l + 1 + block]))
        bwd = np.abs(np.diff(data[l - block:l]))
        if fwd.size == 0 or bwd.size == 0:
            continue
        ratio[l] = fwd.mean() / (bwd.mean() + ratio_epsilon)
        chk = np.abs(np.diff(data[l + 1:l + 1 + check]))
        n_above[l] = float(np.mean(chk > bd_mean)) if chk.size else 0.0
        lng = np.abs(np.diff(data[l + 1:l + 1 + long_win]))
        long_mean[l] = float(lng.mean()) if lng.size else 0.0

    if not np.any(np.isfinite(ratio)):
        return None
    r_max = float(np.nanmax(ratio))
    r_ix = int(np.nanargmax(ratio))
    if not np.isfinite(r_max) or r_max <= 0:
        return None

    # Earliest sample, at or before the peak ratio, that still reaches the
    # cutoff: the last upward crossing of the cutoff before the peak.
    mask = np.nan_to_num(ratio, nan=-np.inf) > (ratio_cutoff * r_max)
    trans = np.diff(mask.astype(np.int8))
    ups = np.flatnonzero(trans[:r_ix] == 1)
    if ups.size == 0:
        onset_idx = r_ix
    else:
        onset_idx = int(ups[-1]) + 1

    if not np.isfinite(ratio[onset_idx]):
        return None

    # ── Gates ────────────────────────────────────────────────────────────────
    # Initial slope: most of the first few forward derivatives must exceed the
    # baseline mean derivative.
    if n_above[onset_idx] < deriv_check_duty:
        return None
    # Overall slope: the following window must be clearly steeper than baseline.
    if long_mean[onset_idx] < bd_cutoff:
        return None

    latency = (onset_idx - stim) * ms

    # The tighter of the profile ceiling and the algorithm's own applies.
    ceiling = float(boyles_max_latency_ms)
    if max_latency_ms is not None:
        ceiling = min(ceiling, float(max_latency_ms))
    if latency > ceiling:
        return None
    if min_latency_ms is not None and latency < float(min_latency_ms):
        return None
    if latency <= 0:
        return None

    return round(float(latency), 2)
