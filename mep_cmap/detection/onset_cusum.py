"""
mep_cmap.detection.onset_cusum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Onset detection by cumulative sum (CUSUM) change-point analysis.

CUSUM accumulates the running excess of the signal over its baseline mean,
resetting to zero whenever the accumulation goes negative:

    S[n] = max(0, S[n-1] + (v[n] - mu0) - k)

An alarm is raised when S exceeds a decision interval h. Both k (the allowance
or reference value) and h are expressed in baseline SD units, so the detector
transfers across recordings, muscles and amplifier gains without retuning.

Why this differs from a threshold detector in a way that matters
---------------------------------------------------------------
A threshold detector reports the time at which the signal became large. CUSUM
reports the time at which the signal's MEAN CHANGED, which is the quantity an
onset latency is supposed to measure. The change point is estimated as the last
sample at which S was zero before the alarm — the accumulator only started
growing once the mean shifted, so this estimate does not inherit the delay
between the true change and the moment enough evidence had accrued to declare
it. Threshold and envelope methods have no equivalent property and need
explicit bias correction.

h also has a direct interpretation: increasing it lengthens the average run
length to false alarm, so it is a principled false-positive control rather
than a tuning constant.

Formulation notes
-----------------
The test statistic is the rectified signal (or the TKEO, which is already an
energy). Rectification makes the detector one-sided by construction: an MEP of
either polarity, and a biphasic MEP of either leading polarity, all produce an
increase in |x|. A two-sided CUSUM on the raw signal would be sensitive to
which way the first deflection happened to go, so only the upper arm is used.

  * detect_mep_onset_cusum
"""

import numpy as np

from .envelope_stats import (
    compute_envelope_baseline,
    compute_rms_envelope,
    passes_width_guard,
)
from .tkeo import apply_tkeo

_EPS = 1e-12


def detect_mep_onset_cusum(
        signal, fs, *,
        pre_ms=100,
        search_start_ms=5,
        search_end_ms=60,
        min_latency_ms=None,
        max_latency_ms=None,
        min_peak_amplitude=0.05,
        k_mult=0.5,
        h_mult=20.0,
        max_accum_ms=10.0,
        use_tkeo=False,
        smooth_ms=0.0,
        min_response_ms=3.0,
        guard_window_ms=1.0,
        guard_sd_mult=2.5,
        artefact_blank_ms=2.0):
    """
    CUSUM change-point onset detector.

    Parameters
    ----------
    signal             : 1-D np.ndarray  EMG segment (pre-stim + post-stim)
    fs                 : float  sampling frequency in Hz
    pre_ms             : float  ms of pre-stimulus data in the segment
    search_start_ms    : float  ms post-stim to begin accumulating
    search_end_ms      : float  ms post-stim to stop accumulating
    min_latency_ms     : float or None  physiological floor (ms post-stim)
    max_latency_ms      : float or None  physiological ceiling (ms post-stim)
    min_peak_amplitude : float  amplitude gate on the raw signal, in mV
    k_mult             : float  allowance in baseline SD units (default 0.5).
                         Shifts smaller than k_mult SD are not accumulated;
                         0.5 detects a shift of ~1 SD most efficiently.
    h_mult             : float  decision interval in baseline SD units
                         (default 20.0). Raise to reduce false alarms, lower to
                         detect smaller responses. Swept on synthetic data: at
                         h=5 the false-positive rate on pure noise was 26.5%,
                         at h=10 it was 0.5%, and from h=20 upward it was 0%
                         with no loss of sensitivity even for a 0.15 mV
                         response in 0.03 mV noise. 20 therefore buys a wide
                         safety margin at no measured cost.
    use_tkeo           : bool   accumulate the TKEO rather than |signal|
    smooth_ms          : float  optional moving-average pre-smoothing of the
                         test statistic in ms; 0 disables. CUSUM already
                         integrates, so smoothing is usually unnecessary.
    max_accum_ms       : float  windowed-CUSUM reset interval (default 10.0).
                         If the accumulator has been positive for this long
                         without breaching h, it is reset. Without this, a
                         favourable run of baseline noise can keep the
                         accumulator marginally positive for many
                         milliseconds, so the last-zero backtrack anchors the
                         change point well before the response; this placed 2
                         of 30 clean synthetic trials about 5 ms early. A real
                         evoked response drives the accumulator past h within
                         a few ms, so a bounded window costs no sensitivity.
                         Set to 0 to disable (classical unbounded CUSUM).
    min_response_ms    : float  width guard (default 3.0); see
                         ``envelope_stats.passes_width_guard``. CUSUM
                         integrates, so a single large artefact sample can
                         breach the decision interval on its own. Set to 0 to
                         disable.
    guard_window_ms    : float  short envelope width used by the width guard
    guard_sd_mult      : float  SD multiplier used by the width guard
    artefact_blank_ms  : float  hard floor — accumulation never starts before
                         this, so the stimulus artefact cannot trigger the
                         alarm

    Returns
    -------
    latency_ms : float, or None if no change point was detected
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    if n < 16 or fs <= 0:
        return None

    ms_per_samp = 1000.0 / fs
    stim_idx = int(pre_ms * fs / 1000.0)
    if stim_idx < 5 or stim_idx >= n:
        return None

    _min_lat = artefact_blank_ms if min_latency_ms is None else min_latency_ms
    _max_lat = search_end_ms if max_latency_ms is None else max_latency_ms
    _min_lat = max(float(_min_lat), float(artefact_blank_ms))
    if _max_lat <= _min_lat:
        return None

    win_start = int((pre_ms + search_start_ms) * fs / 1000.0)
    win_end = min(int((pre_ms + search_end_ms) * fs / 1000.0), n)
    if win_start >= win_end:
        return None

    # ── Amplitude gate (raw signal, mV) ──────────────────────────────────────
    raw_window = signal[win_start:win_end]
    if raw_window.size == 0:
        return None
    if float(raw_window.max() - raw_window.min()) < min_peak_amplitude:
        return None

    # ── Test statistic ───────────────────────────────────────────────────────
    stat = apply_tkeo(signal) if use_tkeo else np.abs(signal)
    if smooth_ms and smooth_ms > 0:
        w = max(1, int(round(smooth_ms * fs / 1000.0)))
        if w > 1:
            pad_l = (w - 1) // 2
            pad_r = w - 1 - pad_l
            stat = np.convolve(np.pad(stat, (pad_l, pad_r), mode="edge"),
                               np.ones(w) / float(w), mode="valid")

    # ── Baseline parameters ──────────────────────────────────────────────────
    pre = stat[:stim_idx]
    if pre.size < 5:
        return None
    mu0 = float(pre.mean())
    sd0 = float(pre.std(ddof=1)) if pre.size > 1 else 0.0
    if not np.isfinite(mu0) or not np.isfinite(sd0):
        return None
    sd0 = max(sd0, abs(mu0) * 1e-3, _EPS)

    k = k_mult * sd0
    h = h_mult * sd0

    # ── Accumulate ───────────────────────────────────────────────────────────
    scan_lo = max(win_start, int((pre_ms + _min_lat) * fs / 1000.0))
    scan_hi = min(win_end, int((pre_ms + _max_lat) * fs / 1000.0))
    # Allow the alarm itself to occur slightly past the latency ceiling: the
    # change point is what must lie inside the physiological window, and the
    # evidence for it necessarily accrues afterwards.
    alarm_hi = min(win_end, scan_hi + int(round(10.0 * fs / 1000.0)))
    if scan_lo >= scan_hi:
        return None

    max_accum = (int(round(max_accum_ms * fs / 1000.0))
                 if max_accum_ms and max_accum_ms > 0 else 0)

    s = 0.0
    last_zero = scan_lo
    change_idx = None
    for i in range(scan_lo, alarm_hi):
        s = s + (stat[i] - mu0) - k
        if s <= 0.0:
            s = 0.0
            last_zero = i
        elif s > h:
            change_idx = last_zero
            break
        elif max_accum and (i - last_zero) > max_accum:
            # Windowed reset: evidence this slow cannot be an evoked response.
            s = 0.0
            last_zero = i

    if change_idx is None:
        return None

    # The change point is the first sample after the accumulator last sat at
    # zero, bounded by the physiological floor.
    onset_idx = max(int(change_idx) + 1, scan_lo)

    # ── Width guard against single-sample artefacts ──────────────────────────
    if min_response_ms and min_response_ms > 0:
        fine = compute_rms_envelope(signal, fs, window_ms=guard_window_ms,
                                    causal=False)
        gw = max(1, int(round(guard_window_ms * fs / 1000.0)))
        gbase = compute_envelope_baseline(
            fine[:max(0, stim_idx - (gw // 2 + 1))], criterion=guard_sd_mult)
        if gbase is not None:
            need = max(1, int(round(min_response_ms * fs / 1000.0)))
            if not passes_width_guard(fine, gbase.threshold, onset_idx, need):
                return None

    latency_ms = (onset_idx - stim_idx) * ms_per_samp

    if latency_ms < _min_lat or latency_ms > _max_lat:
        return None
    return round(float(latency_ms), 2)
