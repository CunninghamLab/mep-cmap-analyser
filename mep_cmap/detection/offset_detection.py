"""
mep_cmap.detection.offset_detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MEP/CMAP offset detection — the return of the evoked response to baseline.

Offset completes the response window and yields MEP duration, a measure that
is routinely neglected despite being informative about temporal dispersion and
polyphasia. It also gives area-under-curve a principled right-hand bound in
resting-state recordings, where there is no cortical silent period to end the
integration window and the endpoint has otherwise had to be dragged by hand.

The definitional problem
------------------------
"MEP offset" does not mean one thing across paradigms:

  * At REST the response returns to a quiet baseline, and offset is the mirror
    image of onset — the point at which the envelope falls back below the
    baseline threshold and stays there.

  * During VOLUNTARY CONTRACTION the response is followed by the cortical
    silent period. The end of the MEP and the start of the silent period are
    the same physical event. Detecting them independently would produce two
    columns that disagree by a few milliseconds for no physiological reason,
    and would invite readers to treat the difference as meaningful.

``resolve_mep_offset`` therefore applies one explicit precedence rule rather
than letting each caller improvise, and records which branch fired in an
accompanying source field so the provenance of every value is recoverable
from the output alone:

    manual override        -> source 'manual'
    cSP enabled + detected -> source 'csp_start'
    otherwise              -> source 'envelope'   (detector below)
    nothing usable         -> source 'none'

  * detect_mep_offset    -- envelope return-to-baseline detector
  * resolve_mep_offset   -- precedence rule + duration
  * OFFSET_SOURCES
"""

from collections import namedtuple

import numpy as np

from .envelope_stats import (
    bootstrap_runlength_criterion,
    compute_envelope_baseline,
    compute_rms_envelope,
    find_sustained_run,
)
from .tkeo import apply_tkeo

OFFSET_SOURCES = ("manual", "csp_start", "envelope", "none")

MepOffset = namedtuple("MepOffset", "offset_ms source duration_ms")


def detect_mep_offset(
        signal, fs, *,
        onset_ms,
        pre_ms=100,
        search_end_ms=100,
        min_duration_ms=5.0,
        max_duration_ms=None,
        env_window_ms=5.0,
        criterion=2.5,
        significance=0.99,
        n_boot=500,
        min_return_ms=10.0,
        peak_frac=0.12,
        use_tkeo=False,
        refine_forward=True,
        refine_window_ms=1.0,
        refine_sd_mult=1.0,
        refine_sustain_ms=1.0,
        seed=42):
    """
    Envelope return-to-baseline offset detector.

    Scans forward from the dominant peak of the response for the first point
    at which the RMS envelope drops below the baseline threshold and REMAINS
    below it for ``min_return_ms``. The sustained-return requirement is what
    prevents the scan from stopping in the trough between the phases of a
    biphasic or polyphasic MEP, where the envelope dips briefly but the
    response is plainly still in progress.

    Parameters
    ----------
    signal          : 1-D np.ndarray  EMG segment (pre-stim + post-stim)
    fs              : float  sampling frequency in Hz
    onset_ms        : float  detected onset (ms post-stim). Required — offset
                      is meaningless without it. Pass None to get None back.
    pre_ms          : float  ms of pre-stimulus data in the segment
    search_end_ms   : float  ms post-stim to stop searching
    min_duration_ms : float  offset must be at least this far after onset
    max_duration_ms : float or None  offset must be no further than this after
                      onset; None means bounded only by ``search_end_ms``
    env_window_ms   : float  RMS window width, matching the onset detector
    criterion       : float  SD multiplier for the baseline threshold
    significance    : float  percentile for the chance run-length criterion
    n_boot          : int    bootstrap resamples
    min_return_ms   : float  sustained sub-threshold duration required
    peak_frac       : float  fraction of the response's own peak envelope used
                      as a floor under the return threshold (default 0.12).
                      Set to 0 to use the baseline threshold alone.
    use_tkeo        : bool   run the detection on the TKEO signal
    refine_forward  : bool   refine the coarse crossing on a short envelope
    refine_window_ms, refine_sd_mult, refine_sustain_ms
                    : refinement parameters, mirroring the onset detector
    seed            : int    RNG seed

    Returns
    -------
    offset_ms : float, or None if no confident return to baseline was found
    """
    if onset_ms is None:
        return None

    signal = np.asarray(signal, dtype=float)
    n = signal.size
    if n < 16 or fs <= 0:
        return None

    ms_per_samp = 1000.0 / fs
    stim_idx = int(pre_ms * fs / 1000.0)
    if stim_idx < 5 or stim_idx >= n:
        return None

    onset_idx = int(stim_idx + onset_ms * fs / 1000.0)
    if onset_idx < 0 or onset_idx >= n:
        return None

    end_idx = min(int((pre_ms + search_end_ms) * fs / 1000.0), n)
    if max_duration_ms is not None:
        end_idx = min(end_idx,
                      int(onset_idx + max_duration_ms * fs / 1000.0) + 1)
    if end_idx <= onset_idx + 2:
        return None

    det = apply_tkeo(signal) if use_tkeo else signal
    env = compute_rms_envelope(det, fs, window_ms=env_window_ms, causal=False)

    env_win = max(1, int(round(env_window_ms * fs / 1000.0)))
    guard = env_win // 2 + 1
    env_pre = env[:max(0, stim_idx - guard)]

    base = compute_envelope_baseline(env_pre, criterion=criterion)
    if base is None:
        return None

    # A 'lower tail' run-length criterion is not the right calibration here:
    # the question is how long a sub-threshold stretch must be to indicate the
    # response has genuinely ended, which is a physiological choice
    # (min_return_ms), not a noise-chance question. The chance criterion is
    # used only as a floor, so an unusually noisy baseline cannot be satisfied
    # by an implausibly short quiet stretch.
    chance_floor = bootstrap_runlength_criterion(
        env_pre, criterion=criterion, significance=significance,
        n_boot=n_boot, seed=seed, tail="upper", min_samples=2,
        block_samples=env_win)
    return_samples = max(int(round(min_return_ms * fs / 1000.0)), chance_floor)

    # Anchor the forward scan at the dominant peak so a dip immediately after
    # onset (before the response has developed) cannot be mistaken for the end.
    resp = np.abs(signal[onset_idx:end_idx])
    if resp.size == 0:
        return None
    peak_idx = onset_idx + int(np.argmax(resp))

    # ── Return threshold: absolute floor OR a fraction of this response ──────
    # A purely baseline-derived threshold is an ABSOLUTE level, and on a
    # resting recording it is a very low one. Measured on real resting data
    # (PreStimRMS ~0.006 mV), requiring a 2 mV response to decay back inside
    # 2.5 SD of that noise floor failed on 112 of 127 trials -- and failed
    # WORSE the larger the response, which is the wrong way round. Median SNR
    # was 12 where an offset was found and 38 where it was not: real EMG does
    # not settle to resting-quiet within tens of milliseconds after a large
    # response, because of after-activity and filter ringing.
    #
    # Taking the larger of the baseline threshold and a fraction of the
    # response's own peak makes the criterion scale with the response, which is
    # the same principle the peak-fraction ONSET detector already uses. A small
    # response still gets the strict baseline threshold; a large one is not
    # held to a floor it will never reach.
    peak_env = float(np.max(env[peak_idx:end_idx])) if end_idx > peak_idx else 0.0
    return_threshold = base.threshold
    if peak_frac and peak_frac > 0 and peak_env > 0:
        return_threshold = max(return_threshold, float(peak_frac) * peak_env)

    crossing = find_sustained_run(env, return_threshold, return_samples,
                                  lo=peak_idx, hi=end_idx, above=False)
    if crossing is None:
        return None

    offset_idx = crossing
    if refine_forward:
        refined = _refine_offset_anchor(
            signal, fs, anchor_idx=crossing, floor_idx=peak_idx,
            ceiling_idx=end_idx, stim_idx=stim_idx,
            search_radius=env_win,
            refine_window_ms=refine_window_ms,
            refine_sd_mult=refine_sd_mult,
            refine_sustain_ms=refine_sustain_ms,
            floor_level=(float(peak_frac) * peak_env
                         if (peak_frac and peak_frac > 0) else 0.0))
        if refined is not None:
            offset_idx = refined

    offset_ms = (offset_idx - stim_idx) * ms_per_samp
    if offset_ms - float(onset_ms) < min_duration_ms:
        return None
    if max_duration_ms is not None and \
            (offset_ms - float(onset_ms)) > max_duration_ms:
        return None
    return round(float(offset_ms), 2)


def _refine_offset_anchor(signal, fs, *, anchor_idx, floor_idx, ceiling_idx,
                          stim_idx, search_radius, refine_window_ms,
                          refine_sd_mult, refine_sustain_ms, floor_level=0.0):
    """
    Re-detect the return to baseline on a short envelope near the anchor.

    The mirror of the onset refinement, and bidirectional for the same reason:
    a centred coarse window of width W makes the envelope fall below threshold
    roughly W/2 LATE, because energy from the response is still inside the
    window, while a causal window makes it early. Re-detecting on a short
    envelope within ``anchor_idx +/- search_radius`` corrects either direction
    without assuming which applies.

    Bounded below by the response peak so refinement can never place the
    offset inside the rising phase of the response.

    Returns an index into ``signal``, or None.
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
    if lo >= hi:
        return None

    # Carry the coarse stage's relative floor through, or the refinement walks
    # the offset back to wherever the strict baseline threshold was last
    # crossed and undoes the correction it was called to make.
    thr = max(fine_base.threshold, float(floor_level))
    return find_sustained_run(fine, thr, sustain, lo=lo, hi=hi, above=False)


def resolve_mep_offset(
        signal, fs, *,
        onset_ms,
        csp_start_ms=None,
        csp_enabled=False,
        manual_offset_ms=None,
        **detector_kwargs):
    """
    Apply the offset precedence rule and return offset, source and duration.

    Precedence, highest first:

    1. ``manual_offset_ms``  -- an explicit marker set in the Data Inspector
       always wins, as everywhere else in the pipeline.
    2. ``csp_start_ms`` when ``csp_enabled`` -- during voluntary contraction
       the end of the MEP and the start of the silent period are the same
       event, so they are reported as the same number rather than as two
       near-duplicate estimates.
    3. ``detect_mep_offset`` -- envelope return to baseline.
    4. None.

    Parameters
    ----------
    signal, fs        : trial segment and sampling frequency
    onset_ms          : float or None  detected/overridden onset
    csp_start_ms      : float or None  detected cSP start (ms post-stim)
    csp_enabled       : bool  whether cSP detection is active for this stim type
    manual_offset_ms  : float or None  inspector override
    **detector_kwargs : forwarded to ``detect_mep_offset``

    Returns
    -------
    MepOffset(offset_ms, source, duration_ms)
        ``source`` is one of ``OFFSET_SOURCES``. ``duration_ms`` is
        ``offset_ms - onset_ms``, or None if either endpoint is missing or the
        result would be non-positive.
    """
    def _finish(offset, source):
        dur = None
        if offset is not None and onset_ms is not None:
            d = float(offset) - float(onset_ms)
            dur = round(d, 2) if d > 0 else None
        return MepOffset(offset, source, dur)

    if manual_offset_ms is not None:
        return _finish(round(float(manual_offset_ms), 2), "manual")

    if csp_enabled and csp_start_ms is not None:
        return _finish(round(float(csp_start_ms), 2), "csp_start")

    if onset_ms is None:
        return _finish(None, "none")

    offset = detect_mep_offset(signal, fs, onset_ms=onset_ms,
                               **detector_kwargs)
    if offset is None:
        return _finish(None, "none")
    return _finish(offset, "envelope")
