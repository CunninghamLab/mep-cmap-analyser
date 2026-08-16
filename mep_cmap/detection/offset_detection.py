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

  * detect_mep_offset      -- envelope return-to-baseline detector
  * resolve_mep_offset     -- precedence rule + duration
  * offset_marker_field    -- which marker carries the offset, for the Inspector
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
        # Use the settled level late in the epoch as a floor under the return
        # threshold when it sits above the pre-stimulus level. Set to 0 to
        # judge the return against the pre-stimulus baseline alone.
        settle_frac=1.0,
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

    base = compute_envelope_baseline(env_pre, criterion=criterion, robust=True)

    # Where has the signal actually SETTLED?
    #
    # The return threshold has been derived from the pre-stimulus baseline
    # alone, which assumes the signal comes back to where it started. On real
    # recordings it often does not: measured across every condition of one
    # session, the envelope floor late in the epoch sat 1.3 to 2.0 times the
    # pre-stimulus floor. "Return to the pre-stimulus level" is then a target
    # the signal never reaches, and the detector waits for a chance dip --
    # reporting an offset 50 to 90 ms after the trace has visibly flattened,
    # and dragging the area-under-curve window along with it.
    #
    # So the floor is taken as the LARGER of the two: where the signal started,
    # and where it ends up. The late window is placed well past any plausible
    # response and is only used when there is enough of it to be meaningful.
    settled = None
    if settle_frac > 0:
        # max_duration_ms is optional; fall back to the end of the search
        # window, which is the other bound on how late an offset can be.
        _tail_ms = (float(max_duration_ms) if max_duration_ms
                    else float(search_end_ms))
        tail_lo = int(stim_idx + _tail_ms * fs / 1000.0)
        if 0 < tail_lo < env.size - int(20 * fs / 1000.0):
            settled = compute_envelope_baseline(env[tail_lo:], criterion=criterion,
                                                robust=True)
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
    if settled is not None and settled.threshold > return_threshold:
        # The signal has settled ABOVE where it started, so the pre-stimulus
        # threshold can never be met and the offset would be reported wherever
        # the envelope happened to dip. Judge the return against where the
        # trace actually flattens instead.
        return_threshold = float(settled.threshold)
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
            # The refinement must not walk back past the settled level either,
            # or it undoes the correction just applied.
            floor_level=max(
                float(peak_frac) * peak_env
                if (peak_frac and peak_frac > 0) else 0.0,
                float(settled.threshold) if settled is not None else 0.0))
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
                                          criterion=refine_sd_mult, robust=True)
    if fine_base is None:
        return None

    sustain = max(1, int(round(refine_sustain_ms * fs / 1000.0)))
    anchor_idx = int(anchor_idx)
    floor_idx = int(floor_idx)
    if anchor_idx <= floor_idx or anchor_idx >= fine.size:
        return None

    # Carry the coarse stage's relative floor through, or the refinement walks
    # the offset back to wherever the strict baseline threshold was last
    # crossed and undoes the correction it was called to make.
    thr = max(fine_base.threshold, float(floor_level))

    # Walk BACK from the coarse crossing to the last sample that was still
    # elevated; the response ends just after it.
    #
    # The previous version scanned a fixed neighbourhood for the first
    # sustained sub-threshold run. When the fine envelope was already quiet at
    # the start of that neighbourhood -- the normal case, since the coarse
    # window smears the crossing late -- it returned the neighbourhood's own
    # lower bound. The reported offset was then exactly `anchor - env_window_ms`
    # on every trial: a function of a smoothing setting rather than of the
    # data, and visibly parked at an arbitrary point on the trace.
    #
    # Searching backwards for the last elevated sample removes the dependence
    # on the search radius entirely: the answer is where the signal actually
    # stopped.
    last_active = None
    for i in range(anchor_idx, floor_idx - 1, -1):
        if fine[i] > thr:
            last_active = i
            break
    if last_active is None:
        # Quiet all the way back to the peak, which means the coarse crossing
        # was not late after all. Keep it rather than inventing a landmark.
        return None

    # Require the quiet stretch after it to be real, not a single dip.
    end = min(last_active + 1 + sustain, fine.size)
    if end - (last_active + 1) < sustain:
        return None
    if float(np.mean(fine[last_active + 1:end] <= thr)) < 0.5:
        return None

    return min(last_active + 1, int(ceiling_idx) - 1)


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


# ── Which marker carries the offset in the Data Inspector ────────────────────

def offset_marker_field(csp_enabled, has_csp_markers):
    """Name of the metadata key the offset marker should read and write.

    During voluntary contraction the end of the MEP and the start of the
    cortical silent period are the same physical event, so the Inspector shows
    ONE marker for them, not two. Two draggable markers for one event can be
    moved apart, and then the file contains two different answers to the same
    question with nothing to say which is right.

    Returns
    -------
    "silent_start_idx" when a silent period is enabled and marked -- the
    existing cSP-start marker doubles as the offset marker, and dragging it
    moves both.

    "mep_offset_idx" otherwise -- a dedicated marker, which is what a
    resting-state recording needs since it has no silent period.

    ``resolve_mep_offset`` applies the same precedence when quantifying, so
    what the analyst drags and what the pipeline reports are the same thing.
    """
    if csp_enabled and has_csp_markers:
        return "silent_start_idx"
    return "mep_offset_idx"
