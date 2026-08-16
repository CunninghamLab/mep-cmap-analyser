"""
mep_cmap.event_delay
~~~~~~~~~~~~~~~~~~~~
Measure the offset between a file's event marker and the actual stimulus.

The marker in a recording is not always the instant the stimulus fired. A
trigger written by software after the pulse, a stimulator delay setting, or a
different signal path for one block will all shift it, and the shift is usually
constant rather than random.

Why this matters more than it looks
-----------------------------------
A late marker does not produce a visibly wrong latency. It produces an epoch
whose t=0 is wrong, so part of the response falls into the pre-stimulus window
-- and then every measure defined relative to the baseline fails, each in a way
that looks like a separate fault. On one real recording whose markers were
2 ms late:

  * the derivative-ratio detector returned 0 onsets from 15 trials, because its
    gates compare the response against a baseline that now contained the
    response;
  * peak-to-peak was read from a 2.1 mV shoulder rather than the 3.8 mV peak,
    because with no onsets the amplitude window could not be anchored and fell
    back to a file-wide default;
  * the offset landed mid-excursion, because the artefact smeared into the
    envelope baseline and raised the return threshold 73-fold.

Those were diagnosed as three unrelated problems before the common cause was
found. Detecting the delay directly is much cheaper than detecting its
consequences.

Finding the artefact: slope, not amplitude
------------------------------------------
The stimulus artefact is the STEEPEST feature in the epoch, but not always the
largest. Measured on real recordings:

                        by |amplitude|        by |slope|
    sub-001  A        6.90 ms (sd 1.44)   0.20 ms (sd 0.15)
    sub-001  C       -1.60 ms (sd 0.07)  -1.60 ms (sd 0.14)
    sub-006  D        0.40 ms (sd 7.69)   0.30 ms (sd 0.13)

Peak amplitude finds the M-wave on sub-001 A -- 6.90 ms is the response peak,
not the stimulus. Peak derivative finds the artefact in every case.

The spread is the confidence signal
-----------------------------------
Where the offset is a genuine fixed trigger delay the standard deviation across
trials is a fraction of a millisecond. Where the marker really does jitter, or
where there is no artefact to find at all, the maximum derivative lands
somewhere different on every trial and the spread blows up -- sub-001 G gives
3.93 ms.

So the same statistic that estimates the delay also says whether a delay is the
right model. Above the threshold this module proposes nothing and reports why,
rather than offering a number that would be applied to every trial.

  * measure_event_delay   -- one stimulus type
  * scan_event_delays     -- every stimulus type in a recording
  * EventDelay            -- the result, including the reason when declining
"""

from collections import namedtuple

import numpy as np

# Half-width of the search window, in ms either side of the marker. Comfortably
# covers the largest delay reported from the field (-10 ms) without reaching so
# far that a large response transient can be mistaken for the artefact.
DEFAULT_SCAN_MS = 20.0

# Standard deviation across trials, in ms, above which a single delay is not a
# good model of what is happening. Real fixed delays measure 0.1-0.2 ms; a
# recording with genuinely inconsistent markers measured 3.93 ms. Nothing
# observed sits near this boundary, so it is not doing delicate work.
DEFAULT_SPREAD_LIMIT_MS = 1.0

# Median offsets smaller than this are reported as "no delay", not proposed as
# a correction: sample-clock rounding should not be dignified as a finding.
DEFAULT_NEGLIGIBLE_MS = 0.3

# Minimum trials before a median and spread mean anything.
DEFAULT_MIN_TRIALS = 3

# Maximum width, in ms at half its own maximum, of a transient that can be
# accepted as a stimulus artefact.
#
# Without this the scan is dangerous rather than merely unhelpful on a
# recording with no artefact: the steepest feature is then the response's own
# rising edge, and because that edge is highly consistent across trials the
# spread test passes. The scan would confidently propose moving t=0 onto the
# response -- far worse than proposing nothing.
#
# The two are cleanly separable by width. Measured across every condition of
# two real recordings, the stimulus artefact is 0.4-0.6 ms wide at half its
# maximum; a response edge in the same units is 4.6 ms. Nothing observed sits
# between 0.6 and 4.6, so this threshold is not doing delicate work either.
DEFAULT_MAX_WIDTH_MS = 1.5


def _transient_width_ms(deriv, k, fs):
    """Width of the transient at index ``k``, at half its maximum, in ms."""
    peak = float(deriv[k])
    if peak <= 0:
        return float("inf")
    i = k
    while i > 0 and deriv[i] > 0.5 * peak:
        i -= 1
    j = k
    while j < len(deriv) - 1 and deriv[j] > 0.5 * peak:
        j += 1
    return (j - i) * 1000.0 / fs


EventDelay = namedtuple(
    "EventDelay",
    ["stim_type", "delay_ms", "median_ms", "sd_ms", "n_trials",
     "proposed", "reason"],
)


def measure_event_delay(emg, fs, stim_times, stim_type="",
                        scan_ms=DEFAULT_SCAN_MS,
                        spread_limit_ms=DEFAULT_SPREAD_LIMIT_MS,
                        negligible_ms=DEFAULT_NEGLIGIBLE_MS,
                        min_trials=DEFAULT_MIN_TRIALS,
                        max_width_ms=DEFAULT_MAX_WIDTH_MS):
    """Offset between the markers of one stimulus type and the artefact.

    Parameters
    ----------
    emg         : 1-D np.ndarray   the continuous recording
    fs          : float            sampling rate, Hz
    stim_times  : sequence[float]  marker times in seconds
    stim_type   : str              label, for the returned record only
    scan_ms     : float            half-width of the search window, ms
    spread_limit_ms : float        SD above which no delay is proposed
    negligible_ms   : float        |median| below which no delay is proposed
    min_trials      : int          fewer than this and nothing is proposed

    Returns
    -------
    EventDelay

    ``delay_ms`` is the correction to apply, and is ``0.0`` whenever
    ``proposed`` is False. ``median_ms`` and ``sd_ms`` are always the measured
    values, so a caller can show what was seen even when nothing is proposed --
    a wide spread is itself worth reporting to the analyst.
    """
    emg = np.asarray(emg, dtype=float)
    times = [float(t) for t in (stim_times or [])]

    def _decline(reason, med=float("nan"), sd=float("nan"), n=0):
        return EventDelay(stim_type, 0.0, med, sd, n, False, reason)

    if emg.size < 4 or fs <= 0:
        return _decline("no signal to measure")
    if len(times) < min_trials:
        return _decline(f"only {len(times)} trial(s); {min_trials} needed",
                        n=len(times))

    half = int(round(scan_ms * fs / 1000.0))
    if half < 2:
        return _decline("scan window is shorter than two samples")

    offsets = []
    widths = []
    for t in times:
        i = int(round(t * fs))
        lo, hi = i - half, i + half
        if lo < 1 or hi >= emg.size:
            continue                      # marker too close to an edge
        window = emg[lo:hi]
        # The artefact is the steepest transient, which is why this is the
        # derivative and not the raw signal: on a supramaximal M-wave the
        # response is several times larger than the artefact but never as
        # abrupt.
        deriv = np.abs(np.diff(window))
        k = int(np.argmax(deriv))
        offsets.append((lo + k - i) * 1000.0 / fs)
        widths.append(_transient_width_ms(deriv, k, fs))

    n = len(offsets)
    if n < min_trials:
        return _decline(f"only {n} trial(s) fully inside the recording", n=n)

    arr = np.asarray(offsets, dtype=float)
    median = float(np.median(arr))
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0

    # Is the steepest feature actually an artefact, or the response itself?
    # This check comes before the spread test on purpose: a response edge is
    # very consistent across trials, so it would sail through the spread test
    # and be proposed with confidence.
    width = float(np.median(widths))
    if width > max_width_ms:
        return _decline(
            f"the steepest feature is {width:.1f} ms wide, too broad for a "
            f"stimulus artefact — this looks like the response itself, so no "
            f"delay can be measured from it",
            median, sd, n)

    if sd > spread_limit_ms:
        return _decline(
            f"artefact timing varies across trials (SD {sd:.2f} ms); the "
            f"markers are inconsistent rather than offset, so a single delay "
            f"would not correct them",
            median, sd, n)

    if abs(median) < negligible_ms:
        return _decline(
            f"artefact is within {negligible_ms:g} ms of the marker; no delay "
            f"needed", median, sd, n)

    # Report a whole number of samples: a delay finer than the sample interval
    # cannot be applied, and rounding here keeps what is shown identical to
    # what is used.
    delay = round(median * fs / 1000.0) * 1000.0 / fs
    return EventDelay(stim_type, float(delay), median, sd, n, True, "")


def scan_event_delays(emg, fs, stim_times_by_type, **kwargs):
    """Run ``measure_event_delay`` for every stimulus type.

    Returns ``{stim_type: EventDelay}`` in sorted order. Types that decline are
    included, so the caller can report why rather than leaving a silent gap.
    """
    return {st: measure_event_delay(emg, fs, times, stim_type=st, **kwargs)
            for st, times in sorted((stim_times_by_type or {}).items())}


def format_scan_report(results, scan_ms=DEFAULT_SCAN_MS):
    """Human-readable lines for the log, one per stimulus type."""
    out = [f"🔎 Event delay scan (stimulus artefact by peak slope, "
           f"\u00b1{scan_ms:g} ms window)"]
    for st, r in results.items():
        if r.proposed:
            out.append(f"   {st}: artefact at {r.median_ms:+.1f} ms "
                       f"(SD {r.sd_ms:.2f}, n={r.n_trials}) — proposing delay "
                       f"{r.delay_ms:+.1f} ms")
        elif np.isfinite(r.median_ms):
            out.append(f"   {st}: artefact at {r.median_ms:+.1f} ms "
                       f"(SD {r.sd_ms:.2f}, n={r.n_trials}) — {r.reason}")
        else:
            out.append(f"   {st}: {r.reason}")
    return out
