"""
mep_cmap.event_sources
~~~~~~~~~~~~~~~~~~~~~~
Where stimulus events come from, stated explicitly.

The problem this replaces
-------------------------
Every reader exposed ``extract_stim_times(path, marker_name)``, and every reader
meant something different by it:

    Spike2 SMR           the event channel to read
    LabChart .mat        ignored -- always the comment table
    BrainVision          ignored -- always the .vmrk markers
    EDF/BDF              ignored -- always the annotations
    AcqKnowledge, CFWB   the LABEL to attach to threshold-detected events
    LabChart text        the trigger channel to threshold

The shared signature was a coincidence of naming rather than an interface.
Asking a reader for something it did not already do was impossible, because the
one parameter that could carry the request already meant something else there.

So LabChart .mat read comments and nothing else -- not a missing feature so much
as a missing question. The format carries comments, digital inputs and
fixed-interval sampling; the tool could only ask for the first.

Why this is affordable across nine formats
------------------------------------------
Threshold and interval detection do not depend on the file format at all. Both
operate on a waveform and a time base, which every reader already provides. They
are written once, here, and a reader only has to say which of its channels are
analogue. Without that, the same detection logic would have to be written nine
times and would drift nine ways.

  * EventSource                  -- where events come from
  * detect_threshold_crossings   -- a TTL or analogue trigger
  * generate_interval_events     -- fixed timing, for triggers the file omits
  * merge_event_sources          -- several sources into one mapping
  * decimate_for_preview         -- a long recording reduced for display
"""

from dataclasses import asdict, dataclass

import numpy as np

# Two events from DIFFERENT sources closer together than this are reported.
#
# They are NOT merged. A comment written by software just after a TTL pulse for
# the same stimulus looks exactly like two genuine events a few milliseconds
# apart, and deciding they are one is the sort of assumption that produces a
# timebase error nobody notices. A visible duplicate can be diagnosed; a silent
# merge cannot.
NEAR_SIMULTANEOUS_MS = 5.0

# Minimum time between accepted threshold crossings. A stimulator pulse rings,
# and without a refractory period each oscillation past the level counts as its
# own stimulus.
DEFAULT_REFRACTORY_MS = 50.0

KINDS = ("embedded", "threshold", "interval")
EDGES = ("rising", "falling", "both")


@dataclass
class EventSource:
    """Where one set of stimulus events comes from.

    kind="embedded"
        Comments, markers, annotations or event channels -- whatever the format
        already carries. ``channel`` names the source within the file where
        that is meaningful; ``codes`` optionally restricts to particular labels.

    kind="threshold"
        A crossing on an analogue channel. ``label`` is the stimulus type the
        detected events belong to, since the signal itself carries no name.

    kind="interval"
        Fixed timing, for recordings triggered by something the file does not
        record. The events are asserted rather than detected, so nothing can
        verify them and the analyst carries the responsibility.
    """

    kind: str = "embedded"
    channel: str = ""
    codes: tuple = ()
    label: str = "A"

    # threshold
    level: float = 0.0
    edge: str = "rising"
    refractory_ms: float = DEFAULT_REFRACTORY_MS

    # interval
    start_s: float = 0.0
    period_s: float = 1.0
    count: int = 0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown event source kind: {self.kind!r}")
        if self.edge not in EDGES:
            raise ValueError(f"unknown edge: {self.edge!r}")
        self.codes = tuple(self.codes or ())

    def describe(self):
        """One line for the interface and the log."""
        if self.kind == "embedded":
            what = self.channel or "the file's own events"
            if self.codes:
                what += f", codes {', '.join(self.codes)}"
            return f"embedded: {what}"
        if self.kind == "threshold":
            return (f"threshold: {self.channel} {self.edge} through "
                    f"{self.level:g}, {self.refractory_ms:g} ms refractory "
                    f"\u2192 {self.label}")
        return (f"interval: every {self.period_s:g} s from {self.start_s:g} s"
                + (f", {self.count} events" if self.count else "")
                + f" \u2192 {self.label}")

    def to_dict(self):
        d = asdict(self)
        d["codes"] = list(self.codes)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        d["codes"] = tuple(d.get("codes") or ())
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


def detect_threshold_crossings(signal, fs, level, edge="rising",
                               refractory_ms=DEFAULT_REFRACTORY_MS,
                               t0=0.0):
    """Times, in seconds, at which ``signal`` crosses ``level``.

    Parameters
    ----------
    signal        : 1-D array_like
    fs            : float   sampling rate, Hz
    level         : float   in the signal's own units
    edge          : "rising" | "falling" | "both"
    refractory_ms : float   minimum spacing between accepted crossings
    t0            : float   time of the first sample, seconds

    The refractory period is not cosmetic. A stimulator pulse rings, and each
    oscillation back past the level is a crossing; without it a single stimulus
    yields several events a fraction of a millisecond apart, which then look
    like a very fast train rather than an artefact of the detection.

    The crossing is reported at the first sample on the far side, not the last
    on the near side, so a rising edge is timed at the sample where the signal
    is first above the level.
    """
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 2 or not fs:
        return []

    above = x > float(level)
    if edge == "rising":
        idx = np.flatnonzero(~above[:-1] & above[1:]) + 1
    elif edge == "falling":
        idx = np.flatnonzero(above[:-1] & ~above[1:]) + 1
    else:
        idx = np.flatnonzero(above[:-1] != above[1:]) + 1

    if idx.size == 0:
        return []

    gap = int(round(max(0.0, float(refractory_ms)) * fs / 1000.0))
    if gap > 0:
        kept = [int(idx[0])]
        for i in idx[1:]:
            if int(i) - kept[-1] >= gap:
                kept.append(int(i))
        idx = np.asarray(kept, dtype=int)

    return [float(t0) + float(i) / float(fs) for i in idx]


def generate_interval_events(start_s, period_s, count=0, duration_s=None):
    """Events at a fixed interval.

    For recordings triggered by something the file does not record. Nothing
    here is detected: the times are asserted, and no part of the recording can
    confirm them. That is the point -- and the reason a source of this kind
    should be stated in the methods of anything reported from it.

    ``count`` gives an exact number; otherwise events continue to
    ``duration_s``. One of the two must be supplied.
    """
    period = float(period_s)
    if period <= 0:
        raise ValueError("interval period must be positive")
    start = float(start_s)

    if count and int(count) > 0:
        n = int(count)
    elif duration_s:
        n = int(np.floor((float(duration_s) - start) / period)) + 1
    else:
        raise ValueError("interval source needs a count or a recording length")

    if n <= 0:
        return []
    times = [start + k * period for k in range(n)]
    if duration_s is not None:
        times = [t for t in times if t <= float(duration_s)]
    return times


def merge_event_sources(per_source, near_ms=NEAR_SIMULTANEOUS_MS):
    """Combine ``{stim_type: [times]}`` mappings from several sources.

    Returns ``(merged, warnings)``.

    Two rules, both chosen so that a mistake is visible rather than absorbed:

    A stimulus type receiving events from more than one source is an ERROR, not
    a union. Two sources disagreeing about the same type is far more likely to
    be a misconfiguration than an intention, and silently combining them would
    give a trial count that matches neither source.

    Events from different sources closer together than ``near_ms`` are KEPT and
    reported. They may be one stimulus recorded twice -- a comment written just
    after a TTL pulse -- or two genuine stimuli in a paired-pulse protocol, and
    nothing in the data distinguishes those. Merging would silently halve a
    paired-pulse trial count.
    """
    merged, warnings, seen = {}, [], {}

    for src_name, mapping in (per_source or {}):
        for stim_type, times in (mapping or {}).items():
            if stim_type in merged:
                warnings.append(
                    f"stimulus type '{stim_type}' is produced by both "
                    f"{seen[stim_type]} and {src_name}; rename one, or remove "
                    f"the source that should not define it")
                continue
            merged[stim_type] = sorted(float(t) for t in times)
            seen[stim_type] = src_name

    # Near-simultaneous events ACROSS sources.
    flat = sorted((t, st) for st, ts in merged.items() for t in ts)
    tol = float(near_ms) / 1000.0
    close = 0
    for (t1, s1), (t2, s2) in zip(flat, flat[1:]):
        if s1 != s2 and (t2 - t1) <= tol:
            close += 1
    if close:
        warnings.append(
            f"{close} event(s) from different stimulus types fall within "
            f"{near_ms:g} ms of each other. They are kept as separate events. "
            f"If one recording device logged the same stimulus twice, remove "
            f"the source that duplicates it; if this is a paired-pulse "
            f"protocol, no action is needed")

    return merged, warnings


def decimate_for_preview(signal, fs, max_points=4000, t0=0.0):
    """Reduce a recording to something a preview plot can draw.

    Returns ``(times, lows, highs)``: for each displayed column, the minimum
    and maximum of the samples it covers. Plotting those as a filled band
    preserves every transient, which plain subsampling does not -- and a
    stimulus trigger is exactly the kind of one-sample spike that subsampling
    drops. A preview that loses the pulses would be worse than none, since the
    analyst would set a level against a trace that does not show what the
    detector sees.

    A recording of two thousand seconds at five kilohertz is ten million
    samples; drawing that directly is slow enough to make the level box feel
    broken.
    """
    x = np.asarray(signal, dtype=float).ravel()
    n = x.size
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    step = max(1, int(np.ceil(n / max(1, int(max_points)))))
    if step == 1:
        t = t0 + np.arange(n) / float(fs)
        return t, x, x

    usable = (n // step) * step
    block = x[:usable].reshape(-1, step)
    lows, highs = block.min(axis=1), block.max(axis=1)
    if usable < n:                      # keep the tail rather than dropping it
        tail = x[usable:]
        lows = np.append(lows, tail.min())
        highs = np.append(highs, tail.max())
    t = t0 + (np.arange(lows.size) * step + step / 2.0) / float(fs)
    return t, lows, highs


# A refractory suggestion was written here and removed. It proposed half the
# shortest detected interval, which on a protocol with ten-second spacing gave
# five seconds -- long enough to discard every stimulus of a faster block in
# the same recording. The fixed default is 50 ms, and the preview shows what
# any value does; a suggestion with no evidence behind it is worse than none,
# because it arrives looking like an answer.
