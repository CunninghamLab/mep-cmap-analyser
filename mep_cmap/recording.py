"""
mep_cmap.recording
~~~~~~~~~~~~~~~~~~~
Normalised in-memory representation of a single EMG recording, plus a builder
that aggregates ``mep_cmap.io``'s existing per-channel public API into one
object.

This is the single struct the BIDS-ify writer consumes, so the writer never has
to know about format-specific quirks:

    io.py        any supported source  ->  channels + fs + events
    recording.py (this module)         ->  one faithful Recording object
    bidsify.py   Recording             ->  EDF/BDF + sidecars

Decoupling
----------
This module depends only on ``io.py``'s *existing* public functions
(``list_waveform_channels``, ``extract_emg_waveform_and_fs``,
``extract_stim_times``, ``detect_format``); ``io.py`` is imported lazily inside
the builder so there is no import cycle, and nothing here requires a change to
``io.py``. An optional ``io.read_recording()`` wrapper can be added later as
one-line sugar over :func:`build_recording`.

The :meth:`Recording.signature` / :func:`compare_signatures` pair is the
verification primitive for the post-conversion read-back check (confirm an
EDF/BDF written from a Recording matches the source on channel count, sampling
rate, sample count and per-channel RMS before declaring success).
"""

import os
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np


# ── Channel ───────────────────────────────────────────────────────────────────
@dataclass
class Channel:
    """One waveform channel."""
    name:    str
    samples: np.ndarray
    unit:    Optional[str] = None
    kind:    str = "EMG"            # BIDS channels.tsv 'type' (EMG, MISC, TRIG, ...)
    fs:      Optional[float] = None  # per-channel fs as reported by the reader

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0]) if self.samples is not None else 0

    def rms(self) -> float:
        if self.samples is None or self.samples.size == 0:
            return 0.0
        x = np.asarray(self.samples, dtype=np.float64)
        return float(np.sqrt(np.mean(np.square(x))))


# ── Event ─────────────────────────────────────────────────────────────────────
@dataclass
class Event:
    """One stimulation/marker event, in seconds from recording start."""
    onset:      float
    duration:   float = 0.0
    trial_type: str = "n/a"
    value:      Optional[Any] = None   # original marker code/label, if any


# ── Recording ─────────────────────────────────────────────────────────────────
@dataclass
class Recording:
    """A single EMG recording: channels sharing one sampling frequency, plus events."""
    source_path:        str
    source_format:      str = ""
    sampling_frequency: float = 0.0
    channels:           list = field(default_factory=list)
    events:             list = field(default_factory=list)
    warnings:           list = field(default_factory=list)   # non-fatal build notes

    # ---- summary queries ------------------------------------------------------
    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def channel_names(self) -> list:
        return [c.name for c in self.channels]

    @property
    def units(self) -> list:
        return [c.unit for c in self.channels]

    @property
    def n_samples(self) -> int:
        """Sample count of the longest channel (channels are normally equal)."""
        return max((c.n_samples for c in self.channels), default=0)

    @property
    def duration_s(self) -> float:
        if not self.sampling_frequency:
            return 0.0
        return self.n_samples / float(self.sampling_frequency)

    def channels_equal_length(self) -> bool:
        lengths = {c.n_samples for c in self.channels}
        return len(lengths) <= 1

    # ---- writer-facing views --------------------------------------------------
    def data_matrix(self,
                    dtype=np.float64,
                    on_length_mismatch: str = "error") -> np.ndarray:
        """
        Return data as a 2-D array shaped (n_channels, n_samples).

        ``on_length_mismatch`` controls behaviour when channels differ in length:
          'error'    — raise (default; faithful, surfaces problems loudly)
          'truncate' — clip all channels to the shortest
          'pad'      — zero-pad all channels to the longest
        """
        if not self.channels:
            return np.empty((0, 0), dtype=dtype)

        lengths = [c.n_samples for c in self.channels]
        if len(set(lengths)) == 1:
            return np.vstack([np.asarray(c.samples, dtype=dtype)
                              for c in self.channels])

        if on_length_mismatch == "error":
            raise ValueError(
                f"Channel length mismatch {lengths}; pass on_length_mismatch="
                f"'truncate' or 'pad' to coerce.")
        if on_length_mismatch == "truncate":
            n = min(lengths)
            return np.vstack([np.asarray(c.samples[:n], dtype=dtype)
                              for c in self.channels])
        if on_length_mismatch == "pad":
            n = max(lengths)
            rows = []
            for c in self.channels:
                row = np.asarray(c.samples, dtype=dtype)
                if row.shape[0] < n:
                    row = np.concatenate([row, np.zeros(n - row.shape[0], dtype=dtype)])
                rows.append(row)
            return np.vstack(rows)
        raise ValueError(f"Unknown on_length_mismatch: {on_length_mismatch!r}")

    def events_table(self) -> list:
        """
        Return events as a list of dicts (onset/duration/trial_type/value),
        sorted by onset — ready to write a BIDS ``_events.tsv``.
        """
        rows = [{"onset": float(e.onset),
                 "duration": float(e.duration),
                 "trial_type": e.trial_type if e.trial_type is not None else "n/a",
                 "value": e.value}
                for e in self.events]
        rows.sort(key=lambda r: r["onset"])
        return rows

    # ---- verification ---------------------------------------------------------
    def signature(self) -> dict:
        """
        Compact fingerprint for the post-conversion read-back check:
        channel count, sampling frequency, and per-channel (name, n_samples, rms).
        """
        return {
            "n_channels": self.n_channels,
            "sampling_frequency": float(self.sampling_frequency),
            "channels": [
                {"name": c.name, "n_samples": c.n_samples, "rms": c.rms()}
                for c in self.channels
            ],
        }

    def summary(self) -> str:
        return (f"{os.path.basename(self.source_path)} "
                f"[{self.source_format or '?'}]: "
                f"{self.n_channels} ch, {self.sampling_frequency:g} Hz, "
                f"{self.duration_s:.2f} s, {len(self.events)} events")


# ── Builder over io.py's existing public API ──────────────────────────────────
def build_recording(path: str,
                    marker_names: Optional[Sequence[str]] = None,
                    stim_channel: Optional[str] = None,
                    channel_indices: Optional[Sequence[int]] = None,
                    io_module: Any = None,
                    fs_rtol: float = 1e-6) -> Recording:
    """
    Aggregate a source file into a :class:`Recording` using ``io.py``'s existing
    per-channel public functions.

    Parameters
    ----------
    path            : source data file
    marker_names    : stim marker/label names to pull events from via
                      ``io.extract_stim_times``. If None, no events are attached
                      (the caller usually already knows the selected stim labels).
    channel_indices : 0-based indices to include; None = all waveform channels.
    io_module       : injected for testing; defaults to ``mep_cmap.io``.
    fs_rtol         : relative tolerance for flagging per-channel fs mismatch.

    Notes
    -----
    Depends only on the documented public API of ``io.py``. If your live
    signatures differ from the project copy, this is the one place to adjust.
    """
    if io_module is None:
        from . import io as io_module   # lazy import → no cycle at module load

    rec = Recording(source_path=path)
    try:
        rec.source_format = io_module.detect_format(path)
    except Exception as exc:                       # detection is best-effort here
        rec.warnings.append(f"detect_format failed: {exc}")

    names = list(io_module.list_waveform_channels(path))
    if channel_indices is None:
        channel_indices = range(len(names))

    base_fs = None
    for idx in channel_indices:
        emg, fs, unit = io_module.extract_emg_waveform_and_fs(path, idx)
        emg = np.asarray(emg)
        ch_name = names[idx] if 0 <= idx < len(names) else f"ch{idx}"
        rec.channels.append(Channel(name=ch_name, samples=emg,
                                    unit=unit, kind="EMG", fs=float(fs)))
        if base_fs is None:
            base_fs = float(fs)
        elif base_fs and abs(float(fs) - base_fs) > fs_rtol * base_fs:
            rec.warnings.append(
                f"channel '{ch_name}' fs={fs} differs from {base_fs}; "
                f"recording fs kept as {base_fs}")

    rec.sampling_frequency = base_fs or 0.0

    if not rec.channels_equal_length():
        rec.warnings.append(
            f"channels differ in length: "
            f"{[c.n_samples for c in rec.channels]}")

    # Events
    if marker_names:
        for marker in marker_names:
            try:
                stim = io_module.extract_stim_times(path, marker, stim_channel=stim_channel) or {}
            except Exception as exc:
                rec.warnings.append(f"extract_stim_times({marker!r}) failed: {exc}")
                continue
            for stim_type, times in stim.items():
                for t in times:
                    rec.events.append(
                        Event(onset=float(t), duration=0.0,
                              trial_type=str(stim_type), value=str(stim_type)))

    return rec


# ── Verification helper ───────────────────────────────────────────────────────
def compare_signatures(ref: dict,
                       test: dict,
                       rms_rtol: float = 1e-3,
                       rms_atol: float = 1e-9,
                       fs_rtol: float = 1e-6,
                       sample_pad_tolerance: int = 1) -> tuple:
    """
    Compare a reference signature (from the source Recording) against a test
    signature (from the re-read EDF/BDF). Returns ``(ok, discrepancies)``.

    Channel order and names are compared positionally; sampling frequency within
    ``fs_rtol``; per-channel RMS within ``rms_rtol`` / ``rms_atol`` (quantisation
    to 16/24-bit means RMS won't match bit-for-bit, so a small tolerance is
    expected).

    Sample count: EDF/BDF stores whole data records, so a written file is
    legitimately zero-padded up to the next record boundary. The written length
    must therefore be >= the source and padded by *less than one record*
    (``sample_pad_tolerance`` = samples per record). Truncation (written <
    source) or padding of a whole record or more still fails. The default of 1
    requires exact equality (for callers that don't pass a record size).
    """
    d = []
    pad_tol = max(1, int(sample_pad_tolerance))

    if ref.get("n_channels") != test.get("n_channels"):
        d.append(f"channel count: source={ref.get('n_channels')} "
                 f"written={test.get('n_channels')}")

    rfs, tfs = ref.get("sampling_frequency", 0.0), test.get("sampling_frequency", 0.0)
    if rfs and abs(tfs - rfs) > fs_rtol * rfs:
        d.append(f"sampling frequency: source={rfs} written={tfs}")

    rch, tch = ref.get("channels", []), test.get("channels", [])
    for i in range(min(len(rch), len(tch))):
        r, t = rch[i], tch[i]
        diff_n = int(t.get("n_samples", 0)) - int(r.get("n_samples", 0))
        if diff_n < 0 or diff_n >= pad_tol:
            d.append(f"ch{i} '{r.get('name')}' samples: "
                     f"source={r.get('n_samples')} written={t.get('n_samples')} "
                     f"(diff {diff_n:+d}, allowed 0..{pad_tol - 1} padding)")
        rr, tr = r.get("rms", 0.0), t.get("rms", 0.0)
        if not math.isclose(rr, tr, rel_tol=rms_rtol, abs_tol=rms_atol):
            d.append(f"ch{i} '{r.get('name')}' RMS: "
                     f"source={rr:.6g} written={tr:.6g}")

    return (len(d) == 0, d)
