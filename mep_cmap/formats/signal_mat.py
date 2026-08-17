"""
mep_cmap.formats.signal_mat
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reader for **CED Signal MATLAB exports** — frame-based sweeps saved from
Signal's File ▸ Export As ▸ MATLAB.

Written from the files themselves. Signal's export is self-describing: field
names, sample interval, epoch start, channel titles and units, and a label per
frame are all recorded in the file, so nothing here is derived from any other
toolbox's source. That matters, because the MATLAB reader for this format
belongs to a GPL project and this package is MIT.

Shape of the file
-----------------
A single top-level struct whose name ends in ``wave_data``::

    wave_data.values     (nframe, nchan, npoint)  float
    wave_data.interval   sample interval, seconds
    wave_data.start      time of the first sample RELATIVE to the trigger
    wave_data.points     samples per frame
    wave_data.frames     number of frames
    wave_data.chaninfo.title / .units       per channel
    wave_data.frameinfo.label / .state      per frame
    wave_data.frameinfo.start               frame time in the session, seconds

The variable is NOT always called ``wave_data``: Signal names it after the
recording, so a second file in the same folder is ``V231115_SICF000_wave_data``.
Matching on the suffix rather than the whole name is the difference between
reading a lab's files and reading only the two that happen to be examples.

MATLAB v7.3
-----------
These are HDF5, which ``scipy.io.loadmat`` refuses. Every other ``.mat`` reader
in this package uses scipy, so a v7.3 file previously fell through detection
entirely and reported ``unsupported_binary``. h5py is used here directly and
imported lazily, so the dependency is only needed by analysts who have such
files.

Why it presents as pre-epoched
------------------------------
Frames are trials already cut around the trigger, which is exactly what
``epoched_mat`` handles, so the same machinery is reused rather than
reimplemented: the epochs are stitched into a pseudo-continuous trace with
mirror-tiled, DC-anchored, cosine-crossfaded guard bands. That logic exists so
filter transients and over-long analysis windows cannot reach out of one trial
into the previous one, and the reasoning applies here identically.

``get_epoch_bounds`` reports what the frames actually contain, so callers clamp
their windows to it. A Signal file with ±500 ms frames has no data beyond that,
and an unclamped window would measure guard padding and report it as signal.

Stimulus types
--------------
``frameinfo.label`` gives Signal's frame state — "State 1", "State 2" and so on
— which is a per-trial grouping, so each becomes a StimType. A file whose
frames are all one state yields a single type, which is the ordinary case.

Units
-----
``chaninfo.units`` states the unit per channel, so nothing is inferred and
``units_assumed`` is always False — unlike ``epoched_mat``, where the unit is
guessed and confirmed by dialogue.

Public API (the io.py contract, plus two extensions)
----------------------------------------------------
  is_signal_mat(file_path)                     -> bool
  list_waveform_channels(file_path)            -> list[str]
  extract_emg_waveform_and_fs(file_path, ch)   -> (np.ndarray, int, str|None)
  extract_stim_times(file_path, marker_name)   -> dict[str, list[float]]
  get_epoch_bounds(file_path)                  -> (pre_ms, post_ms)      [ext]
  units_assumed(file_path)                     -> bool                   [ext]
"""

from __future__ import annotations

import os
import threading

import numpy as np

from .epoched_mat import GUARD_MS, _stitch

#: MATLAB v7.3 is HDF5 behind a 512-byte user block: the file opens with the
#: ASCII banner "MATLAB 7.3 MAT-file ..." and the HDF5 signature sits at 512.
#: Testing for the signature at offset 0 finds nothing, which is what made the
#: first version of this reader decline every file it was written for.
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_HDF5_USERBLOCK = 512
_MAT73_BANNER = b"MATLAB 7.3"

#: Signal names the exported variable after the recording, so only the tail is
#: dependable.
_VAR_SUFFIX = "wave_data"

_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def _h5py():
    """Import h5py on demand, with a message that says what to do."""
    try:
        import h5py
    except ImportError as exc:                  # pragma: no cover - env dependent
        raise ImportError(
            "Reading CED Signal MATLAB exports needs h5py, because Signal "
            "writes MATLAB v7.3 files (HDF5).  Install it with:  "
            "pip install h5py"
        ) from exc
    return h5py


def _var_name(h):
    """The wave_data group in an open file, whatever it is called."""
    for key in h.keys():
        if key.startswith("#"):
            continue
        if key == _VAR_SUFFIX or key.endswith("_" + _VAR_SUFFIX):
            return key
    return None


def looks_like_mat73(file_path: str) -> bool:
    """True for any MATLAB v7.3 file, without needing h5py.

    Separated from is_signal_mat so a missing h5py can be reported as a
    missing package rather than as an unreadable file. The two are not the
    same thing and the analyst can only act on one of them.
    """
    if not str(file_path).lower().endswith(".mat"):
        return False
    try:
        with open(file_path, "rb") as fh:
            head = fh.read(_HDF5_USERBLOCK + len(_HDF5_MAGIC))
    except Exception:
        return False
    return bool(head.startswith(_MAT73_BANNER)
                or head[:len(_HDF5_MAGIC)] == _HDF5_MAGIC
                or head[_HDF5_USERBLOCK:_HDF5_USERBLOCK + len(_HDF5_MAGIC)]
                == _HDF5_MAGIC)


def h5py_available() -> bool:
    try:
        _h5py()
        return True
    except Exception:
        return False


def is_signal_mat(file_path: str) -> bool:
    """True if this is a Signal MATLAB export.

    Cheap and total: rejects on extension, then on the HDF5 magic bytes,
    before h5py is imported at all. Never raises.
    """
    if not str(file_path).lower().endswith(".mat"):
        return False
    if not looks_like_mat73(file_path):
        return False
    try:
        h5py = _h5py()
        with h5py.File(file_path, "r") as h:
            if _var_name(h) is None:
                return False
            grp = h[_var_name(h)]
            return all(k in grp for k in ("values", "interval", "points"))
    except Exception:
        return False


# ── Loading ──────────────────────────────────────────────────────────────────

def _codes_to_text(arr) -> str:
    """A MATLAB char array is uint16 code points."""
    try:
        return "".join(chr(int(c)) for c in np.asarray(arr).ravel()
                       if int(c) != 0).strip()
    except Exception:
        return ""


def _cell_of_text(h, node, n: int) -> list:
    """Read a MATLAB cell of strings, however it happens to be stored.

    A multi-element cell is an array of object references, one per string. A
    ONE-element cell is written as the char array itself, with no references
    at all -- which is what a single-channel Signal export produces. Treating
    that as references fails silently and the channel loses its name and unit,
    so both shapes are read here rather than assuming the common one.
    """
    try:
        arr = np.array(node)
    except Exception:
        return []
    if arr.dtype == object or str(arr.dtype).startswith("|O"):
        out = []
        for r in arr.ravel()[:n]:
            try:
                out.append(_codes_to_text(np.array(h[r])))
            except Exception:
                out.append("")
        return out
    # Inline char array: the whole thing is one string.
    return [_codes_to_text(arr)]


def _scalar(grp, key, cast=float):
    return cast(np.array(grp[key]).ravel()[0])


def _load(file_path: str) -> dict:
    """Parse the file once, then cache on (path, mtime)."""
    key = (os.path.abspath(file_path), os.path.getmtime(file_path))
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        return hit

    h5py = _h5py()
    with h5py.File(file_path, "r") as h:
        name = _var_name(h)
        if name is None:
            raise ValueError(
                f"{os.path.basename(file_path)}: no 'wave_data' variable, so "
                f"this is not a Signal export")
        w = h[name]

        values = np.asarray(w["values"], dtype=float)   # (nframe, nchan, npoint)
        if values.ndim != 3:
            raise ValueError(
                f"{os.path.basename(file_path)}: expected frames x channels x "
                f"points, found an array of shape {values.shape}")

        interval = _scalar(w, "interval")
        if interval <= 0:
            raise ValueError(
                f"{os.path.basename(file_path)}: sample interval is "
                f"{interval!r}, which cannot be a rate")
        start = _scalar(w, "start")

        nframe, nchan, npoint = values.shape

        def _per(group, field, n):
            try:
                return _cell_of_text(h, w[group][field], n)
            except Exception:
                return []

        titles = _per("chaninfo", "title", nchan)
        units = _per("chaninfo", "units", nchan)
        labels = _per("frameinfo", "label", nframe)

        # Frame start times are doubles behind references, not text.
        # Frame start times are doubles, stored the same two ways as the
        # strings above: references for many frames, inline for one.
        frame_starts = []
        try:
            node = np.array(w["frameinfo"]["start"])
            if node.dtype == object or str(node.dtype).startswith("|O"):
                frame_starts = [float(np.array(h[r]).ravel()[0])
                                for r in node.ravel()[:nframe]]
            else:
                frame_starts = [float(x) for x in node.ravel()[:nframe]]
        except Exception:
            frame_starts = []

    parsed = dict(
        values=values,
        fs=int(round(1.0 / interval)),
        start_s=start,
        n_frame=nframe, n_chan=nchan, n_point=npoint,
        titles=[t or f"Channel {i + 1}" for i, t in enumerate(titles)]
               or [f"Channel {i + 1}" for i in range(nchan)],
        units=units,
        labels=labels,
        frame_starts=frame_starts,
    )
    with _CACHE_LOCK:
        _CACHE[key] = parsed
    return parsed


def clear_cache():
    with _CACHE_LOCK:
        _CACHE.clear()


def _stim_index(parsed: dict) -> int:
    """Sample index of the trigger within each frame.

    ``start`` is the time of the first sample relative to the trigger, so a
    frame running -500 to +500 ms has its trigger 500 ms in. A frame that
    begins at or after the trigger has its stimulus at sample zero.
    """
    idx = int(round(-parsed["start_s"] * parsed["fs"]))
    return max(0, min(idx, parsed["n_point"] - 1))


# ── Public API ───────────────────────────────────────────────────────────────

def list_waveform_channels(file_path: str) -> list:
    parsed = _load(file_path)
    names = list(parsed["titles"])
    while len(names) < parsed["n_chan"]:
        names.append(f"Channel {len(names) + 1}")
    return names[:parsed["n_chan"]]


def get_epoch_bounds(file_path: str):
    """(pre_ms, post_ms) the frames actually contain."""
    parsed = _load(file_path)
    fs = parsed["fs"]
    idx = _stim_index(parsed)
    pre = idx / fs * 1000.0
    post = (parsed["n_point"] - idx) / fs * 1000.0
    return float(pre), float(post)


def units_assumed(file_path: str) -> bool:
    """False: the file states its unit per channel, so nothing is inferred."""
    return False


def get_unit(file_path: str, channel_idx: int = 0):
    parsed = _load(file_path)
    units = parsed["units"]
    if 0 <= channel_idx < len(units) and units[channel_idx]:
        return units[channel_idx]
    return None


def get_trial_count(file_path: str) -> int:
    return int(_load(file_path)["n_frame"])


def list_frame_states(file_path: str) -> list:
    """The distinct frame labels, in the order Signal reports them."""
    seen, out = set(), []
    for lab in _load(file_path)["labels"]:
        if lab and lab not in seen:
            seen.add(lab)
            out.append(lab)
    return out


def extract_emg_waveform_and_fs(file_path: str, channel_idx: int = 0):
    """One channel as a stitched pseudo-continuous trace.

    Frames are concatenated with guard bands rather than butted together, so
    a filter transient or an over-long window at the end of one trial cannot
    reach into the next.
    """
    parsed = _load(file_path)
    n_chan = parsed["n_chan"]
    if not 0 <= channel_idx < n_chan:
        raise IndexError(
            f"channel {channel_idx} is out of range; this file has "
            f"{n_chan} channel(s)")
    fs = parsed["fs"]
    guard_n = int(round(GUARD_MS * fs / 1000.0))
    epochs = parsed["values"][:, channel_idx, :]
    trace, _ = _stitch(epochs, _stim_index(parsed), guard_n)
    return trace, fs, get_unit(file_path, channel_idx)


def extract_stim_times(file_path: str, marker_name: str = "A") -> dict:
    """Stimulus times on the stitched time base, grouped by frame state.

    Signal's frame label is a per-trial grouping, so each distinct label
    becomes a StimType. A recording whose frames share one state yields a
    single group, which is the ordinary case.
    """
    parsed = _load(file_path)
    fs = parsed["fs"]
    guard_n = int(round(GUARD_MS * fs / 1000.0))
    _, stim_idxs = _stitch(parsed["values"][:, 0, :],
                           _stim_index(parsed), guard_n)

    labels = parsed["labels"]
    out: dict = {}
    for i, idx in enumerate(stim_idxs):
        lab = labels[i] if i < len(labels) and labels[i] else (marker_name or "A")
        out.setdefault(lab, []).append(idx / fs)
    return {k: sorted(v) for k, v in out.items()}
