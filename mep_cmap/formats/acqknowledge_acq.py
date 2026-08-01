"""
mep_cmap.formats.acqknowledge_acq
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BIOPAC AcqKnowledge raw binary reader (.acq), via the ``bioread`` package.

``bioread`` is the de-facto Python reader for BIOPAC .acq files and has been
verified here against real AcqKnowledge files spanning versions 3.8 through
5.0.  It is pure-Python (NumPy only), so it is PyInstaller-friendly and adds no
compiled/vendor-binary dependency.

AcqKnowledge allows *per-channel* sample rates, so ``extract_emg_waveform_and_fs``
returns the selected channel's own rate rather than a single global rate.

Stimulation times
-----------------
Two sources, tried in order:
  1. A channel whose name contains 'stim', 'trig', or 'ttl' — threshold-crossed
     (consistent with the AcqKnowledge .mat reader and the typical TMS analogue
     stim channel).
  2. Event markers, excluding segment/boundary markers ('Append' etc.),
     grouped by marker text.

NOTE: the .acq waveform / rate / unit path is validated against real files, but
the stim-detection path could only be exercised on non-TMS demo files (no stim
events).  Stim extraction from real TMS .acq recordings should be stress-tested
on genuine data.

Public API (mirrors the io.py contract)
----------------------------------------
  list_waveform_channels(file_path)            -> list[str]
  extract_emg_waveform_and_fs(file_path, ch)   -> (np.ndarray, int, str|None)
  extract_stim_times(file_path, marker_name)   -> dict[str, list[float]]

Dependency: ``bioread`` (``pip install bioread``).  If it is not installed, the
functions raise a clear ImportError-style message rather than failing at import
time, so the rest of the package still loads.
"""

import re

import numpy as np

try:
    import bioread as _bioread
    _BIOREAD_AVAILABLE = callable(getattr(_bioread, 'read_file', None))
except ImportError:
    _bioread = None
    _BIOREAD_AVAILABLE = False

_STIM_KEYS = ('stim', 'trig', 'ttl')
# Structural segment/append boundary markers, not stimulation events.  Older
# AcqKnowledge versions leave type/type_code empty, so we also match the
# auto-generated "Segment N" text.
_BOUNDARY_MARKER_TYPES = {'append', 'segment'}
_BOUNDARY_MARKER_CODES = {'apnd', 'seg'}
_SEGMENT_TEXT = re.compile(r'^\s*segment\s*\d*\s*$', re.IGNORECASE)


def _is_boundary_marker(mk) -> bool:
    """True if a marker is a segment/append boundary (across AcqKnowledge versions)."""
    tc = str(getattr(mk, 'type_code', '') or '').strip().lower()
    ty = str(getattr(mk, 'type', '') or '').strip().lower()
    txt = str(getattr(mk, 'text', '') or '').strip()
    return (tc in _BOUNDARY_MARKER_CODES
            or ty in _BOUNDARY_MARKER_TYPES
            or bool(_SEGMENT_TEXT.match(txt)))


def _require_bioread():
    if not _BIOREAD_AVAILABLE:
        raise ImportError(
            "Reading AcqKnowledge .acq files requires the 'bioread' package. "
            "Install it with:  pip install bioread")


def _norm_unit(u):
    """Normalise a unit string to the codebase's short forms; keep unknowns."""
    if not u:
        return None
    s = str(u).strip()
    table = {
        'volts': 'V', 'volt': 'V', 'v': 'V',
        'millivolts': 'mV', 'millivolt': 'mV', 'mv': 'mV',
        'microvolts': '\u00b5V', 'microvolt': '\u00b5V',
        'uv': '\u00b5V', '\u00b5v': '\u00b5V', '\u03bcv': '\u00b5V',
    }
    return table.get(s.lower(), s)


def _read(file_path: str):
    _require_bioread()
    try:
        return _bioread.read_file(file_path)
    except Exception as exc:
        raise ValueError(
            f"Could not read AcqKnowledge .acq file ({file_path!r}): {exc}"
        ) from exc


# ── Public API ────────────────────────────────────────────────────────────────

def list_waveform_channels(file_path: str) -> list:
    """Return channel names."""
    df = _read(file_path)
    return [c.name for c in df.channels] or ['Channel 1']


def extract_emg_waveform_and_fs(file_path: str, channel_idx: int = 0):
    """
    Return one channel's waveform, its own sample rate, and native unit.

    Returns
    -------
    emg  : np.ndarray  waveform (native unit, no rescaling)
    fs   : int         that channel's sampling rate in Hz
    unit : str | None  normalised unit string
    """
    df = _read(file_path)
    nchan = len(df.channels)
    if not (0 <= channel_idx < nchan):
        raise IndexError(
            f"channel_idx {channel_idx} out of range (0..{nchan - 1})")
    ch = df.channels[channel_idx]
    fs = int(round(float(ch.samples_per_second)))
    return np.asarray(ch.data, dtype=float), fs, _norm_unit(ch.units)


def extract_stim_times(file_path: str, marker_name: str = 'A') -> dict:
    """
    Return stim times, preferring a named stim/trigger channel (threshold-
    crossed), falling back to non-boundary event markers grouped by text.

    Returns
    -------
    dict mapping label -> list of timestamps (seconds).
    """
    df = _read(file_path)
    label = marker_name.strip() if marker_name and marker_name.strip() else 'A'

    # 1) Named stim/trigger channel → threshold crossing
    stim_ch = next(
        (c for c in df.channels
         if any(k in (c.name or '').lower() for k in _STIM_KEYS)), None)
    if stim_ch is not None:
        stim = np.asarray(stim_ch.data, dtype=float)
        fs = float(stim_ch.samples_per_second)
        if stim.size and stim.max() > stim.min() and fs > 0:
            threshold = stim.min() + (stim.max() - stim.min()) * 0.5
            rising = np.where(np.diff((stim >= threshold).astype(int)) == 1)[0]
            times = [(idx + 1) / fs for idx in rising]
            if times:
                return {label: times}

    # 2) Event markers (excluding structural boundaries), grouped by text
    markers = getattr(df, 'event_markers', None) or []
    base_fs = float(getattr(df, 'samples_per_second', 0) or 0)
    if markers and base_fs > 0:
        out = {}
        for mk in markers:
            if _is_boundary_marker(mk):
                continue
            si = getattr(mk, 'sample_index', None)
            if si is None:
                continue
            txt = str(getattr(mk, 'text', '') or '').strip() or label
            out.setdefault(txt, []).append(float(si) / base_fs)
        for k in out:
            out[k].sort()
        if out:
            return out

    return {}
