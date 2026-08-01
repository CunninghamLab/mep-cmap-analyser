"""
mep_cmap.formats.acqknowledge_mat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BIOPAC AcqKnowledge "Save as MATLAB" export reader (.mat).

This is the .mat that AcqKnowledge itself writes (File → Save As → MATLAB), not
a raw .acq (see ``acqknowledge_acq.py`` for that).  It is a flat, single-block
continuous recording with a simple schema:

  data          [samples x channels]  sample matrix
  labels        char rows              channel names
  units         char rows              per-channel unit strings ('mV', 'Volts')
  isi           scalar                 inter-sample interval
  isi_units     char                   'ms' | 's' | 'us'/'µs'
  start_sample  scalar                 (unused)

There are no embedded event markers in this export, so stimulation times are
detected by threshold-crossing a channel whose name contains 'stim', 'trig',
or 'ttl' (AcqKnowledge TMS recordings include a dedicated analogue stim
channel, e.g. a 0-5 V TTL named 'Stim').

Public API (mirrors the io.py contract)
----------------------------------------
  is_acqknowledge_mat(file_path)               -> bool   (detection helper)
  list_waveform_channels(file_path)            -> list[str]
  extract_emg_waveform_and_fs(file_path, ch)   -> (np.ndarray, int, str|None)
  extract_stim_times(file_path, marker_name)   -> dict[str, list[float]]

Units are returned in their native form (no rescaling); long spellings are
normalised to the codebase's short forms ('Volts' -> 'V', etc.).

Dependency: SciPy's ``scipy.io.loadmat`` (already a project dependency).
"""

import numpy as np
from scipy.io import loadmat, whosmat

# Variables that identify an AcqKnowledge MATLAB export (and distinguish it
# from a LabChart .mat, which instead has datastart/dataend/titles).
_SIGNATURE = {'data', 'isi', 'isi_units', 'labels', 'units'}

_STIM_KEYS = ('stim', 'trig', 'ttl')


# ── Detection helper ──────────────────────────────────────────────────────────

def is_acqknowledge_mat(file_path: str) -> bool:
    """True if the MAT-file carries the AcqKnowledge export signature."""
    try:
        names = {entry[0] for entry in whosmat(file_path)}
    except Exception:
        return False
    return _SIGNATURE.issubset(names)


# ── Small helpers ─────────────────────────────────────────────────────────────

def _load(file_path: str) -> dict:
    try:
        return loadmat(file_path)
    except Exception as exc:
        raise ValueError(
            f"Could not read AcqKnowledge .mat file ({file_path!r}): {exc}"
        ) from exc


def _char_rows(arr) -> list:
    """Normalise a MATLAB char field into a list of stripped strings."""
    if arr is None:
        return []
    a = np.asarray(arr)
    if a.dtype.kind in ('U', 'S'):
        if a.ndim == 0:
            return [str(a).strip()]
        return [str(x).strip() for x in a.ravel()]
    out = []
    for x in a.ravel():
        try:
            out.append(str(x).strip())
        except Exception:
            out.append('')
    return out


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


def _fs_from_isi(mat: dict) -> int:
    isi = float(np.asarray(mat['isi']).ravel()[0])
    iu = _char_rows(mat.get('isi_units'))
    iu = iu[0].lower() if iu else 's'
    factor = {'ms': 1e-3, 's': 1.0, 'sec': 1.0,
              'us': 1e-6, '\u00b5s': 1e-6, '\u03bcs': 1e-6}.get(iu, 1.0)
    if isi <= 0:
        raise ValueError("AcqKnowledge: non-positive inter-sample interval.")
    return int(round(1.0 / (isi * factor)))


def _labels(mat: dict, nchan: int) -> list:
    labels = _char_rows(mat.get('labels'))
    if len(labels) < nchan:
        labels += [f'Channel {i + 1}' for i in range(len(labels), nchan)]
    return labels[:nchan]


# ── Public API ────────────────────────────────────────────────────────────────

def list_waveform_channels(file_path: str) -> list:
    """Return channel names from ``labels``."""
    mat = _load(file_path)
    nchan = int(np.asarray(mat['data']).shape[1])
    return _labels(mat, nchan) or ['Channel 1']


def extract_emg_waveform_and_fs(file_path: str, channel_idx: int = 0):
    """
    Return one channel's continuous waveform, sample rate, and native unit.

    Returns
    -------
    emg  : np.ndarray  waveform (native unit, no rescaling)
    fs   : int         sampling rate in Hz
    unit : str | None  normalised unit string (e.g. 'mV', 'V')
    """
    mat = _load(file_path)
    data = np.asarray(mat['data'], dtype=float)
    nchan = data.shape[1]
    if not (0 <= channel_idx < nchan):
        raise IndexError(
            f"channel_idx {channel_idx} out of range (0..{nchan - 1})")

    fs = _fs_from_isi(mat)
    units = _char_rows(mat.get('units'))
    unit = _norm_unit(units[channel_idx]) if channel_idx < len(units) else None
    return data[:, channel_idx].copy(), fs, unit


def extract_stim_times(file_path: str, marker_name: str = 'A') -> dict:
    """
    Detect stim times by threshold-crossing the named stim/trigger channel.

    Parameters
    ----------
    marker_name : used as the stim-type label (default 'A').

    Returns
    -------
    dict mapping label -> list of timestamps (seconds).
    """
    mat = _load(file_path)
    data = np.asarray(mat['data'], dtype=float)
    fs = _fs_from_isi(mat)
    nchan = data.shape[1]
    labels = _labels(mat, nchan)
    label = marker_name.strip() if marker_name and marker_name.strip() else 'A'

    stim_idx = next(
        (i for i, l in enumerate(labels)
         if any(k in l.lower() for k in _STIM_KEYS)), None)
    if stim_idx is None:
        return {}

    stim = data[:, stim_idx]
    if stim.size == 0 or stim.max() <= stim.min():
        return {}
    threshold = stim.min() + (stim.max() - stim.min()) * 0.5
    rising = np.where(np.diff((stim >= threshold).astype(int)) == 1)[0]
    times = [(idx + 1) / fs for idx in rising]
    return {label: times} if times else {}
