"""
mep_cmap.formats.labchart_mat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LabChart MATLAB (.mat) export reader.

LabChart's "Save As… → MATLAB" export stores the whole recording in a flat
``data`` vector that is indexed per channel/block by ``datastart``/``dataend``.
This is a *different* format from the LabChart text export handled by
``labchart.py`` — same acquisition system, different file — so it lives in its
own module and is detected separately (``detect_format`` returns
``'labchart_mat'``).

Multiple recording blocks are concatenated into a single continuous waveform
using ``blocktimes`` (MATLAB serial date numbers) for absolute placement; gaps
between blocks are zero-filled.  This mirrors the block-placement logic of the
text reader so the two behave identically downstream.

Stimulation events come from the ``com`` (comment) table rather than a stim
channel.  Each comment carries a block, a within-block tick position, a type
(1 = user comment, 2 = event marker — both are treated as stim events), and an
index into ``comtext``.  Stim times are grouped by their comment-text label.

Public API (mirrors the io.py contract)
----------------------------------------
  is_labchart_mat(file_path)                   -> bool   (detection helper)
  list_waveform_channels(file_path)            -> list[str]
  extract_emg_waveform_and_fs(file_path, ch)   -> (np.ndarray, int, str|None)
  extract_stim_times(file_path, marker_name)   -> dict[str, list[float]]

Notes on extract_stim_times
----------------------------
  The returned dict is keyed by the actual comment-text labels found in the
  file (e.g. {'Trigger': [...]}).  If ``marker_name`` is a non-empty string
  that matches one of those labels (case-insensitive), only that label is
  returned; otherwise all labels are returned.

LabChart .mat schema (from the ADInstruments MATLAB-export specification)
-------------------------------------------------------------------------
  data          1 x N   flat sample vector (all channels, all blocks)
  datastart     C x B   1-based start index into ``data`` per channel/block
  dataend       C x B   1-based (inclusive) end index into ``data``
  titles        C-row char array of channel names
  samplerate    C x B   sample rate (Hz) per channel/block (0 = empty channel)
  tickrate      1 x B   max sample rate per block (tick base for comments)
  blocktimes    1 x B   MATLAB serial date number of each block's first sample
  unittext      char rows of unit strings (e.g. 'V', 'mV')
  unittextmap   C x B   1-based index into ``unittext`` per channel/block
  com           K x 5   [channel, block, tick position, type, comtext index]
  comtext       char rows of comment-text strings

Dependency
----------
SciPy's ``scipy.io.loadmat`` (already a project dependency).  LabChart writes
classic (v5/v7) MAT-files, not the HDF5-based v7.3 format, so ``loadmat`` is
sufficient; a clear error is raised if the file cannot be parsed.
"""

import numpy as np
from scipy.io import loadmat, whosmat

# Variables that uniquely identify a LabChart MATLAB export.
_SIGNATURE = {'data', 'datastart', 'dataend', 'titles', 'samplerate'}

# MATLAB serial date number → seconds
_DAY_SECONDS = 86400.0


# ── Detection helper ──────────────────────────────────────────────────────────

def is_labchart_mat(file_path: str) -> bool:
    """
    Return True if the file is a MAT-file containing the LabChart export
    signature variables.  Uses ``whosmat`` so the (potentially large) ``data``
    vector is never loaded during detection.  Any read error → False.
    """
    try:
        names = {entry[0] for entry in whosmat(file_path)}
    except Exception:
        return False
    return _SIGNATURE.issubset(names)


# ── Internal load + small parsing helpers ─────────────────────────────────────

def _load(file_path: str) -> dict:
    try:
        return loadmat(file_path)
    except Exception as exc:
        raise ValueError(
            f"Could not read LabChart .mat file "
            f"({file_path!r}): {exc}"
        ) from exc


def _char_rows(arr) -> list:
    """
    Normalise a MATLAB char-array field (titles / unittext / comtext) into a
    list of stripped Python strings, tolerating the several shapes SciPy may
    return (1-D array of str, 2-D char matrix, or a bare string).
    """
    if arr is None:
        return []
    a = np.asarray(arr)
    if a.dtype.kind in ('U', 'S'):
        if a.ndim == 0:
            return [str(a).strip()]
        return [str(x).strip() for x in a.ravel()]
    # Fallback: object/cell array
    out = []
    for x in a.ravel():
        try:
            out.append(str(x).strip())
        except Exception:
            out.append('')
    return out


def _first_block(mat: dict, ch: int) -> int:
    """Index of the first block in which channel ``ch`` has data (sr > 0)."""
    sr = np.asarray(mat['samplerate'], dtype=float)
    nblocks = sr.shape[1]
    for b in range(nblocks):
        if sr[ch, b] > 0:
            return b
    return 0


# ── Public API ────────────────────────────────────────────────────────────────

def list_waveform_channels(file_path: str) -> list:
    """Return channel names from ``titles``."""
    mat = _load(file_path)
    titles = _char_rows(mat.get('titles'))
    nchan = int(np.asarray(mat['datastart']).shape[0])
    if len(titles) < nchan:
        titles += [f'Channel {i + 1}' for i in range(len(titles), nchan)]
    return titles[:nchan] if titles else ['Channel 1']


def extract_emg_waveform_and_fs(file_path: str, channel_idx: int = 0):
    """
    Concatenate all blocks of one channel into a continuous waveform.

    Blocks are placed at their absolute positions derived from ``blocktimes``;
    gaps between blocks are zero-filled.  Data are returned in their native
    unit (no V/mV rescaling); the unit string is returned separately.

    Returns
    -------
    emg  : np.ndarray  concatenated waveform
    fs   : int         sampling rate in Hz
    unit : str | None  unit string (e.g. 'V'), or None if unavailable
    """
    mat = _load(file_path)
    data = np.asarray(mat['data'], dtype=float).ravel()
    datastart = np.asarray(mat['datastart'], dtype=float)
    dataend = np.asarray(mat['dataend'], dtype=float)
    samplerate = np.asarray(mat['samplerate'], dtype=float)
    blocktimes = np.asarray(mat['blocktimes'], dtype=float).ravel()
    nchan, nblocks = datastart.shape

    if not (0 <= channel_idx < nchan):
        raise IndexError(
            f"channel_idx {channel_idx} out of range (0..{nchan - 1})")

    b0 = _first_block(mat, channel_idx)
    fs = int(round(samplerate[channel_idx, b0]))
    if fs <= 0:
        raise ValueError("No valid sample rate for the selected channel.")

    # Unit for this channel (first non-empty block)
    unit = None
    unittext = _char_rows(mat.get('unittext'))
    umap = mat.get('unittextmap')
    if unittext and umap is not None:
        try:
            ui = int(np.asarray(umap, dtype=float)[channel_idx, b0]) - 1
            if 0 <= ui < len(unittext) and unittext[ui]:
                unit = unittext[ui].strip('*') or None
        except Exception:
            unit = unittext[0] if unittext else None

    t0 = blocktimes[b0] if b0 < len(blocktimes) else 0.0

    # First pass: compute placement and total length
    placements = []  # (offset_samples, segment)
    max_end = 0
    for b in range(nblocks):
        if samplerate[channel_idx, b] <= 0:
            continue
        s = int(datastart[channel_idx, b])
        e = int(dataend[channel_idx, b])
        if s <= 0 or e < s or e > data.size:
            continue
        segment = data[s - 1:e]           # MATLAB 1-based inclusive → Python
        bt = blocktimes[b] if b < len(blocktimes) else t0
        offset = int(round((bt - t0) * _DAY_SECONDS * fs))
        if offset < 0:
            offset = 0
        placements.append((offset, segment))
        max_end = max(max_end, offset + segment.size)

    if not placements:
        raise ValueError("No LabChart data blocks found for this channel.")

    output = np.zeros(max_end, dtype=float)
    for offset, segment in placements:
        output[offset:offset + segment.size] = segment

    return output, fs, unit


def extract_stim_times(file_path: str, marker_name: str = '') -> dict:
    """
    Return stimulation times from the ``com`` comment table, grouped by the
    comment-text label.  Both user comments (type 1) and event markers
    (type 2) are included.

    Times are absolute seconds on the same origin as
    ``extract_emg_waveform_and_fs`` (block 0's start = 0).

    Parameters
    ----------
    marker_name : optional label filter.  If it matches one of the file's
                  comment-text labels (case-insensitive), only that label is
                  returned; otherwise all labels are returned.

    Returns
    -------
    dict mapping comment-text label -> list of absolute timestamps (seconds)
    """
    mat = _load(file_path)
    com = mat.get('com')
    comtext = _char_rows(mat.get('comtext'))
    if com is None or np.asarray(com).size == 0 or not comtext:
        return {}

    com = np.asarray(com, dtype=float)
    if com.ndim == 1:
        com = com.reshape(1, -1)

    samplerate = np.asarray(mat['samplerate'], dtype=float)
    tickrate = np.asarray(mat['tickrate'], dtype=float).ravel()
    blocktimes = np.asarray(mat['blocktimes'], dtype=float).ravel()

    # Origin = first block that actually holds data on any channel
    valid_blocks = np.where((samplerate > 0).any(axis=0))[0]
    b0 = int(valid_blocks[0]) if valid_blocks.size else 0
    t0 = blocktimes[b0] if b0 < len(blocktimes) else 0.0

    out = {}
    for row in com:
        block = int(row[1]) - 1                    # 1-based → 0-based
        pos_ticks = float(row[2])
        txt_idx = int(row[4]) - 1                   # 1-based → 0-based
        if not (0 <= txt_idx < len(comtext)):
            continue
        if 0 <= block < len(tickrate) and tickrate[block] > 0:
            tr = tickrate[block]
        else:
            tr = float(samplerate[0, max(block, 0)]) or 1.0
        bt = blocktimes[block] if 0 <= block < len(blocktimes) else t0
        abs_time = (bt - t0) * _DAY_SECONDS + (pos_ticks / tr)
        label = comtext[txt_idx]
        out.setdefault(label, []).append(abs_time)

    for label in out:
        out[label].sort()

    # Optional label filter
    if marker_name:
        want = marker_name.strip().lower()
        match = {k: v for k, v in out.items() if k.lower() == want}
        if match:
            return match

    return out
