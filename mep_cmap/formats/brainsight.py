"""
mep_cmap.formats.brainsight
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Brainsight neuronavigation EMG export reader (.txt).

Unlike a continuous recording, a Brainsight export is a stack of *already
epoched* TMS trials — one row per stimulation ("Sample").  The file is a
tab-delimited, multi-section text file; each section is introduced by a line
beginning with ``#``.  The section we read is headed ``# Sample Name`` and has
one row per stimulation.  Each row stores that trial's whole EMG epoch as a
semicolon-delimited string in the ``EMG Data 1`` (and optionally ``EMG Data 2``)
column, together with per-trial metadata:

  EMG Start / EMG End   epoch bounds (file time unit; here ms, e.g. -50..150)
  EMG Res.              sample resolution (file time unit; 0.333 ms -> 3000 Hz)
  EMG Channels          number of EMG channels used for that trial (1 or 2)
  Loc. X / Y / Z        neuronavigation target coordinate for that trial

Adapting epoched data to the continuous io.py contract
------------------------------------------------------
The per-trial epochs are concatenated back-to-back into a single pseudo-
continuous waveform, separated by a zero-filled guard band, and a stim time is
emitted at each epoch's pre-trigger point (``|EMG Start|``).  When the pipeline
re-epochs around those stim times it recovers each original trial.  The guard
band is zero-filled (not NaN) to stay filter-safe and consistent with the gap
handling in the LabChart / EDF readers; it prevents an over-wide analysis
window from bleeding into the neighbouring trial.

Public API (mirrors the io.py contract)
----------------------------------------
  is_brainsight(file_path)                     -> bool   (detection helper)
  list_waveform_channels(file_path)            -> list[str]
  extract_emg_waveform_and_fs(file_path, ch)   -> (np.ndarray, int, str|None)
  extract_stim_times(file_path, marker_name)   -> dict[str, list[float]]

Extra (outside the contract; no data loss for a future NIBS-BIDS path)
----------------------------------------------------------------------
  extract_coordinates(file_path)               -> (list[str], np.ndarray[N,3])

Units / sample rate
-------------------
EMG data are returned in their native unit (no rescaling); the unit string
('µV' / 'mV' / 'V') is derived from the ``# Units:`` preamble line.  The sample
rate is read directly from ``EMG Res.`` and cross-checked against
(frames - 1) / (EMG End - EMG Start); a clear error is raised if trials
disagree — the rate is never inferred or guessed.
"""

import numpy as np

# ── Detection helper ──────────────────────────────────────────────────────────

def is_brainsight(file_path: str) -> bool:
    """
    True if the file looks like a Brainsight export: a ``# Version:`` line at
    the top plus a ``Brainsight`` created-by line or a ``# Sample Name`` header
    within the first several lines.  Cheap: reads only the header preamble.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            head = [f.readline() for _ in range(15)]
    except Exception:
        return False
    if not head:
        return False
    if not head[0].startswith('# Version:'):
        return False
    joined = ''.join(head)
    return ('Brainsight' in joined) or ('# Sample Name' in joined)


# ── Internal parse ────────────────────────────────────────────────────────────

def _time_convert_to_seconds(units_line: str) -> float:
    """Factor to convert the file's time unit to seconds, from the Units line."""
    u = units_line.lower()
    if 'millisecond' in u:
        return 1e-3
    if 'microsecond' in u:
        return 1e-6
    return 1.0  # seconds


def _emg_unit_string(units_line: str) -> str:
    """Return the native EMG unit string ('µV' / 'mV' / 'V') from Units line."""
    u = units_line.lower()
    if 'microvolt' in u:
        return '\u00b5V'   # µV (U+00B5, matching pipeline.py)
    if 'millivolt' in u:
        return 'mV'
    return 'V'


def _parse_semicolon(cell: str) -> np.ndarray:
    """Parse a ';'-delimited numeric string into a float ndarray."""
    if cell is None:
        return np.empty(0, dtype=float)
    vals = [p for p in cell.strip().split(';') if p.strip() != '']
    if not vals:
        return np.empty(0, dtype=float)
    return np.array([float(v) for v in vals], dtype=float)


def _parse(file_path: str) -> dict:
    """
    Parse the Brainsight file once and return a structured dict shared by all
    public functions:

      fs            : int
      emg_unit      : str
      trials        : list of dicts {name, session, pre_s, ch1, ch2|None, loc}
      channels      : list[str]   channel titles present
      offsets       : list[int]   start sample of each trial in the stream
      total_len     : int         total length of the concatenated stream
      guard         : int         zero-fill guard band length (samples)
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # Units line (preamble)
    units_line = next((l for l in lines if l.startswith('# Units')), '')
    time_conv = _time_convert_to_seconds(units_line)
    emg_unit = _emg_unit_string(units_line)

    # Locate the '# Sample Name' header row
    hdr_idx = next((i for i, l in enumerate(lines)
                    if l.lstrip().startswith('# Sample Name')), None)
    if hdr_idx is None:
        raise ValueError("Brainsight: '# Sample Name' section not found.")

    header = [h.strip() for h in lines[hdr_idx].rstrip('\n').split('\t')]
    # Strip the leading '# ' from the first column name
    if header and header[0].startswith('#'):
        header[0] = header[0].lstrip('#').strip()

    def col(name):
        return header.index(name) if name in header else None

    c_name = col('Sample Name')
    c_sess = col('Session Name')
    c_start = col('EMG Start')
    c_end = col('EMG End')
    c_res = col('EMG Res.')
    c_nch = col('EMG Channels')
    c_lx, c_ly, c_lz = col('Loc. X'), col('Loc. Y'), col('Loc. Z')
    c_d1 = col('EMG Data 1')
    c_d2 = col('EMG Data 2')

    if c_d1 is None or c_start is None or c_end is None or c_res is None:
        raise ValueError("Brainsight: required EMG columns are missing.")

    # Data rows: from header+1 until the next '#' section (or EOF)
    trials = []
    fs_seen = []
    for l in lines[hdr_idx + 1:]:
        if l.lstrip().startswith('#'):
            break
        if not l.strip():
            continue
        row = l.rstrip('\n').split('\t')
        if len(row) <= c_d1 or not row[c_name].strip():
            continue

        def g(idx):
            return row[idx] if (idx is not None and idx < len(row)) else ''

        try:
            emg_start = float(g(c_start)) * time_conv     # seconds
            emg_end = float(g(c_end)) * time_conv          # seconds
            emg_res = float(g(c_res)) * time_conv          # seconds/sample
        except ValueError:
            continue
        if emg_res <= 0:
            continue

        ch1 = _parse_semicolon(g(c_d1))
        if ch1.size == 0:
            continue
        ch2_raw = _parse_semicolon(g(c_d2)) if c_d2 is not None else np.empty(0)
        ch2 = ch2_raw if ch2_raw.size > 0 else None

        # Sample rate from EMG Res., cross-checked against frame count
        fs_res = 1.0 / emg_res
        dur = emg_end - emg_start
        fs_frames = (ch1.size - 1) / dur if dur > 0 else fs_res
        if abs(fs_res - fs_frames) / fs_res > 0.02:   # >2% disagreement
            raise ValueError(
                f"Brainsight: sample-rate mismatch in trial {row[c_name]!r} "
                f"(EMG Res. -> {fs_res:.1f} Hz, frames -> {fs_frames:.1f} Hz).")
        fs_seen.append(fs_res)

        loc = np.array([
            float(g(c_lx)) if g(c_lx).strip() else np.nan,
            float(g(c_ly)) if g(c_ly).strip() else np.nan,
            float(g(c_lz)) if g(c_lz).strip() else np.nan,
        ], dtype=float)

        trials.append(dict(
            name=row[c_name].strip(),
            session=(g(c_sess).strip() if c_sess is not None else ''),
            pre_s=abs(emg_start),
            ch1=ch1, ch2=ch2, loc=loc,
        ))

    if not trials:
        raise ValueError("Brainsight: no EMG sample rows found.")

    # Validate a single consistent fs across all trials
    fs_arr = np.array(fs_seen)
    if fs_arr.std() / fs_arr.mean() > 0.02:
        raise ValueError("Brainsight: inconsistent sample rate across trials.")
    fs = int(round(float(fs_arr.mean())))

    # Which channels exist? ch2 only listed if present in at least one trial
    has_ch2 = any(t['ch2'] is not None for t in trials)
    channels = ['EMG Data 1'] + (['EMG Data 2'] if has_ch2 else [])

    # Layout: [epoch][zero guard] per trial; guard = max epoch length
    lengths = [t['ch1'].size for t in trials]
    guard = max(lengths)
    offsets = []
    cursor = 0
    for L in lengths:
        offsets.append(cursor)
        cursor += L + guard
    total_len = cursor

    return dict(fs=fs, emg_unit=emg_unit, trials=trials, channels=channels,
                offsets=offsets, total_len=total_len, guard=guard)


# ── Public API ────────────────────────────────────────────────────────────────

def list_waveform_channels(file_path: str) -> list:
    """Return the EMG channel titles present in the file."""
    return _parse(file_path)['channels']


def extract_emg_waveform_and_fs(file_path: str, channel_idx: int = 0):
    """
    Concatenate all trial epochs of one channel into a pseudo-continuous
    waveform (zero-filled guard band between trials).

    Returns
    -------
    emg  : np.ndarray  concatenated waveform (native unit, no rescaling)
    fs   : int         sampling rate in Hz
    unit : str | None  'µV' / 'mV' / 'V'
    """
    p = _parse(file_path)
    nchan = len(p['channels'])
    if not (0 <= channel_idx < nchan):
        raise IndexError(
            f"channel_idx {channel_idx} out of range (0..{nchan - 1})")

    output = np.zeros(p['total_len'], dtype=float)
    for off, t in zip(p['offsets'], p['trials']):
        L = t['ch1'].size
        if channel_idx == 0:
            seg = t['ch1']
        else:
            # ch2: zero-fill this trial's slot if missing (preserve alignment)
            seg = t['ch2'] if t['ch2'] is not None else np.zeros(L, dtype=float)
            if seg.size != L:                       # length-align to ch1
                fixed = np.zeros(L, dtype=float)
                n = min(L, seg.size)
                fixed[:n] = seg[:n]
                seg = fixed
        output[off:off + L] = seg

    return output, p['fs'], p['emg_unit']


def extract_stim_times(file_path: str, marker_name: str = 'A') -> dict:
    """
    Return one stim time per trial at its pre-trigger point, all grouped under
    a single label.

    Parameters
    ----------
    marker_name : used as the stim-type label (kept as-is so a string label
                  survives); defaults to 'A'.

    Returns
    -------
    dict mapping label -> list of absolute timestamps (seconds), on the same
    origin as extract_emg_waveform_and_fs.
    """
    p = _parse(file_path)
    fs = p['fs']
    label = marker_name.strip() if marker_name and marker_name.strip() else 'A'

    times = []
    for off, t in zip(p['offsets'], p['trials']):
        stim_sample = off + int(round(t['pre_s'] * fs))
        times.append(stim_sample / fs)

    return {label: times} if times else {}


def extract_coordinates(file_path: str):
    """
    Return per-trial neuronavigation target coordinates.

    Outside the io.py contract — preserved for a future NIBS-BIDS coordinate
    sidecar so the spatial targeting data is not lost.

    Returns
    -------
    names  : list[str]        per-trial sample names
    coords : np.ndarray[N,3]  Loc. X / Y / Z per trial (NaN where absent)
    """
    p = _parse(file_path)
    names = [t['name'] for t in p['trials']]
    coords = np.vstack([t['loc'] for t in p['trials']]) if p['trials'] \
        else np.empty((0, 3))
    return names, coords
