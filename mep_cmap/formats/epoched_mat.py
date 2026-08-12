"""
mep_cmap.formats.epoched_mat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reader for **pre-epoched** TMS-EMG MATLAB exports (.mat).

Unlike every other reader in this package, the file does not contain a
continuous recording — it contains trials that have already been cut around
the stimulus.  Schema (MATLAB v5, verified against OpenNeuro ds002094
sourcedata; the writing software is not identified in the file):

  Chans.SamplingRate   scalar Hz
  Chans.Label          (nchan,) cell of channel names, e.g. {'FDI','ADM'}
  Chans.Data           (ntrial, nsample, nchan) double
  Chans.t              (nsample,) time axis in ms, stimulus at t = 0
  Markers.Codes        (ntrial,) marker code per trial
  Markers.Times        (ntrial,) marker time (native units)
  Conds.Label          condition name, e.g. 'All'
  Conds.Code           condition code
  Conds.Trials         (ntrial,) trial index
  Conds.TrialCodes     (ntrial,) per-trial condition code  -> stim types
  Events.Label         event name, e.g. 'TMS pulse'
  Events.Code          (ntrial,) event code
  Events.Times         (ntrial,) event time in ms **relative to Chans.t**
  CenterTimes          (ntrial,) stimulus offset into the epoch, 0.01 ms units

Epoch stitching
---------------
The io.py contract is continuous-waveform + stim-times-in-seconds, so this
reader synthesises a pseudo-continuous trace:

    [lead guard] epoch_0 [gap guard] epoch_1 [gap guard] ... [trail guard]

Guard bands are **not data**.  They exist so that (a) filter transients decay
before reaching an analysed sample, and (b) an over-long analysis window can
never reach backwards out of one epoch into the previous trial's MEP — which
would silently contaminate the baseline used for bootstrap onset thresholds
and RMS outlier gating.

Each guard is built by mirror-tiling the *pre-stimulus baseline* of the
adjacent epoch(s), DC-anchored to join continuously, and cosine-crossfaded
across the gap.  Baseline-like padding (rather than zeros or a DC hold) keeps
the amplitude distribution realistic and avoids the step discontinuities that
would ring through a 20-450 Hz bandpass.

Callers must still clamp their analysis windows to ``get_epoch_bounds()``;
the guard is a safety net, not a licence to analyse fabricated samples.

Units
-----
The format carries no unit field.  ``suggest_unit()`` infers one from the ADC
quantisation grid plus physiological amplitude plausibility, and the caller
confirms it once via a dialog; the answer is stored in a sidecar.  Choosing
"unknown" passes the waveform through unscaled and sets ``units_assumed`` so
downstream output can record that amplitudes are not verified millivolts.

Sidecar file
------------
  <data_file_stem>.epoched_config.json
  { "unit": "mV" | "V" | "uV" | "unknown" }

Public API (mirrors the io.py contract, plus two extensions)
-------------------------------------------------------------
  is_epoched_mat(file_path)                    -> bool
  has_config(file_path)                        -> bool
  load_config(file_path)                       -> dict
  save_config(file_path, unit)                 -> None
  suggest_unit(file_path)                      -> (str, list[str])
  list_waveform_channels(file_path)            -> list[str]
  extract_emg_waveform_and_fs(file_path, ch)   -> (np.ndarray, int, str|None)
  extract_stim_times(file_path, marker_name)   -> dict[str, list[float]]
  get_epoch_bounds(file_path)                  -> (pre_ms, post_ms)      [ext]
  get_clipped_trials(file_path, ch)            -> list[int]              [ext]

Dependency: SciPy's ``scipy.io.loadmat`` (already a project dependency).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat

# Variables that identify this export.  Deliberately strict: all five must be
# present, which distinguishes it from the LabChart .mat (datastart/dataend/
# titles) and the AcqKnowledge .mat (data/isi/isi_units/labels/units).
_SIGNATURE = {'Chans', 'Markers', 'Conds', 'Events', 'CenterTimes'}

# Guard-band length either side of every epoch, in milliseconds.  Sized to
# exceed the largest default analysis window (post_ms = 400) so that even an
# unclamped configuration cannot read across an epoch boundary into real data
# belonging to another trial.
GUARD_MS = 300.0

# Physiologically plausible peak-to-peak range for a surface-EMG MEP, in mV.
# Used only to disambiguate the unit; deliberately generous.
_PLAUSIBLE_MEP_PTP_MV = (0.05, 20.0)

_UNIT_CHOICES = ('mV', 'V', 'uV', 'unknown')


# ── Detection helper ──────────────────────────────────────────────────────────

def is_epoched_mat(file_path: str) -> bool:
    """True if the MAT-file carries the epoched-export signature.

    Cheap: reads the variable directory only, never the sample data.
    """
    try:
        names = {entry[0] for entry in whosmat(file_path)}
    except Exception:
        return False
    return _SIGNATURE.issubset(names)


# ── Sidecar config helpers (pattern mirrors spike2_smr.py) ────────────────────

def _sidecar_path(file_path: str) -> Path:
    return Path(file_path).with_suffix('.epoched_config.json')


def has_config(file_path: str) -> bool:
    p = _sidecar_path(file_path)
    if not p.exists():
        return False
    try:
        cfg = json.loads(p.read_text(encoding='utf-8'))
        return cfg.get('unit') in _UNIT_CHOICES
    except Exception:
        return False


def load_config(file_path: str) -> dict:
    p = _sidecar_path(file_path)
    if not p.exists():
        raise FileNotFoundError(
            f"No epoched-MAT config found for {Path(file_path).name}")
    return json.loads(p.read_text(encoding='utf-8'))


def save_config(file_path: str, unit: str) -> None:
    """Persist the analyst's unit decision.  ``unit`` must be in _UNIT_CHOICES."""
    if unit not in _UNIT_CHOICES:
        raise ValueError(
            f"unit must be one of {_UNIT_CHOICES}, got {unit!r}")
    _sidecar_path(file_path).write_text(
        json.dumps({'unit': unit}, indent=2), encoding='utf-8')


# ── Load + cache (LRU-1, mirrors spike2_smr.py) ───────────────────────────────

_cache_lock = threading.Lock()
_cached_path: list = [None]
_cached_mat:  list = [None]


def _load(file_path: str) -> dict:
    """Load and cache the MAT contents as a dict of mat_struct objects."""
    key = str(file_path)
    with _cache_lock:
        if _cached_path[0] == key and _cached_mat[0] is not None:
            return _cached_mat[0]
    try:
        mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        raise ValueError(
            f"Could not read epoched .mat file ({file_path!r}): {exc}") from exc
    if not _SIGNATURE.issubset(set(mat.keys())):
        missing = sorted(_SIGNATURE - set(mat.keys()))
        raise ValueError(
            f"{Path(file_path).name} is missing expected variables: {missing}")
    with _cache_lock:
        _cached_path[0] = key
        _cached_mat[0] = mat
    return mat


def clear_cache():
    with _cache_lock:
        _cached_path[0] = None
        _cached_mat[0] = None


# ── Small accessors ───────────────────────────────────────────────────────────

def _chans(mat: dict):
    return mat['Chans']


def _data(mat: dict) -> np.ndarray:
    """Return Chans.Data as (ntrial, nsample, nchan), promoting 2-D inputs."""
    d = np.asarray(_chans(mat).Data, dtype=float)
    if d.ndim == 2:            # single channel — restore the trailing axis
        d = d[:, :, None]
    if d.ndim != 3:
        raise ValueError(
            f"Chans.Data must be 2-D or 3-D, got shape {d.shape}")
    return d


def _time_axis(mat: dict) -> np.ndarray:
    return np.asarray(_chans(mat).t, dtype=float).ravel()


def _fs(mat: dict) -> int:
    fs = float(np.asarray(_chans(mat).SamplingRate).ravel()[0])
    if fs <= 0:
        raise ValueError("Chans.SamplingRate is not positive.")
    return int(round(fs))


def _labels(mat: dict, nchan: int) -> list:
    raw = getattr(_chans(mat), 'Label', None)
    if raw is None:
        names = []
    else:
        arr = np.asarray(raw)
        names = [str(x).strip() for x in arr.ravel()] if arr.ndim else [str(arr).strip()]
    names = [n for n in names if n]
    if len(names) < nchan:
        names += [f'Channel {i + 1}' for i in range(len(names), nchan)]
    return names[:nchan]


def _stim_index(mat: dict) -> int:
    """Return the sample index of the stimulus within an epoch.

    Prefers Events.Times (documented as milliseconds relative to Chans.t);
    falls back to the sample nearest t = 0.
    """
    t = _time_axis(mat)
    try:
        ev = np.asarray(getattr(mat['Events'], 'Times')).ravel().astype(float)
        if ev.size and np.all(ev == ev[0]):
            return int(np.argmin(np.abs(t - ev[0])))
    except Exception:
        pass
    return int(np.argmin(np.abs(t)))


# ── ADC quantisation fingerprint ──────────────────────────────────────────────

def _infer_quantisation(data: np.ndarray) -> dict:
    """Recover the ADC grid from the sample values.

    Fixed-point acquisition leaves every sample on an exact integer multiple
    of one LSB.  Recovering that step, and checking whether the extreme codes
    sit on a power-of-two boundary, tells us the converter width, the
    full-scale range, and which values represent saturation.

    Returns a dict with keys lsb, bits, full_scale, neg_rail, pos_rail — any
    of which may be None when the data is not on a recognisable grid (e.g.
    already filtered or resampled in floating point).
    """
    out = {'lsb': None, 'bits': None, 'full_scale': None,
           'neg_rail': None, 'pos_rail': None}
    v = np.unique(np.asarray(data, dtype=float).ravel())
    v = v[np.isfinite(v)]
    if v.size < 16:
        return out

    dv = np.diff(v)
    dv = dv[dv > 0]
    if dv.size == 0:
        return out
    lsb = float(dv.min())
    if lsb <= 0:
        return out

    codes = v / lsb
    # Every value must land on the grid, else this is not fixed-point data.
    if not np.allclose(codes, np.round(codes), atol=1e-6, rtol=0):
        return out
    out['lsb'] = lsb

    code_min = int(round(codes.min()))
    code_max = int(round(codes.max()))
    for bits in (12, 14, 16, 24):
        neg = -(2 ** (bits - 1))
        pos = 2 ** (bits - 1) - 1
        if code_min == neg or code_max == pos:
            out['bits'] = bits
            out['full_scale'] = lsb * (2 ** bits)
            out['neg_rail'] = neg * lsb
            out['pos_rail'] = pos * lsb
            break
    return out


def suggest_unit(file_path: str):
    """Infer the amplitude unit and return (unit, evidence_lines).

    ``unit`` is one of _UNIT_CHOICES.  ``evidence_lines`` is a short list of
    human-readable statements for display in the confirmation dialog, so the
    analyst can see *why* a unit is being proposed rather than being asked to
    guess blind.
    """
    mat = _load(file_path)
    data = _data(mat)
    t = _time_axis(mat)
    ev: list = []

    q = _infer_quantisation(data)
    if q['lsb'] is not None:
        ev.append(f"Samples lie on an exact {q['lsb'] * 1000:.6g} \u00b5-unit "
                  f"quantisation grid.")
    if q['bits'] is not None:
        ev.append(f"Extreme codes match a {q['bits']}-bit converter with a "
                  f"full-scale range of \u00b1{q['full_scale'] / 2:.6g} units.")

    # Physiological plausibility: median peak-to-peak in the post-stimulus
    # window, evaluated under each candidate unit.
    post = (t > 0)
    if post.sum() < 4:
        post = np.ones_like(t, dtype=bool)
    ptp = data[:, post, :].max(axis=1) - data[:, post, :].min(axis=1)
    med = float(np.median(ptp))
    ev.append(f"Median post-stimulus peak-to-peak is {med:.4g} units.")

    lo, hi = _PLAUSIBLE_MEP_PTP_MV
    for unit, scale in (('mV', 1.0), ('V', 1e3), ('uV', 1e-3)):
        if lo <= med * scale <= hi:
            ev.append(f"Interpreting units as {unit} gives "
                      f"{med * scale:.4g} mV, within the expected range for a "
                      f"surface-EMG MEP.")
            return unit, ev

    ev.append("No unit interpretation places the amplitudes in the expected "
              "range for a surface-EMG MEP.")
    return 'unknown', ev


def get_clipped_trials(file_path: str, channel_idx: int = 0) -> list:
    """Return indices of trials containing ADC-saturated (railed) samples.

    A railed sample is one sitting exactly on the converter's extreme code.
    Peak-to-peak amplitude from such a trial is an underestimate of unknown
    size, so these trials warrant review rather than silent inclusion.
    Returns an empty list when the data is not on a recognisable fixed-point
    grid, or when no samples reach a rail.
    """
    mat = _load(file_path)
    data = _data(mat)
    _check_channel(channel_idx, data.shape[2])
    q = _infer_quantisation(data)
    if q['neg_rail'] is None and q['pos_rail'] is None:
        return []
    ch = data[:, :, channel_idx]
    tol = (q['lsb'] or 0.0) * 0.5
    hit = np.zeros(ch.shape, dtype=bool)
    if q['neg_rail'] is not None:
        hit |= np.abs(ch - q['neg_rail']) <= tol
    if q['pos_rail'] is not None:
        hit |= np.abs(ch - q['pos_rail']) <= tol
    return [int(i) for i in np.flatnonzero(hit.any(axis=1))]


# ── Guard-band construction ───────────────────────────────────────────────────

def _mirror_tile(seg: np.ndarray, n: int) -> np.ndarray:
    """Extend ``seg`` to ``n`` samples by alternating forward/reversed copies.

    Mirroring (rather than plain tiling) keeps the joins continuous, so the
    result contains no step edges for a filter to ring on.
    """
    if n <= 0:
        return np.empty(0, dtype=float)
    if seg.size == 0:
        return np.zeros(n, dtype=float)
    if seg.size == 1:
        return np.full(n, float(seg[0]))
    blocks, total = [], 0
    flip = False
    while total < n:
        blk = seg[::-1] if flip else seg
        blocks.append(blk)
        total += blk.size
        flip = not flip
    return np.concatenate(blocks)[:n].astype(float)


def _guard_from_baseline(baseline: np.ndarray, n: int,
                         anchor_first=None, anchor_last=None) -> np.ndarray:
    """Build an ``n``-sample guard from a baseline segment.

    ``anchor_first`` / ``anchor_last`` DC-shift the result so it joins the
    adjacent real sample continuously.  The guard is never analysed, so a DC
    offset within it is harmless; a step at the join would not be.
    """
    g = _mirror_tile(np.asarray(baseline, dtype=float), n)
    if g.size == 0:
        return g
    if anchor_first is not None:
        g = g + (float(anchor_first) - g[0])
    elif anchor_last is not None:
        g = g + (float(anchor_last) - g[-1])
    return g


def _stitch(epochs: np.ndarray, stim_idx: int, guard_n: int):
    """Concatenate epochs into a pseudo-continuous trace with guard bands.

    Parameters
    ----------
    epochs   : (ntrial, nsample) array, one row per trial
    stim_idx : sample index of the stimulus within each epoch
    guard_n  : guard length in samples

    Returns
    -------
    trace       : 1-D float array
    stim_idxs   : list[int]  index of each trial's stimulus within ``trace``
    """
    ntrial, nsample = epochs.shape
    baselines = [epochs[i, :max(stim_idx, 1)] for i in range(ntrial)]

    pieces: list = []
    stim_idxs: list = []
    pos = 0

    lead = _guard_from_baseline(baselines[0], guard_n,
                                anchor_last=epochs[0, 0])
    pieces.append(lead)
    pos += lead.size

    for i in range(ntrial):
        pieces.append(epochs[i])
        stim_idxs.append(pos + stim_idx)
        pos += nsample
        if i < ntrial - 1:
            # Cosine crossfade between a guard anchored to the end of this
            # epoch and one anchored to the start of the next, so both joins
            # are continuous and nothing steps in between.
            n = guard_n * 2
            a = _guard_from_baseline(baselines[i], n,
                                     anchor_first=epochs[i, -1])
            b = _guard_from_baseline(baselines[i + 1], n,
                                     anchor_last=epochs[i + 1, 0])
            w = 0.5 * (1.0 - np.cos(np.pi * np.arange(n) / max(n - 1, 1)))
            pieces.append((1.0 - w) * a + w * b)
            pos += n

    trail = _guard_from_baseline(baselines[-1], guard_n,
                                 anchor_first=epochs[-1, -1])
    pieces.append(trail)

    return np.concatenate(pieces), stim_idxs


# ── Public API ────────────────────────────────────────────────────────────────

def _check_channel(channel_idx: int, nchan: int):
    if not (0 <= channel_idx < nchan):
        raise IndexError(
            f"channel_idx {channel_idx} out of range (0..{nchan - 1})")


def list_waveform_channels(file_path: str) -> list:
    """Return channel names from Chans.Label."""
    mat = _load(file_path)
    return _labels(mat, _data(mat).shape[2]) or ['Channel 1']


def get_epoch_bounds(file_path: str):
    """Return (pre_ms, post_ms) actually available around the stimulus.

    Callers must clamp their analysis windows to these values: the file
    simply does not contain data outside them, and any window that appears to
    extend further is reading guard-band padding, not recorded signal.
    """
    mat = _load(file_path)
    t = _time_axis(mat)
    idx = _stim_index(mat)
    return float(abs(t[0] - t[idx])), float(t[-1] - t[idx])


def get_trial_count(file_path: str) -> int:
    """Return the number of trials in the file."""
    return int(_data(_load(file_path)).shape[0])


def _resolved_unit(file_path: str):
    """Return the unit string to report, or None when unknown/unscalable."""
    if has_config(file_path):
        unit = load_config(file_path).get('unit')
    else:
        unit, _ = suggest_unit(file_path)
    return None if unit == 'unknown' else unit


def units_assumed(file_path: str) -> bool:
    """True when amplitudes are NOT verified against a confirmed unit.

    Downstream output should record this so that a reader of the results can
    tell whether a column labelled "(mV)" is a measurement or an assumption.
    """
    if not has_config(file_path):
        return True
    return load_config(file_path).get('unit') == 'unknown'


def extract_emg_waveform_and_fs(file_path: str, channel_idx: int = 0):
    """Return one channel as a stitched pseudo-continuous trace.

    Returns
    -------
    emg  : np.ndarray  guard-padded concatenation of every epoch
    fs   : int         sampling rate in Hz
    unit : str | None  confirmed/inferred unit; None when unknown
    """
    mat = _load(file_path)
    data = _data(mat)
    _check_channel(channel_idx, data.shape[2])
    fs = _fs(mat)
    guard_n = int(round(GUARD_MS * fs / 1000.0))
    trace, _ = _stitch(data[:, :, channel_idx], _stim_index(mat), guard_n)
    return trace, fs, _resolved_unit(file_path)


def extract_stim_times(file_path: str, marker_name: str = 'A') -> dict:
    """Return stimulus times on the stitched time base, grouped by stim type.

    Stim types come from Conds.TrialCodes when the file carries more than one
    condition; a single-condition file yields one group named after
    Conds.Label (falling back to ``marker_name``).

    Returns
    -------
    dict mapping stim_type -> list of timestamps (seconds)
    """
    mat = _load(file_path)
    data = _data(mat)
    fs = _fs(mat)
    guard_n = int(round(GUARD_MS * fs / 1000.0))
    _, stim_idxs = _stitch(data[:, :, 0], _stim_index(mat), guard_n)
    ntrial = data.shape[0]

    # Per-trial condition codes, when present and well-formed.
    codes = None
    try:
        tc = np.asarray(getattr(mat['Conds'], 'TrialCodes')).ravel()
        if tc.size == ntrial:
            codes = tc
    except Exception:
        codes = None

    default = str(getattr(mat['Conds'], 'Label', '') or '').strip()
    if not default:
        default = (marker_name or 'A').strip() or 'A'

    if codes is None or np.unique(codes).size <= 1:
        return {default: [float(stim_idxs[i]) / fs for i in range(ntrial)]}

    out: dict = {}
    for i in range(ntrial):
        key = f"{default}-{codes[i]}"
        out.setdefault(key, []).append(float(stim_idxs[i]) / fs)
    return out
