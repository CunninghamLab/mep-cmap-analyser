"""
mep_cmap.formats.edf
~~~~~~~~~~~~~~~~~~~~
Reader for EDF+/BDF recordings — including the tool's own BIDS-ify output and
any standards-compliant EDF/BDF file. Exposes the same public functions as the
other format readers so io.py can dispatch to it.

Stim times
----------
Come from the sibling BIDS ``_events.tsv`` (authoritative — what BIDS-ify wrote)
when present, falling back to the EDF+ annotations embedded in the file so a
lone EDF is still usable.

Waveform length
---------------
EDF/BDF stores whole data records and zero-pads the final partial record. When
the sibling ``_emg.json`` records ``RecordingSampleCount`` (the true pre-padding
length, which BIDS-ify writes), the reader trims to it so the returned samples
match the original recording exactly. Without that sidecar it returns the full
(padded) signal.
"""

import os
import csv
import json

import numpy as np

try:
    import pyedflib
    _PYEDFLIB = True
except ImportError:
    _PYEDFLIB = False


def _require() -> None:
    if not _PYEDFLIB:
        raise RuntimeError("pyedflib is required to read EDF/BDF files "
                           "(pip install pyedflib).")


# ─── detection / sidecar path helpers ─────────────────────────────────────────
def is_edf(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() in (".edf", ".bdf")


def _sidecar_json_path(file_path: str) -> str:
    return os.path.splitext(file_path)[0] + ".json"


def _events_tsv_path(file_path: str) -> str:
    """Sibling BIDS events file: <entities>_emg.edf -> <entities>_events.tsv."""
    d = os.path.dirname(file_path)
    stem = os.path.splitext(os.path.basename(file_path))[0]
    if stem.endswith("_emg"):
        stem = stem[:-4]
    return os.path.join(d, stem + "_events.tsv")


def _true_length(file_path: str):
    """RecordingSampleCount from the sibling _emg.json, or None."""
    try:
        with open(_sidecar_json_path(file_path), encoding="utf-8") as fh:
            n = json.load(fh).get("RecordingSampleCount")
        return int(n) if n else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


# ─── public API ───────────────────────────────────────────────────────────────
def list_waveform_channels(file_path: str) -> list:
    """All signal labels in the file (user picks the EMG channel in the GUI)."""
    _require()
    r = pyedflib.EdfReader(file_path)
    try:
        return list(r.getSignalLabels())
    finally:
        r.close()


def list_event_channels(file_path: str) -> list:
    """EDF/BDF has no separate event channels in the Spike2 sense."""
    return []


def extract_emg_waveform_and_fs(file_path: str, channel_idx: int = 0):
    """
    Return (samples, fs_int, unit) for one channel, trimmed to the true
    pre-padding length when the sidecar records it.
    """
    _require()
    r = pyedflib.EdfReader(file_path)
    try:
        n_sig = r.signals_in_file
        if channel_idx < 0 or channel_idx >= n_sig:
            raise IndexError(
                f"channel_idx {channel_idx} out of range (0..{n_sig - 1})")
        sig = np.asarray(r.readSignal(channel_idx), dtype=float)
        fs = int(round(r.getSampleFrequency(channel_idx)))
        unit = (r.getSignalHeader(channel_idx).get("dimension") or "").strip()
        if unit in ("", "n/a"):
            unit = None
    finally:
        r.close()

    n_true = _true_length(file_path)
    if n_true is not None and 0 < n_true <= sig.shape[0]:
        sig = sig[:n_true]        # drop EDF whole-record zero padding
    return sig, fs, unit


def _read_events_tsv(path: str):
    """{trial_type: [onset_seconds, ...]} from a BIDS _events.tsv, or None."""
    if not os.path.isfile(path):
        return None
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames or "onset" not in reader.fieldnames:
            return None
        for row in reader:
            try:
                onset = float(row["onset"])
            except (TypeError, ValueError, KeyError):
                continue
            tt = (row.get("trial_type") or "").strip()
            if tt in ("", "n/a"):
                tt = "stim"
            out.setdefault(tt, []).append(onset)
    return out or None


def _read_annotations(file_path: str):
    """Fallback: {description: [onset_seconds, ...]} from EDF+ annotations, or None."""
    _require()
    r = pyedflib.EdfReader(file_path)
    try:
        onsets, _durations, descriptions = r.readAnnotations()
    finally:
        r.close()
    out = {}
    for onset, desc in zip(onsets, descriptions):
        tt = (str(desc) or "").strip() or "stim"
        out.setdefault(tt, []).append(float(onset))
    return out or None


def extract_stim_times(file_path: str, marker_name: str = None) -> dict:
    """
    Stim timestamps grouped by trial type, in seconds.

    Source order: sibling BIDS ``_events.tsv`` (authoritative), then EDF+
    annotations embedded in the file. ``marker_name`` is accepted for API
    parity but not required — the event/annotation labels define the types.
    """
    ev = _read_events_tsv(_events_tsv_path(file_path))
    if ev:
        return ev
    ann = _read_annotations(file_path)
    if ann:
        return ann
    return {}
