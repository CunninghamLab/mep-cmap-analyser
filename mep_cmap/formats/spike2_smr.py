"""
mep_cmap.formats.spike2_smr
Native Spike2 .smr file reader using the Neo library.

On first open: dialog to choose EMG channel and stim/trigger channel.
Config saved to <file>.smr_config.json sidecar.

Stim time grouping mirrors smr2txt.py: per-event marker codes are decoded
from Neo event labels or waveforms so that DigMark events are split by
code letter (A, B, C...) exactly as the text-export reader does.

extract_stim_times(path, marker_name) behaviour:
  - If marker_name matches a known code in the stim channel,
    return only events with that code: {"A": [t1, t2, ...]}
  - If marker_name is the channel name (first scan), return all
    codes grouped: {"A": [...], "B": [...], ...}
  - Falls back to analogue threshold crossing when no event channels exist.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Graceful dependency check
# ---------------------------------------------------------------------------

def _neo_available() -> bool:
    try:
        import neo  # noqa: F401
        return True
    except ImportError:
        return False


def _require_neo():
    if not _neo_available():
        raise ImportError(
            "The 'neo' package is required to read native Spike2 .smr files.\n"
            "Install it with:  pip install neo"
        )


# ---------------------------------------------------------------------------
# Sidecar config helpers
# ---------------------------------------------------------------------------

#: This reader's sidecar suffix. Held here so the shared locator
#: and the reset path name the same file.
SIDECAR_SUFFIX = ".smr_config.json"

def _sidecar_path(file_path: str) -> Path:
    """Where this recording's configuration lives.

    Under derivatives, not beside the recording: rawdata is what the
    acquisition system wrote, and one program's settings do not belong
    in it. A sidecar still sitting in the old place is moved the first
    time it is looked for, so a study configured before this change is
    not asked to configure itself again.
    """
    from ..sidecars import resolve
    return resolve(file_path, SIDECAR_SUFFIX)


def has_config(file_path: str) -> bool:
    p = _sidecar_path(file_path)
    if not p.exists():
        return False
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        return bool(cfg.get("emg_channel") and cfg.get("stim_channel"))
    except Exception:
        return False


def load_config(file_path: str) -> dict:
    p = _sidecar_path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"No SMR config found for {Path(file_path).name}")
    return json.loads(p.read_text(encoding="utf-8"))


def save_config(file_path: str, emg_channel: str, stim_channel: str,
                analysis_channels=None) -> None:
    """Record the channel assignment beside the recording.

    ``emg_channel`` stays the FIRST channel to be analysed, so a build without
    multi-channel support reads this sidecar unchanged and analyses that one
    rather than failing on an unfamiliar key.

    Channels are stored by NAME, not index. An index means nothing if the
    recording is re-exported with a different channel order, and this file is
    where names are resolved in the first place.
    """
    p = _sidecar_path(file_path)
    cfg = {"emg_channel": emg_channel, "stim_channel": stim_channel}
    names = [str(c) for c in (analysis_channels or []) if str(c).strip()]
    if names:
        cfg["analysis_channels"] = names
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def analysis_channels_from_config(cfg, available):
    """Channel names to analyse, filtered to those the file actually has.

    A sidecar written before multi-channel support has no ``analysis_channels``
    and loads as the single channel it names -- no migration, no prompt.

    A stored name absent from the file is dropped rather than shifting the
    selection onto a neighbour: the recording has been re-exported or renamed,
    and silently analysing a different channel is worse than analysing fewer.
    Returns ``(names, dropped)`` so the caller can report what went missing.
    """
    stored = list((cfg or {}).get("analysis_channels")
                  or ([cfg.get("emg_channel")] if (cfg or {}).get("emg_channel")
                      else []))
    have = list(available or [])
    names = [n for n in stored if n in have]
    dropped = [n for n in stored if n not in have]
    return names, dropped


# ---------------------------------------------------------------------------
# Segment + rawio cache (LRU-1)
# ---------------------------------------------------------------------------

_cache_lock   = threading.Lock()
_cached_path:  list = [None]
_cached_seg:   list = [None]
_cached_names: list = [None]


def _b2s(x):
    return x.decode("latin-1", errors="replace") if isinstance(x, (bytes, bytearray)) else str(x)


def _load(file_path: str):
    """Load SMR via Neo, cache result. Returns (seg, analogue_names)."""
    with _cache_lock:
        if _cached_path[0] == file_path and _cached_seg[0] is not None:
            return _cached_seg[0], _cached_names[0]

    _require_neo()
    import neo
    import warnings

    try:
        reader = neo.io.Spike2IO(filename=file_path, try_signal_grouping=False)
    except TypeError:
        reader = neo.io.Spike2IO(filename=file_path)

    # Use rawio header for analogue names (more reliable than seg.analogsignals[i].name)
    analogue_names = None
    try:
        rawio = reader.rawio
        rawio.parse_header()
        analogue_names = [_b2s(m["name"]) for m in rawio.header["signal_channels"]]
    except Exception:
        try:
            from neo.rawio import Spike2RawIO
            rawio2 = Spike2RawIO(filename=file_path)
            rawio2.parse_header()
            analogue_names = [_b2s(m["name"]) for m in rawio2.header["signal_channels"]]
        except Exception:
            analogue_names = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*can not be converted to a quantity.*")
        warnings.filterwarnings("ignore", message=".*UnitWarning.*")
        block = reader.read_block(lazy=False)
    if not block.segments:
        raise ValueError(f"No segments found in {file_path}")
    seg = block.segments[0]

    if not analogue_names:
        analogue_names = [sig.name for sig in seg.analogsignals] or ["Channel 1"]

    with _cache_lock:
        _cached_path[0]  = file_path
        _cached_seg[0]   = seg
        _cached_names[0] = analogue_names

    return seg, analogue_names


def _load_all(file_path: str):
    """Every segment in the file, and the analogue channel names.

    Spike2 records in SAMPLING BLOCKS. A file paused and restarted between
    trials -- a common way to run a session -- arrives as one segment per
    block, each with its own start time, its own samples and its own events.

    :func:`_load` returns block 0 and nothing else, which is right for the
    questions it is asked (what channels exist, what are they called) and
    catastrophic for the two that matter: on a ten-block recording it read
    twelve per cent of the data and one stimulus of ten, silently, with the
    analysis reporting a clean result for the fraction it had seen.

    Returned in file order, which is time order.
    """
    _require_neo()
    import warnings

    import neo

    try:
        reader = neo.io.Spike2IO(filename=file_path, try_signal_grouping=False)
    except TypeError:
        reader = neo.io.Spike2IO(filename=file_path)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*can not be converted.*")
        warnings.filterwarnings("ignore", message=".*UnitWarning.*")
        block = reader.read_block(lazy=False)
    if not block.segments:
        raise ValueError(f"No segments found in {file_path}")

    _seg0, names = _load(file_path)
    return list(block.segments), names


def segment_count(file_path: str) -> int:
    """How many sampling blocks the recording holds. One for a continuous run."""
    try:
        return len(_load_all(file_path)[0])
    except Exception:
        return 1


def clear_cache():
    with _cache_lock:
        _cached_path[0]  = None
        _cached_seg[0]   = None
        _cached_names[0] = None


# ---------------------------------------------------------------------------
# Marker code decoding  (mirrors smr2txt.py _derive_labels)
# ---------------------------------------------------------------------------

def _decode_marker_code(raw) -> str:
    """
    Decode a single marker code value to a printable character.

    Handles bytes, numeric strings, and direct characters.
    Returns the single ASCII letter (e.g. 'A', 'B') or '?' if undecodable.
    """
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("latin-1", errors="replace").strip()
    s = str(raw).strip()
    # Numeric string (e.g. "66") -> chr(66) = 'B'
    if s.isdigit():
        v = int(s)
        if 32 <= v <= 126:
            return chr(v)
    # Already a single printable ASCII character
    if len(s) == 1 and 32 <= ord(s[0]) <= 126:
        return s
    # Multi-char: return as-is (e.g. already decoded)
    #
    # An EMPTY label returns empty, not "?". A plain trigger channel labels
    # none of its events, and turning that into "?" made every event look
    # decoded-but-unreadable: _get_event_codes tests `any(lb != "")` before
    # falling back to the channel name, so a list of "?" satisfied it and the
    # fallback the docstring describes was never reached. The stimulus type
    # then read "?" on the setup table and in the trial file -- a question the
    # analyst cannot answer, carried into their results.
    return s


def _channel_fallback_label(channel_name: str) -> str:
    """What to call events on a channel that labels none of them.

    Spike2 event channels need not carry marker codes: a plain trigger channel
    reports empty labels, and every event on it is the same kind of thing. The
    channel's own name says what they are, and shows up legibly on the setup
    table and in the trial file -- where '?' is a question the analyst cannot
    answer and will carry into their results.
    """
    name = (channel_name or "").strip()
    return name if name else "stim"


def _get_event_codes(evt) -> list:
    """
    Return a list of per-event marker codes for a Neo Event/Epoch object.

    Priority order (same as smr2txt.py _derive_labels):
    1. evt.labels  (array of per-event label bytes/strings)
    2. evt.waveforms (Spike2 DigMark stores the code as first sample)
    3. Fall back to the channel name repeated for each event
    """
    n = len(evt.times)

    # 1. labels attribute
    if hasattr(evt, "labels") and evt.labels is not None:
        try:
            labs = [_decode_marker_code(_b2s(lb).strip()) for lb in evt.labels]
            if len(labs) == n and any(lb != "" for lb in labs):
                return labs
        except Exception:
            pass

    # 2. waveforms (single-sample; code stored as first value)
    if hasattr(evt, "waveforms") and evt.waveforms is not None:
        try:
            codes = []
            for w in evt.waveforms:
                v = int(float(str(w.flat[0])))
                codes.append(chr(v) if 32 <= v <= 126 else "?")
            if len(codes) == n:
                return codes
        except Exception:
            pass

    # 3. fallback: channel name for all events
    return [evt.name] * n


def get_event_codes_for_channel(file_path: str, channel_name: str) -> list:
    """
    Return the sorted list of unique marker codes present in the named
    event/epoch/spike channel.  Used by app.py to populate the code picker.
    """
    seg, _ = _load(file_path)
    candidates = list(seg.events) + list(seg.epochs) + list(seg.spiketrains)
    cl = channel_name.lower()
    target = next(
        (c for c in candidates if c.name.lower() == cl or cl in c.name.lower()),
        None
    )
    if target is None:
        return []
    codes = _get_event_codes(target)
    return sorted(set(codes))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_STIM_KW = ("stim", "trig", "ttl", "digmark", "keyboard")


def _is_trigger(name: str) -> bool:
    return any(kw in name.lower() for kw in _STIM_KW)


def get_channel_info(file_path: str) -> dict:
    seg, analogue_names = _load(file_path)
    return {
        "analogue": analogue_names,
        "events":   [e.name for e in seg.events],
        "epochs":   [e.name for e in seg.epochs],
        "spikes":   [s.name for s in seg.spiketrains],
    }


def list_waveform_channels(file_path: str) -> list:
    try:
        _, names = _load(file_path)
        return names if names else ["Channel 1"]
    except ImportError:
        raise
    except Exception:
        return ["Channel 1"]


def list_event_channels(file_path: str) -> list:
    try:
        info = get_channel_info(file_path)
        names = info["events"] + info["epochs"] + info["spikes"]
        names += [n for n in info["analogue"] if _is_trigger(n)]
        return names
    except ImportError:
        raise
    except Exception:
        return []


def _pick_signal(seg, analogue_names, file_path, channel_idx):
    """The requested channel within one segment, by sidecar name or by index."""
    sig = None

    # Only resolve by sidecar name when channel_idx == 0 (the primary EMG).
    # For any other index the caller (inspector extra channel) is requesting
    # a specific channel by position — honour that directly.
    if channel_idx == 0 and has_config(file_path):
        target_name = load_config(file_path).get("emg_channel", "")
        tl = target_name.lower()
        for i, n in enumerate(analogue_names):
            if n.lower() == tl or tl in n.lower():
                if i < len(seg.analogsignals):
                    sig = seg.analogsignals[i]
                break

    if sig is None:
        idx = min(channel_idx, len(seg.analogsignals) - 1)
        sig = seg.analogsignals[idx]
    return sig


def _raw_channel_unit(file_path: str, channel_idx: int):
    """The unit string as the FILE wrote it, or None.

    Neo builds its AnalogSignals through `quantities`, which parses the unit
    text and substitutes 'dimensionless' for anything it does not recognise --
    printing 'Units "Nm" can not be converted to a quantity' as it goes. A
    torque channel therefore arrived here already stripped of its unit, and
    channels.tsv recorded it as dimensionless.

    The raw header keeps the original string, so it is read from there. The
    unit is carried as a LABEL and never parsed: this tool does no unit
    arithmetic, and a units library is precisely what lost 'Nm' in the first
    place. Whitespace is trimmed because the header pads some entries (' Volt'),
    and nothing else is normalised -- 'Volt' stays 'Volt'.

    Returns None on any failure, leaving the caller's existing path in place.
    """
    try:
        import neo
        raw = neo.rawio.Spike2RawIO(filename=file_path)
        raw.parse_header()
        chans = raw.header["signal_channels"]
        if channel_idx is None or not (0 <= int(channel_idx) < len(chans)):
            return None
        txt = str(chans[int(channel_idx)]["units"]).strip()
        return txt or None
    except Exception:                       # noqa: BLE001 — fall back below
        return None


def extract_emg_waveform_and_fs(file_path: str, channel_idx: int = 0) -> tuple:
    """The whole recording, across every sampling block.

    Blocks are placed at their real start times and the gaps between them
    zero-filled, so a stimulus timestamp means the same thing here as it does
    in the file. Reading block 0 alone -- which is what this did -- returned
    twelve per cent of a ten-block recording without saying so.
    """
    segments, analogue_names = _load_all(file_path)
    usable = [sg for sg in segments if sg.analogsignals]
    if not usable:
        raise ValueError(f"No analogue signals found in {file_path}")

    first = _pick_signal(usable[0], analogue_names, file_path, channel_idx)
    fs = int(round(float(first.sampling_rate.rescale("Hz").magnitude)))
    unit = _raw_channel_unit(file_path, channel_idx)
    if unit is None:
        try:
            u = str(first.units.dimensionality).strip().split()
            unit = u[-1] if u else None
        except Exception:
            pass

    if len(usable) == 1:
        return np.asarray(first).flatten().astype(float), fs, unit

    t0 = float(usable[0].t_start.rescale("s").magnitude)
    pieces = []
    for sg in usable:
        sig = _pick_signal(sg, analogue_names, file_path, channel_idx)
        start = int(round(
            (float(sg.t_start.rescale("s").magnitude) - t0) * fs))
        pieces.append((start, np.asarray(sig).flatten().astype(float)))

    total = max(start + len(a) for start, a in pieces)
    out = np.zeros(total, dtype=float)
    for start, arr in pieces:
        out[start:start + len(arr)] = arr
    return out, fs, unit


def extract_stim_times(file_path: str, marker_name: str = "A",
                       stim_channel: str = None) -> dict:
    """
    Return stim times grouped by marker code, normalised to t=0.

    If marker_name matches a specific code present in the stim channel
    (e.g. 'A'), only that code's events are returned.
    If marker_name matches the channel name itself (e.g. 'DigMark'),
    all codes are returned grouped: {"A": [...], "B": [...], ...}.
    """
    segments, analogue_names = _load_all(file_path)
    seg = segments[0]

    # Times are relative to the START OF THE RECORDING, not the start of each
    # block, because the waveform this pairs with is one continuous trace built
    # the same way. Reading block 0 alone returned one stimulus of ten on a
    # ten-block file, and reading each block from its own zero would put every
    # stimulus at the same place.
    t0 = (float(seg.analogsignals[0].t_start.rescale("s").magnitude)
          if seg.analogsignals
          else float(seg.t_start.rescale("s").magnitude))

    # An explicit stim_channel (e.g. picked in BIDS-ify "Scan file") wins; else
    # fall back to the saved sidecar config, else treat marker_name as the channel.
    if stim_channel:
        stim_ch = stim_channel
    elif has_config(file_path):
        stim_ch = load_config(file_path).get("stim_channel", marker_name)
    else:
        stim_ch = marker_name

    sl = stim_ch.lower()
    ml = marker_name.lower()

    # --- Find the stim event channel ---
    evt_all = list(seg.events) + list(seg.epochs) + list(seg.spiketrains)
    target  = None
    if evt_all:
        target = next((c for c in evt_all if c.name.lower() == sl), None)
        if target is None:
            target = next((c for c in evt_all if sl in c.name.lower()), None)
        if target is None:
            target = next((c for c in evt_all if _is_trigger(c.name)), None)
        if target is None:
            target = evt_all[0]

    if target is not None:
        # The same channel in every block, not only the first.
        codes, times_abs = [], []
        for sg in segments:
            _all = list(sg.events) + list(sg.epochs) + list(sg.spiketrains)
            _t = next((c for c in _all if c.name == target.name), None)
            if _t is None:
                continue
            codes.extend(_get_event_codes(_t))
            times_abs.extend(list(_t.times.rescale("s").magnitude))
        times_abs = np.asarray(times_abs, dtype=float)
        times_rel = times_abs - t0

        # Group by code
        by_code: dict = {}
        for t, code in zip(times_rel, codes):
            if t >= 0:
                by_code.setdefault(code, []).append(float(t))

        if not by_code:
            return {}

        # If marker_name is a specific code that exists, return only that
        if marker_name in by_code:
            return {marker_name: by_code[marker_name]}

        # If marker_name is the channel name or a fallback, return all codes
        return by_code

    # --- Analogue threshold crossing fallback ---
    trig_sig = None
    for i, n in enumerate(analogue_names):
        if n.lower() == sl or sl in n.lower() or _is_trigger(n):
            if i < len(seg.analogsignals):
                trig_sig = seg.analogsignals[i]
            break
    if trig_sig is None:
        return {}

    stim_arr = np.asarray(trig_sig).flatten().astype(float)
    fs_trig  = float(trig_sig.sampling_rate.rescale("Hz").magnitude)
    t0_trig  = float(trig_sig.t_start.rescale("s").magnitude)
    max_val  = stim_arr.max()
    if max_val <= 0:
        return {}
    thr   = max_val * 0.5
    edges = np.where(np.diff((stim_arr >= thr).astype(np.int8)) == 1)[0]
    label = marker_name[:1].upper() if len(marker_name) == 1 else marker_name
    times = ((edges + 1) / fs_trig + t0_trig - t0).tolist()
    return {label: [t for t in times if t >= 0]} if times else {}


def get_epoch_bounds(file_path: str):
    """The largest window every stimulus can supply, or None for one block.

    Spike2 records in blocks, and a stimulus cannot be epoched past the edges
    of the block it sits in: beyond that lies the zero-fill this reader inserts
    between blocks, and then the next trial. Each stimulus therefore has its
    own hard limit, and the file-wide bound is the smallest of them -- the
    window no trial exceeds.

    Deliberately not conditional on the blocks being stimulus-centred. An
    earlier version reported bounds only when every stimulus sat at the same
    offset within its block, and declined otherwise; but a recording whose
    blocks are merely paused-and-restarted has limits too, and declining left
    the analysis free to read a second of padding as though it were signal.
    Where the blocks ARE cut around the stimulus this returns exactly the
    stored epoch, which is the same answer by a more general route.

    None for a single-block recording, which is continuous and has no bounds.
    """
    try:
        segments, _names = _load_all(file_path)
    except Exception:
        return None
    if len(segments) < 2:
        return None

    pres, posts = [], []
    for sg in segments:
        try:
            a = float(sg.t_start.rescale("s").magnitude)
            b = float(sg.t_stop.rescale("s").magnitude)
        except Exception:
            continue
        for ch in list(sg.events) + list(sg.epochs):
            try:
                for t in ch.times.rescale("s").magnitude:
                    t = float(t)
                    if a <= t <= b:
                        pres.append((t - a) * 1000.0)
                        posts.append((b - t) * 1000.0)
            except Exception:
                continue

    if not pres:
        return None
    return (min(pres), min(posts))

def list_event_codes(file_path: str) -> dict:
    """Return {channel_name: {code: count}} for every event channel in the file.

    Powers the BIDS-ify "scan file" picker so users can see what stim markers
    exist and choose their stimuli, instead of typing a code blind. Codes are
    normalised the same way as extract_stim_times.
    """
    from collections import Counter
    seg, _analogue_names = _load(file_path)
    out = {}
    evt_all = list(seg.events) + list(seg.epochs) + list(seg.spiketrains)
    for ch in evt_all:
        try:
            codes = _get_event_codes(ch)
        except Exception:
            codes = []
        cnt = Counter(str(c) for c in codes if str(c) != "")
        out[str(ch.name)] = dict(cnt)
    return out
