"""
Shared pytest fixtures and headless setup.

``mep_cmap/__init__.py`` imports ``compat``, which imports tkinter.  CI runners
frequently lack python3-tk, so tkinter is stubbed when it is genuinely absent.
The stub is only installed if the real import fails, so a normal development
machine tests against the real thing.
"""

import struct
import sys
import types

import pytest


def _stub_tkinter_if_missing():
    try:
        import tkinter  # noqa: F401
        return False
    except Exception:
        pass
    for name in ("tkinter", "tkinter.ttk", "tkinter.filedialog",
                 "tkinter.messagebox", "tkinter.font", "tkinter.simpledialog",
                 "tkinter.colorchooser", "tkinter.scrolledtext"):
        mod = types.ModuleType(name)
        mod.__getattr__ = lambda _n: types.SimpleNamespace()
        sys.modules[name] = mod
    return True


_stub_tkinter_if_missing()


# ── Synthetic BrainVision fixture ─────────────────────────────────────────────
#
# Built in-process rather than committed as binary test data: no participant
# recordings enter the repository, and every value below is known exactly, so
# assertions can be on precise numbers rather than tolerances.

BV_FS = 5000                 # Hz  -> SamplingInterval 200 us
BV_RESOLUTION = 0.1          # uV per digital unit (as BrainVision Recorder writes)
BV_N_SAMPLES = 50_000        # 10 s
BV_CHANNELS = ["FDI_L", "APB_L", "ADM_L"]

# 1-based .vmrk positions. Deliberately includes the leading New Segment marker
# at position 1, which is what proves positions are 1-based: it denotes the
# first sample of the recording, i.e. t = 0.
BV_MARKERS = [
    ("New Segment", "",     1),
    ("Stimulus",    "S  1", 5001),      # -> (5001-1)/5000 = 1.0 s exactly
    ("Stimulus",    "S  2", 10001),     # -> 2.0 s
    ("Stimulus",    "S  1", 15001),     # -> 3.0 s
    ("Stimulus",    "S  2", 20001),     # -> 4.0 s
]

# Peak-to-peak of channel 0, in digital units -> 2000 * 0.1 uV = 200 uV = 0.2 mV
BV_CH0_PEAK_DIGITAL = 1000


@pytest.fixture
def brainvision_triplet(tmp_path):
    """
    Write a minimal, valid BrainVision .vhdr/.vmrk/.eeg triplet.

    Returns the path to the .vhdr.  Channel 0 carries a square pulse of known
    peak-to-peak amplitude; channels 1 and 2 are zero.
    """
    stem = "sub-001_ses-01_task-tms"
    vhdr = tmp_path / f"{stem}.vhdr"
    vmrk = tmp_path / f"{stem}.vmrk"
    eeg = tmp_path / f"{stem}.eeg"

    n_ch = len(BV_CHANNELS)
    data = [0] * (BV_N_SAMPLES * n_ch)
    # Square pulse on channel 0 spanning +/- BV_CH0_PEAK_DIGITAL
    for i in range(1000, 1100):
        data[i * n_ch + 0] = BV_CH0_PEAK_DIGITAL
    for i in range(1100, 1200):
        data[i * n_ch + 0] = -BV_CH0_PEAK_DIGITAL
    eeg.write_bytes(struct.pack(f"<{len(data)}h", *data))

    ch_lines = "\n".join(
        f"Ch{i + 1}={name},,{BV_RESOLUTION},\u00b5V"
        for i, name in enumerate(BV_CHANNELS))
    vhdr.write_text(
        "BrainVision Data Exchange Header File Version 1.0\n"
        "; synthetic test fixture\n\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        f"DataFile={eeg.name}\n"
        f"MarkerFile={vmrk.name}\n"
        "DataFormat=BINARY\n"
        "DataOrientation=MULTIPLEXED\n"
        f"NumberOfChannels={n_ch}\n"
        f"SamplingInterval={int(1e6 / BV_FS)}\n\n"
        "[Binary Infos]\n"
        "BinaryFormat=INT_16\n\n"
        "[Channel Infos]\n"
        f"{ch_lines}\n",
        encoding="utf-8")

    mk_lines = []
    for n, (mtype, desc, pos) in enumerate(BV_MARKERS, start=1):
        extra = ",20250802173116615995" if mtype == "New Segment" else ""
        mk_lines.append(f"Mk{n}={mtype},{desc},{pos},1,0{extra}")
    vmrk.write_text(
        "BrainVision Data Exchange Marker File Version 1.0\n\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        f"DataFile={eeg.name}\n\n"
        "[Marker Infos]\n"
        + "\n".join(mk_lines) + "\n",
        encoding="utf-8")

    return vhdr
