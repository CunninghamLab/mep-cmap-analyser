"""
Round-trip the event path on files built by the test itself.

The tests in test_event_source_roundtrip.py use real recordings, which is the
stronger evidence -- but those files live on one machine. On any other, and in
CI, all twenty-six of them skip. A test that only runs where its author sits is
close to no test at all, and the change it guards (every reader's event
handling routed through a new function) is the one most able to break quietly.

These build minimal files for the formats that can be written without a vendor
library, so the contract is checked everywhere the suite runs:

    extract_events(path, [EventSource(kind="embedded")])
        == extract_stim_times(path)

Formats needing a vendor library -- EDF via pyedflib, Spike2 SMR via Neo,
AcqKnowledge -- are covered by the real-file tests only, and that gap is
recorded in test_formats_without_synthetic_cover below rather than left
implicit.
"""

import pathlib
import struct
import tempfile

import numpy as np
import pytest

from mep_cmap.event_sources import EventSource
from mep_cmap.io import detect_format, extract_events, extract_stim_times

FS = 1000.0
N = 5000


def _labchart_txt(d):
    """LabChart text export: an EMG column and a trigger column."""
    p = pathlib.Path(d) / "synthetic.txt"
    t = np.arange(N) / FS
    emg = 0.01 * np.sin(2 * np.pi * 7 * t)
    trig = np.zeros(N)
    for k in (500, 1500, 2500, 3500):
        trig[k:k + 20] = 5.0
    with open(p, "w") as f:
        f.write("Interval=\t%.6f s\n" % (1 / FS))
        f.write("ChannelTitle=\tEMG\tTrigger\n")
        for i in range(N):
            f.write("%.6f\t%.6f\t%.6f\n" % (t[i], emg[i], trig[i]))
    return str(p)


def _generic_tsv(d):
    p = pathlib.Path(d) / "synthetic.csv"
    t = np.arange(N) / FS
    emg = 0.01 * np.sin(2 * np.pi * 7 * t)
    trig = np.zeros(N)
    for k in (500, 1500, 2500):
        trig[k:k + 20] = 5.0
    np.savetxt(p, np.column_stack([t, emg, trig]), delimiter=",", fmt="%.6f")
    return str(p)


def _brainvision(d):
    """A .vhdr/.vmrk/.eeg triple with three markers."""
    stem = pathlib.Path(d) / "synthetic"
    data = (0.01 * np.sin(2 * np.pi * 7 * np.arange(N) / FS)).astype("<f4")
    data.tofile(str(stem) + ".eeg")
    with open(str(stem) + ".vhdr", "w") as f:
        # "BrainVision", one word: the reader matches this exactly.
        f.write("BrainVision Data Exchange Header File Version 1.0\n\n"
                "[Common Infos]\nDataFile=synthetic.eeg\n"
                "MarkerFile=synthetic.vmrk\nDataFormat=BINARY\n"
                "DataOrientation=MULTIPLEXED\nNumberOfChannels=1\n"
                "SamplingInterval=%d\n\n"
                "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n\n"
                "[Channel Infos]\nCh1=EMG,,1,µV\n" % int(1e6 / FS))
    with open(str(stem) + ".vmrk", "w") as f:
        f.write("Brain Vision Data Exchange Marker File, Version 1.0\n\n"
                "[Common Infos]\nDataFile=synthetic.eeg\n\n"
                "[Marker Infos]\n"
                "Mk1=New Segment,,1,1,0\n")
        for i, pos in enumerate((500, 1500, 2500), start=2):
            f.write("Mk%d=Stimulus,S128,%d,1,0\n" % (i, pos))
    return str(stem) + ".vhdr"


# generic_tsv is deliberately absent: it cannot be read without a Format
# Wizard configuration sidecar, so a synthetic file would be testing the
# wizard's output rather than the event path. It is covered by the real-file
# tests where a configured file exists.
def _labchart_mat(d):
    """LabChart MATLAB export: two channels and a comment table.

    Worth covering synthetically because it is the format where threshold
    detection newly became possible -- it could previously read only its
    comment table, so a regression here would remove a capability rather than
    change a number.
    """
    from scipy.io import savemat

    p = pathlib.Path(d) / "synthetic_lc.mat"
    t = np.arange(N) / FS
    emg = 0.01 * np.sin(2 * np.pi * 7 * t)
    trig = np.zeros(N)
    for k in (500, 1500, 2500, 3500):
        trig[k:k + 20] = 5.0

    data = np.concatenate([emg, trig])[None, :]
    datastart = np.array([[1.0], [float(N + 1)]])
    dataend = np.array([[float(N)], [float(2 * N)]])
    # com columns: channel, block, tick position, comment-text index, type
    com = np.array([[1.0, 1.0, float(k + 1), 1.0, 1.0] for k in
                    (500, 1500, 2500, 3500)])
    savemat(str(p), {
        "data": data,
        "datastart": datastart,
        "dataend": dataend,
        "titles": np.array(["EMG     ", "Trigger "]),
        "samplerate": np.array([[FS], [FS]]),
        "unittext": np.array(["mV"]),
        "unittextmap": np.array([[1.0], [1.0]]),
        "tickrate": np.array([[FS]]),
        "blocktimes": np.array([[0.0]]),
        "firstsampleoffset": np.array([[0.0], [0.0]]),
        "comtext": np.array(["Stim"]),
        "com": com,
    })
    return str(p)


BUILDERS = [
    ("labchart", _labchart_txt),
    ("brainvision", _brainvision),
    ("labchart_mat", _labchart_mat),
]


@pytest.fixture(scope="module")
def built():
    d = tempfile.mkdtemp(prefix="mepcmap_fmt_")
    return {name: fn(d) for name, fn in BUILDERS}


@pytest.mark.parametrize("fmt", [b[0] for b in BUILDERS])
def test_the_file_is_recognised_as_the_format_it_is(built, fmt):
    """If detection is wrong the round-trip below proves nothing."""
    assert detect_format(built[fmt]) == fmt


@pytest.mark.parametrize("fmt", [b[0] for b in BUILDERS])
def test_one_embedded_source_reproduces_extract_stim_times(built, fmt):
    p = built[fmt]
    old = dict(extract_stim_times(p, "") or {})
    new, warnings = extract_events(p, [EventSource(kind="embedded")])
    assert set(new) == set(old)
    for k in old:
        a, b = sorted(float(t) for t in old[k]), list(new[k])
        assert len(a) == len(b)
        assert all(abs(x - y) < 1e-9 for x, y in zip(a, b))
    assert not warnings


@pytest.mark.parametrize("fmt", [b[0] for b in BUILDERS])
def test_no_sources_behaves_as_before(built, fmt):
    p = built[fmt]
    new, _ = extract_events(p, [])
    assert new == dict(extract_stim_times(p, "") or {})


def test_brainvision_finds_the_markers_that_were_written():
    """Guards the fixture: an empty result would make the round-trip vacuous."""
    d = tempfile.mkdtemp(prefix="mepcmap_bv_")
    p = _brainvision(d)
    got = extract_stim_times(p, "")
    assert got, "the fixture wrote no readable markers"
    assert sum(len(v) for v in got.values()) == 3


def test_a_threshold_source_works_on_a_synthetic_file():
    """
    The capability that did not exist before: threshold detection on a format
    whose reader only ever produced comment timestamps. Both routes should find
    the same four stimuli.
    """
    d = tempfile.mkdtemp(prefix="mepcmap_thr_")
    p = _labchart_mat(d)
    comments = sorted(t for ts in extract_stim_times(p, "").values() for t in ts)
    assert len(comments) == 4, "the fixture's comment table is not as expected"

    found, _ = extract_events(p, [EventSource(
        kind="threshold", channel="Trigger", level=2.5, edge="rising",
        refractory_ms=50.0, label="A")])
    times = found["A"]
    assert len(times) == 4
    assert all(min(abs(t - c) for c in comments) < 0.005 for t in times), (
        "threshold detection found different events from the comment table"
    )


def test_formats_without_synthetic_cover_are_named():
    """
    Recorded rather than left implicit: these are checked only by the
    real-file tests, which skip wherever those recordings are absent.
    """
    uncovered = {"spike2_smr", "edf", "acqknowledge_acq", "acqknowledge_mat",
                 "cfwb", "epoched_mat", "brainsight", "mne", "generic_tsv"}
    covered = {name for name, _ in BUILDERS}
    assert not (covered & uncovered), (
        "a format is listed as uncovered but now has a synthetic fixture; "
        "remove it from the list"
    )
