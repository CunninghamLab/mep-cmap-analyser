"""
The new event path must reproduce the old one exactly.

extract_stim_times is UNCHANGED and is what an embedded source calls.
extract_events builds on it rather than replacing it, so a file configured the
way every file is configured today runs through no new code at all.

That is the whole safety argument for this change, and it is only worth
anything if it is checked on real recordings. These tests use files covering
six formats; where a file is absent the test skips rather than passing
vacuously, and the skip is visible in the run.
"""

import pathlib

import pytest

from mep_cmap.event_sources import EventSource
from mep_cmap.io import (extract_events, extract_stim_times,
                         list_event_sources)

U = pathlib.Path("/mnt/user-data/uploads")

# (filename, marker argument, format) -- one real recording per reader.
REAL_FILES = [
    ("283_000.mat", "", "epoched_mat"),
    ("A025_V2_CS_RC_iSP_120MEP_B.mat", "", "labchart_mat"),
    ("StAn_50_s1_B1.vhdr", "", "brainvision"),
    ("StAn_50_s2_V3.vhdr", "", "brainvision"),
    ("sub-015_ses-2_limb-left_260116_123937_000_emg.edf", "", "edf"),
    ("sub-006_ses-1_limb-right_260810_182240_000.smr", "DigMark", "spike2_smr"),
    ("sub-o001_task-dualhold_acq-cse_cond-30_emg.txt", "", "labchart"),
]


def _need(name):
    p = U / name
    if not p.exists():
        pytest.skip(f"{name} not present in this environment")
    return str(p)


@pytest.mark.parametrize("name,marker,fmt", REAL_FILES,
                         ids=[f"{f[2]}:{f[0][:18]}" for f in REAL_FILES])
def test_one_embedded_source_reproduces_extract_stim_times(name, marker, fmt):
    """
    The contract that makes this change safe. Every existing file, session and
    sidecar goes through this path, so any divergence is a silent change to
    results that nothing else would catch.
    """
    p = _need(name)
    old = dict(extract_stim_times(p, marker) or {})
    new, warnings = extract_events(p, [EventSource(kind="embedded",
                                                   channel=marker)])
    assert set(new) == set(old), f"{fmt}: stimulus types differ"
    for k in old:
        a, b = sorted(float(t) for t in old[k]), list(new[k])
        assert len(a) == len(b), f"{fmt}/{k}: {len(a)} events became {len(b)}"
        assert all(abs(x - y) < 1e-9 for x, y in zip(a, b)), \
            f"{fmt}/{k}: event times moved"
    assert not warnings


@pytest.mark.parametrize("name,marker,fmt", REAL_FILES,
                         ids=[f"{f[2]}:{f[0][:18]}" for f in REAL_FILES])
def test_no_sources_at_all_behaves_as_before(name, marker, fmt):
    """An unconfigured file must still read its own events."""
    p = _need(name)
    new, _ = extract_events(p, [])
    assert new == dict(extract_stim_times(p, "") or {})


@pytest.mark.parametrize("name,marker,fmt", REAL_FILES,
                         ids=[f"{f[2]}:{f[0][:18]}" for f in REAL_FILES])
def test_every_format_can_say_what_it_offers(name, marker, fmt):
    p = _need(name)
    src = list_event_sources(p)
    assert set(src) == {"embedded", "analogue"}
    assert isinstance(src["embedded"], list)
    assert isinstance(src["analogue"], list)


def test_codes_restrict_an_embedded_source():
    """Selecting some of a file's stimulus types, not all of them."""
    p = _need("sub-015_ses-2_limb-left_260116_123937_000_emg.edf")
    everything, _ = extract_events(p, [EventSource(kind="embedded")])
    assert len(everything) > 2, "fixture no longer has several stimulus types"
    two = sorted(everything)[:2]
    some, _ = extract_events(p, [EventSource(kind="embedded", codes=tuple(two))])
    assert set(some) == set(two)
    for k in two:
        assert some[k] == everything[k]


# ── The capability that did not exist ────────────────────────────────────────

def test_a_threshold_source_works_on_a_format_that_only_read_comments():
    """
    LabChart .mat read its comment table and nothing else -- not a missing
    feature so much as a missing question, since the one parameter that could
    have asked for a trigger channel already meant something else there.

    Detection on the analogue trigger channel lands within a millisecond of the
    comment timestamps, which is the evidence that it is finding the same
    physical events by a different route.
    """
    import numpy as np

    p = _need("A025_V2_CS_RC_iSP_120MEP_B.mat")
    comments = sorted(extract_stim_times(p, "")["Trigger"])

    found, _ = extract_events(p, [EventSource(
        kind="threshold", channel="Channel 6", level=1.0, edge="rising",
        refractory_ms=50.0, label="A")])
    times = found["A"]
    assert times, "no crossings detected on the trigger channel"

    offsets = [min(abs(t - c) for c in comments) for t in times]
    assert float(np.median(offsets)) < 0.005, (
        f"detected events sit {np.median(offsets) * 1000:.1f} ms from the "
        f"comment timestamps; they are not the same events"
    )


def test_a_threshold_source_names_the_channel_it_cannot_find():
    p = _need("A025_V2_CS_RC_iSP_120MEP_B.mat")
    with pytest.raises(ValueError) as exc:
        extract_events(p, [EventSource(kind="threshold", channel="Nonexistent",
                                       level=1.0, label="A")])
    assert "Nonexistent" in str(exc.value)


def test_an_interval_source_is_bounded_by_the_recording():
    p = _need("A025_V2_CS_RC_iSP_120MEP_B.mat")
    found, _ = extract_events(p, [EventSource(
        kind="interval", start_s=10.0, period_s=60.0, count=10_000,
        label="A")])
    times = found["A"]
    assert times
    assert times[0] == 10.0
    assert len(times) < 10_000, "events were not limited to the recording"


def test_a_saved_source_list_round_trips():
    """Sources are stored in the sidecar as plain dicts."""
    srcs = [EventSource(kind="embedded", channel="DigMark", codes=("A",)),
            EventSource(kind="threshold", channel="Ch 5", level=2.0, label="C")]
    back = [EventSource.from_dict(d) for d in (s.to_dict() for s in srcs)]
    assert back == srcs
