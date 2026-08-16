"""
Tests for stating where stimulus events come from.

Every reader exposed extract_stim_times(path, marker_name) and every reader
meant something different by it -- the event channel to read, the label to
attach to threshold detections, or nothing at all. The shared signature was a
coincidence of naming rather than an interface, so asking a reader for something
it did not already do was impossible.

Threshold and interval detection do not depend on the file format: both operate
on a waveform and a time base, which every reader already provides. Writing them
once here is what makes this affordable across nine formats rather than nine
implementations that drift apart.
"""

import numpy as np
import pytest

from mep_cmap.event_sources import (DEFAULT_REFRACTORY_MS, EventSource,
                                    detect_threshold_crossings,
                                    generate_interval_events,
                                    merge_event_sources)

FS = 5000.0


def ttl(n_pulses=5, period_s=2.0, first_s=1.0, ring=10, amp=5.0, dur_s=10.0):
    """A TTL train whose pulses ring, as a real stimulator output does."""
    x = np.zeros(int(dur_s * FS))
    for k in range(n_pulses):
        i = int((first_s + k * period_s) * FS)
        x[i:i + 10] = amp
        if ring:
            x[i + 10:i + 10 + 2 * ring] = np.tile([amp, -1.0], ring)
    return x


# ── Threshold detection ──────────────────────────────────────────────────────

def test_a_clean_train_gives_one_event_per_pulse():
    t = detect_threshold_crossings(ttl(ring=0), FS, 2.5)
    assert len(t) == 5
    assert np.allclose(t, [1.0, 3.0, 5.0, 7.0, 9.0])


def test_the_refractory_period_collapses_ringing():
    """
    A stimulator pulse rings, and each oscillation back past the level is a
    crossing. Without a refractory period one stimulus yields several events a
    fraction of a millisecond apart, which look like a very fast train rather
    than an artefact of the detection.
    """
    x = ttl(ring=10)
    assert len(detect_threshold_crossings(x, FS, 2.5, refractory_ms=0)) == 50
    assert len(detect_threshold_crossings(x, FS, 2.5, refractory_ms=50)) == 5


def test_the_default_refractory_is_long_enough_to_be_useful():
    assert DEFAULT_REFRACTORY_MS >= 10.0


@pytest.mark.parametrize("edge", ["rising", "falling", "both"])
def test_each_edge_finds_every_pulse_once(edge):
    t = detect_threshold_crossings(ttl(ring=0), FS, 2.5, edge=edge,
                                   refractory_ms=50)
    assert len(t) == 5


def test_a_rising_edge_is_timed_on_the_far_side():
    """The first sample ABOVE the level, not the last one below it."""
    x = np.zeros(1000)
    x[500:520] = 5.0
    t = detect_threshold_crossings(x, FS, 2.5, "rising", refractory_ms=0)
    assert len(t) == 1
    assert abs(t[0] - 500 / FS) < 1e-9


def test_a_signal_that_never_crosses_gives_nothing():
    assert detect_threshold_crossings(np.zeros(1000), FS, 2.5) == []


def test_a_signal_starting_above_the_level_is_not_a_crossing():
    """Otherwise every recording begins with a spurious event."""
    x = np.full(1000, 5.0)
    assert detect_threshold_crossings(x, FS, 2.5, "rising") == []


def test_the_time_base_offset_is_applied():
    x = np.zeros(1000)
    x[500:520] = 5.0
    t = detect_threshold_crossings(x, FS, 2.5, refractory_ms=0, t0=17.0)
    assert abs(t[0] - (17.0 + 500 / FS)) < 1e-9


def test_a_degenerate_signal_does_not_raise():
    assert detect_threshold_crossings([], FS, 1.0) == []
    assert detect_threshold_crossings([1.0], FS, 1.0) == []
    assert detect_threshold_crossings(np.zeros(10), 0, 1.0) == []


# ── Interval sources ─────────────────────────────────────────────────────────

def test_a_count_gives_exactly_that_many():
    assert generate_interval_events(1.0, 2.0, count=5) == [1.0, 3.0, 5.0, 7.0, 9.0]


def test_a_recording_length_fills_it():
    assert generate_interval_events(1.0, 2.0, duration_s=9.5) == \
        [1.0, 3.0, 5.0, 7.0, 9.0]


def test_events_past_the_end_are_dropped():
    assert generate_interval_events(1.0, 2.0, count=10, duration_s=5.5) == \
        [1.0, 3.0, 5.0]


def test_an_interval_source_needs_a_count_or_a_length():
    with pytest.raises(ValueError):
        generate_interval_events(0.0, 1.0)


def test_a_non_positive_period_is_refused():
    with pytest.raises(ValueError):
        generate_interval_events(0.0, 0.0, count=5)


# ── Merging ──────────────────────────────────────────────────────────────────

def test_one_stimulus_type_from_two_sources_is_an_error_not_a_union():
    """
    Two sources disagreeing about the same type is far more likely to be a
    misconfiguration than an intention, and silently combining them gives a
    trial count matching neither source.
    """
    merged, warnings = merge_event_sources(
        [("TTL", {"A": [1.0, 3.0]}), ("comments", {"A": [1.0]})])
    assert merged["A"] == [1.0, 3.0]          # the first source wins
    assert warnings and "produced by both" in warnings[0]


def test_near_simultaneous_events_are_kept_and_reported():
    """
    They may be one stimulus recorded twice -- a comment written just after a
    TTL pulse -- or two genuine stimuli in a paired-pulse protocol, and nothing
    in the data distinguishes those. Merging would silently halve a
    paired-pulse trial count.
    """
    merged, warnings = merge_event_sources(
        [("TTL", {"A": [1.0, 3.0]}), ("comments", {"C": [1.002]})])
    assert merged == {"A": [1.0, 3.0], "C": [1.002]}
    assert warnings and "kept as separate events" in warnings[0]
    assert "paired-pulse" in warnings[0], (
        "the message must offer the innocent explanation too"
    )


def test_well_separated_sources_produce_no_warning():
    merged, warnings = merge_event_sources(
        [("TTL", {"A": [1.0, 3.0]}), ("comments", {"C": [2.0]})])
    assert not warnings


def test_times_come_back_sorted():
    merged, _ = merge_event_sources([("x", {"A": [3.0, 1.0, 2.0]})])
    assert merged["A"] == [1.0, 2.0, 3.0]


# ── The specification object ─────────────────────────────────────────────────

def test_an_unknown_kind_is_refused_at_construction():
    with pytest.raises(ValueError):
        EventSource(kind="magic")
    with pytest.raises(ValueError):
        EventSource(kind="threshold", edge="sideways")


def test_a_source_round_trips_through_a_dict():
    """Sources are saved beside the channel assignment, so this must hold."""
    src = EventSource(kind="threshold", channel="Stim", level=2.5,
                      edge="falling", refractory_ms=20.0, label="C")
    back = EventSource.from_dict(src.to_dict())
    assert back == src


def test_an_unknown_key_in_a_saved_source_is_ignored():
    """A sidecar written by a later version must not stop this one loading."""
    d = EventSource(kind="embedded", channel="DigMark").to_dict()
    d["something_new"] = 42
    assert EventSource.from_dict(d).channel == "DigMark"


@pytest.mark.parametrize("src", [
    EventSource(kind="embedded", channel="DigMark", codes=("A", "C")),
    EventSource(kind="threshold", channel="Stim", level=2.5, label="A"),
    EventSource(kind="interval", period_s=5.0, count=60, label="A"),
])
def test_every_kind_describes_itself(src):
    text = src.describe()
    assert src.kind in text
    assert len(text) > 15
