"""
Spike2 sampling blocks.

Spike2 records in blocks: a session paused and restarted between trials
arrives as one segment per block, each with its own start time, samples and
events. The loader returned block 0 and nothing else, so a ten-block recording
was analysed as twelve per cent of its data and one stimulus of ten -- silently,
with the analysis reporting a clean result for the fraction it had seen.

A single-block recording, which is how a continuous run arrives, was never
affected: block 0 is the whole file.
"""

import ast
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
SMR = (PKG / "formats" / "spike2_smr.py").read_text(encoding="utf-8")
IO = (PKG / "io.py").read_text(encoding="utf-8")


def _body(name):
    tree = ast.parse(SMR)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found")


def test_there_is_a_loader_that_keeps_every_block():
    assert "def _load_all" in SMR
    assert "list(block.segments)" in _body("_load_all")


def test_the_waveform_spans_every_block():
    """It read block 0, which on a ten-block file is twelve per cent of the
    recording."""
    body = _body("extract_emg_waveform_and_fs")
    assert "_load_all(" in body
    assert "_load(file_path)" not in body


def test_the_blocks_are_placed_at_their_real_start_times():
    """A stimulus timestamp must mean the same thing in the trace as in the
    file, so the gaps between blocks are zero-filled rather than closed up."""
    body = _body("extract_emg_waveform_and_fs")
    assert "t_start" in body
    assert "np.zeros(total" in body


def test_a_single_block_recording_takes_the_simple_path():
    """A continuous run is the common case and must not pay for this."""
    body = _body("extract_emg_waveform_and_fs")
    assert "if len(usable) == 1:" in body


def test_events_come_from_every_block():
    body = _body("extract_stim_times")
    assert "_load_all(" in body
    assert "for sg in segments:" in body


def test_events_are_timed_from_the_start_of_the_recording():
    """Reading each block from its own zero would put every stimulus at the
    same place."""
    body = _body("extract_stim_times")
    assert "times_abs - t0" in body


def test_the_same_channel_is_read_in_each_block():
    """Not whichever channel happens to be first in that block."""
    body = _body("extract_stim_times")
    assert "c.name == target.name" in body


def test_metadata_still_reads_one_block():
    """What channels exist and what they are called is the same in every
    block, and reading ten to answer it would be waste."""
    for fn in ("list_waveform_channels", "get_channel_info"):
        assert "_load(file_path)" in _body(fn)


# ── epoch bounds ─────────────────────────────────────────────────────────────

def test_bounds_are_the_tightest_a_trial_can_supply():
    """A stimulus cannot be epoched past the edges of its own block.

    Beyond that lies the zero-fill between blocks and then the next trial, so
    the file-wide bound is the smallest per-trial limit -- the window no trial
    exceeds.
    """
    body = _body("get_epoch_bounds")
    assert "min(pres), min(posts)" in body


def test_bounds_do_not_require_blocks_to_be_stimulus_centred():
    """An earlier version reported bounds only when every stimulus sat at the
    same offset within its block and declined otherwise -- which left a
    paused-and-restarted recording free to read a second of padding as though
    it were signal. Where blocks ARE cut around the stimulus this returns the
    stored epoch anyway, by a more general route.
    """
    body = _body("get_epoch_bounds")
    assert "max(pres) - min(pres)" not in body


def test_a_single_block_recording_has_no_bounds():
    """It is continuous; there is nothing to clamp to."""
    body = _body("get_epoch_bounds")
    assert "len(segments) < 2" in body


def test_every_event_in_a_block_is_measured_not_only_the_first():
    body = _body("get_epoch_bounds")
    assert "for t in ch.times" in body


def test_an_event_outside_its_block_is_ignored():
    """A timestamp that does not sit inside the block it was read from cannot
    say anything about that block's limits."""
    body = _body("get_epoch_bounds")
    assert "a <= t <= b" in body


def test_no_events_at_all_declines():
    body = _body("get_epoch_bounds")
    assert "if not pres:" in body


def test_the_bounds_reach_io():
    assert "_spike2_smr.get_epoch_bounds(file_path)" in IO


def test_an_unreadable_file_declines_rather_than_raising():
    from mep_cmap.formats.spike2_smr import get_epoch_bounds
    assert get_epoch_bounds("/no/such/file.smr") is None


def test_segment_count_is_available_and_total():
    from mep_cmap.formats.spike2_smr import segment_count
    assert segment_count("/no/such/file.smr") == 1


# ── unlabelled events ────────────────────────────────────────────────────────

def test_an_empty_label_stays_empty():
    """A plain trigger channel labels none of its events.

    Turning that into "?" made every event look decoded-but-unreadable:
    _get_event_codes tests `any(lb != "")` before falling back to the channel
    name, so a list of "?" satisfied it and the documented fallback never ran.
    """
    from mep_cmap.formats.spike2_smr import _decode_marker_code
    assert _decode_marker_code("") == ""
    assert _decode_marker_code(b"") == ""


def test_real_codes_still_decode():
    from mep_cmap.formats.spike2_smr import _decode_marker_code
    assert _decode_marker_code("65") == "A"
    assert _decode_marker_code("B") == "B"
    assert _decode_marker_code(b"C") == "C"
    assert _decode_marker_code("DigMark") == "DigMark"


def test_the_fallback_names_the_channel():
    """'?' is a question the analyst cannot answer, carried into the trial
    file; the channel's own name says what the events are."""
    from mep_cmap.formats.spike2_smr import _channel_fallback_label
    assert _channel_fallback_label("trigger") == "trigger"
    assert _channel_fallback_label("  DigMark ") == "DigMark"
    assert _channel_fallback_label("") == "stim"
