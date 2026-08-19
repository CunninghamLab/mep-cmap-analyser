"""
LabChart block exports are pre-epoched recordings that do not say so.

Each block is a trial already cut about the stimulus, its time column running
from a negative value to a positive one and restarting for the next.
``TimeFormat=StartOfBlock`` names that convention, and the format is otherwise
indistinguishable from a continuous export.

Reporting the bounds is what lets the ordinary clamp apply. Without them a
window longer than a block runs off the end of a trial into the zero-fill this
reader inserts between blocks, and then into the following trial -- measured as
though it were a continuation of the response rather than the next stimulus.
"""

import pytest

HEAD = ("Interval=\t0.00025 s\n"
        "ExcelDateTime=\t4.6141458355493545e+004\t29/04/2026 11:00:01\n"
        "TimeFormat=\tStartOfBlock\n"
        "DateFormat=\t\n"
        "ChannelTitle=\tCh 1 - MEP\tCh 2 - Stim\n"
        "Range=\t112.50 mV\t10.000 V\n"
        "UnitName=\tmV\tV\n"
        "TopValue=\t*\t*\n"
        "BottomValue=\t*\t*\n")


def _block(t0, t1, step=0.00025):
    rows, t = [], t0
    while t < t1 - 1e-9:
        rows.append(f"{t:.5f}\t0.01\t0.0\n")
        t += step
    return HEAD + "".join(rows)


@pytest.fixture
def blocked(tmp_path):
    p = tmp_path / "sub-01_task-x_emg.txt"
    p.write_text(_block(-0.2, 0.5) * 1, encoding="utf-8")
    # two blocks, each -200 to +500 ms
    p.write_text(_block(-0.2, 0.5) + _block(-0.2, 0.5), encoding="utf-8")
    return str(p)


@pytest.fixture
def continuous(tmp_path):
    p = tmp_path / "sub-02_task-x_emg.txt"
    p.write_text(_block(0.0, 1.0), encoding="utf-8")
    return str(p)


def test_a_block_export_reports_its_epoch(blocked):
    from mep_cmap.formats.labchart import get_epoch_bounds
    pre, post = get_epoch_bounds(blocked)
    assert pre == pytest.approx(200.0)
    assert post == pytest.approx(500.0, abs=1.0)


def test_a_continuous_export_reports_nothing(continuous):
    """One block is an ordinary recording with no bounds to clamp to."""
    from mep_cmap.formats.labchart import get_epoch_bounds
    assert get_epoch_bounds(continuous) is None


def test_a_block_not_centred_on_a_stimulus_reports_nothing(tmp_path):
    """A time column that does not start before zero is not centred on
    anything, so there is no pre-stimulus extent to report."""
    from mep_cmap.formats.labchart import get_epoch_bounds
    p = tmp_path / "sub-03_task-x_emg.txt"
    p.write_text(_block(0.0, 0.5) + _block(0.0, 0.5), encoding="utf-8")
    assert get_epoch_bounds(str(p)) is None


def test_the_bounds_reach_io(blocked):
    """The clamp asks io, not the reader."""
    from mep_cmap.io import get_epoch_bounds
    assert get_epoch_bounds(blocked) is not None


def test_an_over_long_window_is_clamped(blocked):
    """Reading past the block means zero-fill, then the next trial."""
    from mep_cmap.io import get_epoch_bounds
    from mep_cmap.pipeline import clamp_config_to_epoch_bounds

    cfg, changes = clamp_config_to_epoch_bounds(
        dict(pre_ms=20, post_ms=10000, prestim_ms=100,
             window_map={"A": (None, 10000.0)}),
        get_epoch_bounds(blocked))
    # The last sample sits one interval short of the nominal end, so the
    # bound is 499.75 rather than 500: it reports what the block CONTAINS,
    # which is the number the clamp needs.
    assert cfg["post_ms"] == pytest.approx(500, abs=1)
    assert cfg["window_map"]["A"][1] == pytest.approx(500.0, abs=1.0)
    assert any("post_ms" in str(c[0]) for c in changes), \
        "the shortening must be reported, not applied silently"


def test_a_missing_or_unreadable_file_reports_nothing():
    from mep_cmap.formats.labchart import get_epoch_bounds
    assert get_epoch_bounds("/no/such/file.txt") is None
