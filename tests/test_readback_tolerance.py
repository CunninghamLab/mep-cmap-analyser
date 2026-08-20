"""
The read-back check must accept integer storage and reject real corruption.

EDF stores integers, so every written sample differs from the source by less
than one quantisation step. The RMS therefore cannot move by more than a step
either. Anything inside that bound is storage; anything outside it is not.

The numbers here are measured from a real conversion (sub-002, EMG 2), where a
correct write was reported as a failure: pyedflib truncates rather than rounds,
leaving a systematic offset of about half a step, and that offset beats against
the channel's own DC mean to shift the RMS by far more than a relative
tolerance allows.
"""

import pytest

from mep_cmap.recording import compare_signatures


LSB = 0.00034194          # measured: EMG 2, phys range -5.56031..16.8488
SRC = 0.049581915         # measured source RMS
WRITTEN = 0.049721714     # measured written RMS


def _sig(rms, lsb=LSB, name="EMG 2", n=100):
    return {"n_channels": 1, "sampling_frequency": 5000.0,
            "samples_per_record": 5000,
            "channels": [{"name": name, "n_samples": n, "rms": rms,
                          "lsb": lsb}]}


def _ref(rms=SRC, name="EMG 2", n=100):
    return {"n_channels": 1, "sampling_frequency": 5000.0,
            "channels": [{"name": name, "n_samples": n, "rms": rms}]}


def test_a_real_conversion_verifies():
    """The exact numbers that were being reported as a failure."""
    ok, disc = compare_signatures(_ref(), _sig(WRITTEN))
    assert ok, disc


def test_a_difference_of_one_step_is_the_boundary():
    ok, _ = compare_signatures(_ref(), _sig(SRC + LSB * 0.99))
    assert ok


def test_a_difference_beyond_one_step_still_fails():
    """The bound is a bound, not a licence."""
    ok, disc = compare_signatures(_ref(), _sig(SRC + LSB * 1.5))
    assert not ok
    assert "quantisation step" in disc[0]


@pytest.mark.parametrize("factor", [0.5, 0.8, 1.2, 2.0])
def test_a_gain_error_fails(factor):
    """Orders of magnitude outside the bound, which is the point."""
    ok, _ = compare_signatures(_ref(), _sig(SRC * factor))
    assert not ok


def test_a_silent_channel_fails():
    ok, _ = compare_signatures(_ref(), _sig(0.0))
    assert not ok


def test_without_a_step_the_old_strict_behaviour_holds():
    """Older pyedflib cannot report the physical range. A file that cannot be
    checked properly must not be waved through."""
    ok, _ = compare_signatures(_ref(), _sig(WRITTEN, lsb=0.0))
    assert not ok
    ok, _ = compare_signatures(_ref(), _sig(SRC, lsb=0.0))
    assert ok


def test_a_coarse_channel_gets_a_proportionally_wider_bound():
    """A channel carrying a stimulus artefact has a wide range and a coarse
    step, and its bound widens with it rather than being a fixed percentage."""
    coarse = 0.0129
    ok, _ = compare_signatures(_ref(), _sig(SRC + coarse * 0.9, lsb=coarse))
    assert ok
    ok, _ = compare_signatures(_ref(), _sig(SRC + coarse * 1.1, lsb=coarse))
    assert not ok


def test_other_checks_are_unaffected():
    ref = _ref()
    bad = _sig(WRITTEN)
    bad["sampling_frequency"] = 1000.0
    ok, disc = compare_signatures(ref, bad)
    assert not ok and "sampling frequency" in disc[0]
