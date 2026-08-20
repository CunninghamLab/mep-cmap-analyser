"""
The worklist must not offer the tool its own output back.

Conversion writes into the same tree that is being scanned when the output root
sits inside the study folder, which is the ordinary layout. Without this, every
conversion adds a file to the queue, and the duplicates reach the analysis
queue too, where they look like extra recordings for the same participant.

The rule must also not hide real data, which is the harder half: EDF and BDF
are input formats this tool legitimately reads.
"""

import os

import pytest

from mep_cmap.bidsify_tab import _BIDS_DATATYPE_DIRS


class _Tab:
    """Just the predicate, without building a window."""
    from mep_cmap.bidsify_tab import BidsifyTabMixin as _M  # noqa: N814
    _bidsify_is_own_output = _M._bidsify_is_own_output


def _p(*parts):
    return os.path.join("C:" + os.sep, "study", *parts)


tab = _Tab()


# ── excluded: the tool's own output ──────────────────────────────────────────

@pytest.mark.parametrize("path", [
    _p("rawdata", "sourcedata", "sub-002", "ses-1", "a.smr"),
    _p("rawdata", "sub-001", "ses-01", "emg", "sub-001_ses-01_emg.edf"),
    _p("rawdata", "sub-002", "ses-1", "emg", "sub-002_ses-1_emg.edf"),
    _p("rawdata", "sub-001", "emg", "sub-001_emg.edf"),          # no session
    _p("derivatives", "sub-001", "trials.csv"),
    _p("rawdata", "sub-001", "ses-01", "nibs", "x_nibs.tsv"),
    _p("rawdata", "sub-001", "ses-01", "eeg", "x_eeg.edf"),
])
def test_tool_output_is_excluded(path):
    assert tab._bidsify_is_own_output(path) is True


# ── kept: real source recordings ─────────────────────────────────────────────

@pytest.mark.parametrize("path", [
    _p("rawdata", "sub-001", "pilot", "sub-001_limb-right.smr"),
    _p("rawdata", "sub-002", "Spike", "sub-002_ses-1_limb-right.smr"),
    _p("raw", "sub-003", "session1", "recording.smr"),
    # EDF as a SOURCE format, which the tool reads. Excluding by extension
    # would hide this.
    _p("incoming", "participant7", "day2.edf"),
    _p("incoming", "emg", "day2.edf"),        # 'emg' without a sub-/ses- parent
    _p("myemgstudy", "sub-001", "notes.smr"),
])
def test_real_sources_are_kept(path):
    assert tab._bidsify_is_own_output(path) is False


def test_an_analysts_own_emg_folder_is_not_hidden():
    """The sub-/ses- parent requirement is what makes this safe."""
    assert tab._bidsify_is_own_output(_p("emg", "recording.smr")) is False
    assert tab._bidsify_is_own_output(_p("data", "emg", "rec.edf")) is False


def test_a_short_path_does_not_raise():
    assert tab._bidsify_is_own_output("a.smr") is False
    assert tab._bidsify_is_own_output("") is False


def test_case_does_not_matter():
    """Windows paths arrive in whatever case the analyst typed."""
    assert tab._bidsify_is_own_output(
        _p("rawdata", "SUB-001", "SES-01", "EMG", "x.edf")) is True
    assert tab._bidsify_is_own_output(
        _p("RawData", "SourceData", "sub-1", "a.smr")) is True


def test_the_datatype_list_covers_what_this_tool_writes():
    assert {"emg", "nibs"} <= _BIDS_DATATYPE_DIRS
