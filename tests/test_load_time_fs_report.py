"""
The sampling rate and amplitude unit must be reported when a file is opened.

Every reader returns both, but nothing surfaced them until the analysis ran and
printed "(fs=... Hz, ...)". Opening a file therefore gave no confirmation of
what had been detected, which looks the same as nothing having been detected.

They are also the two values most worth checking before committing to an
analysis: a wrong rate silently rescales every latency, and a wrong unit every
amplitude. Neither error announces itself in the results.
"""

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = (ROOT / "mep_cmap" / "app.py").read_text(encoding="utf-8")
IO = (ROOT / "mep_cmap" / "io.py").read_text(encoding="utf-8")


def test_the_probe_helper_exists_and_is_exported():
    from mep_cmap.io import probe_fs_and_unit

    assert callable(probe_fs_and_unit)


def test_the_probe_discards_the_waveform():
    """It is called on every file open, including very long recordings."""
    a = IO.index("def probe_fs_and_unit")
    body = IO[a:]
    assert "del wave" in body, "the array must not be retained"
    assert "return (int(fs) if fs else None), unit" in body


def test_the_open_path_reports_the_rate():
    a = APP.index("_fmt = detect_format(fpath)")
    b = APP.index("if _fmt == 'generic_tsv'", a)
    body = APP[a:b]
    assert "probe_fs_and_unit(fpath)" in body
    assert "Sampling rate" in body


def test_a_reader_that_cannot_answer_is_reported_not_swallowed():
    """
    Some formats need a channel assignment before they can be read. Failing
    silently would leave the same blank the change is meant to fill.
    """
    a = APP.index("probe_fs_and_unit(fpath)")
    b = APP.index("if _fmt == 'generic_tsv'", a)
    body = APP[a:b]
    assert "except Exception" in body
    assert "Could not read the sampling rate" in body


def test_a_failed_probe_does_not_stop_the_file_loading():
    """The rate is read again when the analysis runs; a probe failure is cosmetic."""
    a = APP.index("probe_fs_and_unit(fpath)")
    b = APP.index("if _fmt == 'generic_tsv'", a)
    body = APP[a:b]
    assert "return" not in body.split("except Exception")[1].split("\n\n")[0]


@pytest.mark.parametrize("name,expected_fs", [
    ("A025_V2_CS_RC_iSP_120MEP_B.mat", 4000),
])
def test_probe_reads_a_labchart_mat_export(name, expected_fs):
    """
    LabChart exports carry the rate in `samplerate` (per channel, per block)
    and `tickrate`. This file prompted the change: the rate was being read
    correctly all along and simply never shown.
    """
    p = pathlib.Path("/mnt/user-data/uploads") / name
    if not p.exists():
        pytest.skip("sample file not present in this environment")
    from mep_cmap.io import probe_fs_and_unit

    fs, unit = probe_fs_and_unit(str(p))
    assert fs == expected_fs
    assert unit
