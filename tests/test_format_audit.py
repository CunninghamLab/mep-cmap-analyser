"""
Every format reaches every stage of the workflow.

A format is not "supported" because a reader exists for it. It has to be
detected, have its channels listed, its events read, its waveform extracted,
be given a branch in the load flow, and reach the channel assignment dialogue.
A reader wired into four of those six is a format that works until the day
someone uses the fifth.
"""

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
IO = (ROOT / "mep_cmap" / "io.py").read_text(encoding="utf-8")
APP = (ROOT / "mep_cmap" / "app.py").read_text(encoding="utf-8")

#: Values detect_format returns to say "this is not a recording".
SENTINELS = {"unsupported_binary", "unsupported_text"}


def _formats():
    a = IO.index("def detect_format")
    b = IO.index("\ndef ", a + 10)
    return sorted(set(re.findall(r"return '([a-z_0-9]+)'", IO[a:b])) - SENTINELS)


def _handled(fn, src=IO, var="fmt"):
    i = src.index(f"def {fn}")
    j = src.find("\ndef ", i + 10)
    body = src[i:j if j > 0 else len(src)]
    out = set(re.findall(rf"{var} == '([a-z_0-9]+)'", body))
    for grp in re.findall(rf"{var} in \(([^)]*)\)", body):
        out |= set(re.findall(r"'([a-z_0-9]+)'", grp))
    return out


#: The Spike2 text reader is the fallback every unrecognised-but-numeric text
#: file reaches, so it is handled by falling through rather than by name.
FALLTHROUGH = {"spike2"}


@pytest.mark.parametrize("fmt", _formats())
def test_every_format_lists_its_channels(fmt):
    assert fmt in _handled("list_waveform_channels") | FALLTHROUGH


@pytest.mark.parametrize("fmt", _formats())
def test_every_format_extracts_a_waveform(fmt):
    assert fmt in _handled("_extract_emg_native") | FALLTHROUGH


@pytest.mark.parametrize("fmt", _formats())
def test_every_format_extracts_its_events(fmt):
    assert fmt in _handled("extract_stim_times") | FALLTHROUGH


@pytest.mark.parametrize("fmt", _formats())
def test_every_format_has_a_load_branch(fmt):
    """Without one the file loads and the setup table is built from nothing."""
    assert fmt in _handled("_browse_file_path", src=APP, var="_fmt")


def test_every_format_reaches_channel_assignment():
    """It ran for Spike2 text exports alone, so a six-channel LabChart export
    was analysed on whichever channel came first."""
    assert "_needs_assign_dlg = (_fmt != 'spike2_smr')" in APP


# ── the sentinels ────────────────────────────────────────────────────────────

def test_text_without_data_rows_is_declined():
    """A README or a settings file used to be claimed as a Spike2 export and
    fail several steps later with a bare ValueError from inside a parser."""
    from mep_cmap.io import detect_format
    assert detect_format(str(ROOT / "README.md")) == "unsupported_text"
    assert detect_format(str(ROOT / "pyproject.toml")) == "unsupported_text"


def test_a_real_export_is_not_declined(tmp_path):
    """The check is deliberately generous: a false yes merely restores the
    previous behaviour, while a false no would refuse a recording."""
    from mep_cmap.io import _has_numeric_rows
    p = tmp_path / "d.txt"
    p.write_text("some header line\nand another\n0.0\t1.23\t4.56\n", encoding="utf-8")
    assert _has_numeric_rows(str(p)) is True


def test_an_unreadable_file_is_not_mistaken_for_a_non_recording():
    from mep_cmap.io import _has_numeric_rows
    assert _has_numeric_rows("/no/such/file.txt") is True


def test_both_sentinels_stop_the_load():
    """Continuing past either would build a setup table from nothing."""
    for name in ("unsupported_text", "unsupported_binary"):
        i = APP.index(f'if _fmt == "{name}":')
        j = APP.index("return", i)
        assert "showerror" in APP[i:j]


def test_the_declining_message_says_which_problem_it_is():
    """'Not in a format the tool can read' is true of a corrupt export too."""
    assert "text file with no data rows" in IO


def test_a_reader_with_no_dispatch_is_caught():
    """kinemg_csv shipped with the full three-function contract and
    detect_format never returned it, so the format was supported by every
    measure except being reachable: a KinEMG export fell through to the
    generic TSV path or the Spike2 fallback depending on its punctuation.

    Every module in formats/ that implements the contract must be dispatched.
    """
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap" / "formats"
    unreachable = []
    for path in sorted(pkg.glob("*.py")):
        if path.name.startswith("_"):
            continue
        src = path.read_text(encoding="utf-8")
        implements = all(f"def {fn}" in src for fn in
                         ("list_waveform_channels", "extract_emg_waveform_and_fs",
                          "extract_stim_times"))
        if not implements:
            continue
        stem = path.stem
        # mne_bridge is dispatched under the name 'mne'; spike2 is the fallback.
        alias = {"mne_bridge": "mne"}.get(stem, stem)
        if f"'{alias}'" not in IO and alias != "spike2":
            unreachable.append(path.name)
    assert not unreachable, (
        "these readers implement the contract and detect_format never returns "
        "them: " + ", ".join(unreachable))
