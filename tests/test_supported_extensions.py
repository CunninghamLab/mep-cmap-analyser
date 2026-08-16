"""
One list of readable extensions, used everywhere.

The filter preview carried its own copy -- (".txt", ".smr", ".adibin") -- which
was never updated when EDF, BrainVision, MATLAB, AcqKnowledge and CSV support
arrived. Opening a .mat file there silently loaded nothing, so the sampling
rate was never set and the preview asked the analyst to type in a rate the file
had already declared. The M-wave reference dialogue had drifted the same way,
stopping at .bdf.

A stale copy of a list like this fails quietly: the file simply does not
appear, or a value the file supplies is asked for instead.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = (ROOT / "mep_cmap" / "app.py").read_text(encoding="utf-8")
PREVIEW = (ROOT / "mep_cmap" / "filter_preview.py").read_text(encoding="utf-8")


def test_the_canonical_list_covers_every_reader():
    from mep_cmap.io import SUPPORTED_EXTENSIONS

    for ext in (".txt", ".smr", ".adibin", ".edf", ".bdf",
                ".vhdr", ".acq", ".mat", ".csv"):
        assert ext in SUPPORTED_EXTENSIONS, f"{ext} is readable but not listed"


def test_the_filter_preview_uses_it():
    assert "_SUPPORTED_EXTS = SUPPORTED_EXTENSIONS" in PREVIEW
    assert 'SUPPORTED_EXTENSIONS' in PREVIEW.split("class ")[0], \
        "it must be imported, not redefined"
    assert '(".txt", ".smr", ".adibin")' not in PREVIEW, \
        "the stale copy is back"


def test_no_module_hardcodes_a_shorter_list():
    """
    Catches the specific shape of the bug: a literal tuple or string of
    extensions that stops before the full set.
    """
    from mep_cmap.io import SUPPORTED_EXTENSIONS

    for name, src in (("app.py", APP), ("filter_preview.py", PREVIEW)):
        for m in re.finditer(r'"\*\.txt[^"]*"', src):
            listed = set(re.findall(r"\*(\.\w+)", m.group(0)))
            if len(listed) <= 2:
                continue          # a narrow, deliberate filter
            missing = set(SUPPORTED_EXTENSIONS) - listed
            assert not missing, (
                f"{name} lists {sorted(listed)} but the readers also handle "
                f"{sorted(missing)}: {m.group(0)}"
            )


def test_the_mmax_dialogue_is_built_from_the_shared_list():
    a = APP.index("def _browse_mmax_file")
    b = APP.index("\n    def ", a + 10)
    body = APP[a:b]
    assert "SUPPORTED_EXTENSIONS" in body


# ── Files no reader can open ─────────────────────────────────────────────────

def test_an_unrecognised_binary_is_named_not_guessed_at():
    """
    detect_format ended with `return 'spike2'` as a catch-all, so ANY
    unrecognised file -- a Word document, a YAML file, a LabChart .adicht --
    was reported as a Spike2 text export and then failed somewhere downstream
    with a message naming the wrong format.

    A binary file that matched no magic is not a text export, and saying so is
    more use than a parse error from whichever reader happened to be tried
    last.
    """
    import tempfile

    from mep_cmap.io import detect_format

    with tempfile.NamedTemporaryFile(suffix=".adicht", delete=False) as f:
        f.write(b"A\x00D\x00I\x00n\x00s\x00t\x00r\x00u\x00m\x00e\x00n\x00t\x00")
        name = f.name
    try:
        assert detect_format(name) == "unsupported_binary"
    finally:
        pathlib.Path(name).unlink()


def test_a_text_export_is_still_detected_as_before():
    """The change must not catch text files, which have no NUL bytes."""
    import tempfile

    from mep_cmap.io import detect_format

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("Interval=\t0.0002 s\nChannelTitle=\tEMG\n")
        name = f.name
    try:
        assert detect_format(name) == "labchart"
    finally:
        pathlib.Path(name).unlink()


def test_known_unreadable_formats_explain_the_way_out():
    """
    Telling someone a file cannot be read is only half an answer; the useful
    half is what to do instead.
    """
    from mep_cmap.io import UNREADABLE_FORMATS

    assert ".adicht" in UNREADABLE_FORMATS
    for ext, why in UNREADABLE_FORMATS.items():
        assert len(why) > 40, f"{ext}: the explanation is too terse to act on"
        assert any(w in why.lower() for w in ("export", "open", "instead")), \
            f"{ext}: no alternative is offered"


def test_the_open_path_reports_it_rather_than_failing_later():
    a = APP.index('if _fmt == "unsupported_binary":')
    b = APP.index("if _fmt == 'generic_tsv'", a)
    body = APP[a:b]
    assert "UNREADABLE_FORMATS" in body
    assert "showerror" in body
    assert "return" in body, "the load must stop here, not continue"
