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
