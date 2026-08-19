"""
Block edges bound the epoch, in every format that has blocks.

Several acquisition systems record in blocks: Spike2 sampling segments,
LabChart blocks in both its text and MATLAB exports, Signal frames, and
pre-epoched MATLAB. The readers place blocks at their real times and fill the
gaps, so a stimulus cannot be epoched past the edges of its own block --
beyond lies that padding, and then the next trial.

The rule is the same everywhere: the file-wide bound is the smallest per-trial
limit, the window no trial exceeds. It does not depend on the blocks being cut
around the stimulus. An earlier version of the Spike2 reader made it
conditional on that, and declined for a paused-and-restarted recording, which
left the analysis free to read a second of padding as though it were signal.
"""

import pathlib
import re

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
IO = (PKG / "io.py").read_text(encoding="utf-8")

#: Readers whose files can hold more than one recording block.
BLOCK_FORMATS = ("spike2_smr", "labchart", "labchart_mat",
                 "signal_mat", "epoched_mat")


@pytest.mark.parametrize("fmt", BLOCK_FORMATS)
def test_every_block_format_reports_bounds(fmt):
    """A reader that stitches blocks and reports no bounds leaves nothing to
    clamp against."""
    mod = {"labchart_mat": "labchart_mat"}.get(fmt, fmt)
    src = (PKG / "formats" / f"{mod}.py").read_text(encoding="utf-8")
    assert "def get_epoch_bounds" in src, f"{mod}.py stitches blocks silently"


@pytest.mark.parametrize("fmt", BLOCK_FORMATS)
def test_io_dispatches_the_bounds(fmt):
    """The clamp asks io, not the reader; a reader nobody asks is no use."""
    i = IO.index("def get_epoch_bounds")
    j = IO.find("\ndef ", i + 10)
    body = IO[i:j if j > 0 else len(IO)]
    assert f"'{fmt}'" in body, f"io.get_epoch_bounds never returns for {fmt}"


@pytest.mark.parametrize("fmt", ("spike2_smr", "labchart_mat"))
def test_the_bound_is_the_tightest_not_the_typical(fmt):
    """Trials differ in how much room they have; the bound has to satisfy the
    worst of them."""
    src = (PKG / "formats" / f"{fmt}.py").read_text(encoding="utf-8")
    i = src.index("def get_epoch_bounds")
    j = src.index("\ndef ", i + 10)
    body = src[i:j]
    assert "min(pres), min(posts)" in body


@pytest.mark.parametrize("fmt", ("spike2_smr", "labchart_mat"))
def test_a_single_block_recording_is_continuous(fmt):
    src = (PKG / "formats" / f"{fmt}.py").read_text(encoding="utf-8")
    i = src.index("def get_epoch_bounds")
    j = src.index("\ndef ", i + 10)
    assert re.search(r"< 2:\s*\n\s*return None", src[i:j])


@pytest.mark.parametrize("fmt", ("spike2_smr", "labchart_mat"))
def test_bounds_never_raise(fmt):
    """Called on every load; an unreadable file must give None, not an error."""
    mod = __import__(f"mep_cmap.formats.{fmt}", fromlist=["get_epoch_bounds"])
    assert mod.get_epoch_bounds("/no/such/file") is None


def test_a_continuous_format_reports_nothing():
    """EDF, BrainVision and the rest are single continuous recordings; a bound
    invented for them would clamp a window that has no reason to be clamped."""
    i = IO.index("def get_epoch_bounds")
    j = IO.find("\ndef ", i + 10)
    body = IO[i:j if j > 0 else len(IO)]
    for fmt in ("edf", "brainvision", "acqknowledge_mat", "cfwb", "generic_tsv"):
        assert f"'{fmt}'" not in body


def test_the_clamp_consumes_whatever_io_reports():
    """One clamp, fed by one accessor: a format added to io is clamped
    everywhere without further wiring."""
    pipe = (PKG / "pipeline.py").read_text(encoding="utf-8")
    assert "def clamp_config_to_epoch_bounds" in pipe
    assert "clamp_window_map" in pipe


# ── the baseline field must not promise what the file cannot supply ──────────

APP = (PKG / "app.py").read_text(encoding="utf-8")


def _app_body(name):
    import ast
    tree = ast.parse(APP)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found")


def test_the_baseline_is_capped_at_what_the_recording_holds():
    """The clamp has always reduced prestim_ms at run time; the box went on
    saying 100, so the analyst had no way to know the analysis would use less.
    """
    body = _app_body("_apply_epoch_limit_to_prestim")
    assert "get_epoch_bounds" in body
    assert "self.prestim_ms.set(" in body


def test_the_field_is_not_disabled():
    """Asking for LESS baseline than the file can supply is a legitimate
    choice, and disabling the box would prevent it."""
    body = _app_body("_apply_epoch_limit_to_prestim")
    assert "state=" not in body
    assert "disabled" not in body


def test_the_limit_is_explained_beside_the_field():
    """A value that silently changed itself is worse than one that did not."""
    body = _app_body("_apply_epoch_limit_to_prestim")
    assert "_prestim_limit_note" in body
    assert "blocks" in body


def test_the_reduction_is_logged():
    body = _app_body("_apply_epoch_limit_to_prestim")
    assert "self.log(" in body


def test_a_continuous_recording_clears_the_note():
    """There is no limit to report, and a stale note from the previous file
    would describe a recording that is no longer open."""
    body = _app_body("_apply_epoch_limit_to_prestim")
    # ast.unparse normalises quotes; match the call, not the literal
    assert "note.config(text=" in body
    assert "if not bounds:" in body


def test_it_runs_when_a_file_is_opened():
    body = _app_body("_browse_file_path")
    assert "_apply_epoch_limit_to_prestim(fpath)" in body
