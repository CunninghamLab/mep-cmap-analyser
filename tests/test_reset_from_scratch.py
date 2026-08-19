"""
Reset & reprocess from scratch.

"From scratch" has to mean it. The tool writes several files beside a
recording -- format sidecars, a saved session, and the events file holding
assigned conditions -- and one left behind governs the next run while the
analyst believes nothing does.

The events file is the one that matters most: the readers PREFER a sibling
_events.tsv to the recording's own markers, so a stale one silently decides
which stimuli are analysed.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = (ROOT / "mep_cmap" / "app.py").read_text(encoding="utf-8")


def _body(name):
    tree = ast.parse(APP)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found")


RESET = _body("_queue_reset_file")


def test_every_format_sidecar_is_removed():
    """One left behind reuses the configuration being discarded."""
    for sidecar in (".smr_config.json", ".tsv_config.json",
                    ".epoched_config.json"):
        assert sidecar in RESET, f"{sidecar} survives a reset"


def test_the_saved_session_is_removed():
    assert "_session.json" in RESET


def test_the_derivatives_output_is_removed():
    assert "derivatives_json" in RESET


def test_the_events_file_is_asked_about_separately():
    """It is the analyst's own work, not something the tool can produce again.

    Deleting it silently under a button labelled "reset" would take an
    afternoon of condition assignment with it.
    """
    assert "_also_events" in RESET
    assert "askyesno" in RESET


def test_the_events_question_says_what_keeping_it_means():
    """'Delete these too?' with no consequence stated is not a question anyone
    can answer."""
    assert "prefers an" in RESET or "prefers" in RESET
    assert "govern the next run" in RESET


def test_the_events_sidecar_goes_with_the_events_file():
    """A .json describing columns that no longer exist is worse than neither."""
    assert 'with_suffix(\'.json\')' in RESET or 'with_suffix(".json")' in RESET


def test_the_events_path_comes_from_the_writer():
    """Two rules for the same filename drift, and the one that drifts here
    leaves the file the reader will still find."""
    assert "events_tsv_path" in RESET


def test_a_file_that_cannot_be_deleted_is_reported_not_ignored():
    assert "skipped.append" in RESET


def test_the_file_entry_state_is_cleared():
    """Otherwise the queue still shows it as processed."""
    assert "STATUS_NOT_STARTED" in RESET
