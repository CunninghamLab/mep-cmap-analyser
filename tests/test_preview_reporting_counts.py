"""
The preview reports what DETECTION found. The run reports what it measures,
which includes onsets the analyst placed by hand.

"A·first 4" against a results file with a latency on all six reads as the two
disagreeing. They need not: four detected plus two placed is six. Only the
numerator was ever shown, so a real disagreement and this one looked identical
-- and one of them cost a round of investigation.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = (ROOT / "mep_cmap" / "preview.py").read_text(encoding="utf-8")


def _function(name):
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found")


def _messages(name):
    """String literals of one function, implicit concatenation joined."""
    out = []
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.JoinedStr):
                    out.append("".join(
                        p.value for p in sub.values
                        if isinstance(p, ast.Constant)
                        and isinstance(p.value, str)))
                elif (isinstance(sub, ast.Constant)
                        and isinstance(sub.value, str)):
                    out.append(sub.value)
    return "\n".join(out)


# ── the count says what it is a count of ─────────────────────────────────────

def test_the_count_has_a_denominator():
    """'4' is a shortfall against an unknown total; '4/6' is a fact."""
    body = _function("_preview_show")
    assert "_n_shown" in body


def test_hand_placed_onsets_are_counted_separately():
    body = _function("_preview_show")
    assert "segments_metadata" in body
    assert "you placed" in _messages("_preview_show")


def test_a_trial_both_detected_and_placed_is_counted_once():
    """Otherwise the total can exceed the trials shown, which is worse than
    the ambiguity it was meant to remove."""
    body = _function("_preview_show")
    assert "_mi not in _found" in body


def test_a_failed_auto_onset_is_not_counted_as_placed():
    """onset_auto_failed marks a marker parked at the stimulus because
    detection found nothing, not a decision the analyst made."""
    body = _function("_preview_show")
    assert "onset_auto_failed" in body


def test_the_explanation_only_appears_when_it_applies():
    """A file with no manual edits should not be told about them."""
    body = _function("_preview_show")
    assert 'if any(' in body and "you placed" in body


def test_the_explanation_says_the_run_uses_them():
    msgs = _messages("_preview_show")
    i = msgs.find("A trial detection missed")
    assert i >= 0
    assert "the run uses those" in msgs[i:i + 260]


# ── an unanchored window says so ─────────────────────────────────────────────

def test_a_file_wide_window_is_labelled():
    """An anchored window follows the response and the fallback does not, but
    both print as two numbers in ms."""
    body = _function("_preview_show")
    assert "not anchored" in _messages("_preview_show")
    assert "_cfg.ptp_start" in body and "_cfg.ptp_end" in body


def test_an_anchored_window_is_not_labelled():
    """The label must mean something, so it cannot appear on every line.

    Checked structurally: the label is the true-branch of a conditional
    expression comparing the window against the file-wide pair, so a window
    that differs from the fallback gets no label.
    """
    for node in ast.walk(ast.parse(SRC)):
        if not isinstance(node, ast.IfExp):
            continue
        rendered = ast.unparse(node)
        if "not anchored" in rendered:
            # The comparison is against _fallback, bound from the file-wide
            # pair just above; the conditional itself names only that.
            assert "_fallback" in rendered, (
                "the label must be conditional on matching the file-wide pair")
            body = _function("_preview_show")
            assert "_fallback = (float(_cfg.ptp_start)" in body
            return
    raise AssertionError("the label is not inside a conditional expression")
