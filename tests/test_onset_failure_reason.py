"""
"Not detected" must say which kind of not-detected.

min_peak_amplitude gates all seven detectors: a trial whose response is smaller
than the gate is rejected before an onset is looked for. That is the gate doing
its job -- it is what stops an onset being fitted to noise -- but a rejected
trial and a trial whose latency window was looking in the wrong place both read
"Latency: not detected", and the two need opposite fixes.

Both happened on one recording in one session: a Vastus lateralis profile
(18-30 ms) on a tibialis anterior channel whose responses start near 32 ms, and
then, once that was corrected, a condition whose responses sit at 0.04-0.06 mV
against a 0.05 mV gate. Distinguishing them took two rounds of reading
waveforms.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = (ROOT / "mep_cmap" / "inspector.py").read_text(encoding="utf-8")


def _function(name, src=SRC):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found")


# ── the reason is worked out ─────────────────────────────────────────────────

def test_the_gate_is_checked_when_detection_fails():
    body = _function("_plot")
    assert "min_peak_amplitude" in body
    assert "_fail_reason" in body


def test_it_is_only_claimed_when_it_is_true():
    """Saying 'below the minimum' about a trial that is above it would send the
    analyst to the wrong setting."""
    assert "if _p2p < _gate:" in SRC


def test_the_measured_amplitude_is_quoted():
    """A number to compare against the threshold, so the analyst can judge how
    far to lower it rather than guessing."""
    assert "_p2p:.3f" in SRC and "_gate:.3f" in SRC


def _joined_strings(name, src=SRC):
    """String literals of one function, implicit concatenation joined.

    The message is wrapped mid-sentence across source lines, so a phrase can be
    correct at run time and absent from the file as a contiguous substring. The
    parser joins the adjacent literals; a plain `in SRC` check does not.
    """
    out = []
    for node in ast.walk(ast.parse(src)):
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


def _all_messages():
    """The reason text lives in _plot, where detection runs."""
    return _joined_strings("_plot")


def test_it_says_where_to_change_it():
    text = _all_messages()
    assert "Min peak amplitude" in text
    assert "Preferences" in text


def test_it_says_to_re_run():
    """Changing the setting does nothing to results already computed."""
    text = _all_messages()
    i = text.index("Min peak amplitude")
    assert "re-run" in text[i:i + 240]


def test_a_failure_to_work_out_the_reason_does_not_block_the_draw():
    """A status line is decoration next to a waveform."""
    body = _function("_plot")
    i = body.index("_fail_reason = None")
    assert "except Exception" in body[i:i + 1400]


# ── the reason is shown ──────────────────────────────────────────────────────

def test_the_status_line_carries_it():
    body = _function("_refresh_status")
    assert "onset_fail_reason" in body
    assert "not detected" in body


def test_a_plain_non_detection_still_reads_plainly():
    """Not every failure is the gate, and diluting the message would hide the
    ones that are not."""
    body = _function("_refresh_status")
    assert "'Latency: not detected'" in body or '"Latency: not detected"' in body


# ── it is not mistaken for data ──────────────────────────────────────────────

def test_the_reason_is_cleared_when_an_onset_is_found():
    assert "m.pop('onset_fail_reason', None)" in SRC


def test_the_reason_is_not_exported():
    """Explanatory prose re-derived on every draw. Kept, it would be stale text
    in the session file."""
    body = _function("_close_and_save")
    assert "_m.pop('onset_fail_reason', None)" in body
