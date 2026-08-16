"""
The pipeline reaches the Data Inspector through three hops, and they must agree.

    pipeline.run_pipeline  ──kwargs──▶  app._show_inspector_cb
                                            │ puts a POSITIONAL tuple on a queue
                                            ▼
                                        app._open_inspector_gui(*payload)
                                            │
                                            ▼
                                        DataInspectorWindow(...)

Adding an argument means changing all three. Missing the middle one raises
`TypeError: _show_inspector_cb() got an unexpected keyword argument`, and only
when an analysis actually reaches the inspector -- so it imports cleanly, the
suite passes, and the failure appears in front of the analyst.

That has now happened twice in this project: once passing detection parameters
into run_pipeline, once passing the per-stimulus-type amplitude window into the
inspector. These tests walk the chain instead of checking either end.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = (ROOT / "mep_cmap" / "app.py").read_text(encoding="utf-8")
PIPE = (ROOT / "mep_cmap" / "pipeline.py").read_text(encoding="utf-8")
INSP = (ROOT / "mep_cmap" / "inspector.py").read_text(encoding="utf-8")


def _params(src, func, cls=None):
    """Parameter names of a function, optionally scoped to a class.

    The scoping matters: inspector.py defines several classes and walking the
    module for the first `__init__` finds DraggablePoint's, not the window's.
    """
    tree = ast.parse(src)
    if cls is not None:
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef) and n.name == cls:
                tree = n
                break
        else:
            raise AssertionError(f"class {cls} not found")
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == func:
            return [a.arg for a in n.args.args if a.arg != "self"] + \
                   [a.arg for a in n.args.kwonlyargs]
    raise AssertionError(f"{func} not found")


def test_the_two_app_signatures_match_exactly():
    """
    The payload is unpacked positionally, so the order matters as much as the
    names: a mismatch would deliver the wrong value to the right parameter
    without any error at all.
    """
    assert _params(APP, "_show_inspector_cb") == _params(APP, "_open_inspector_gui")


def test_the_queued_payload_has_one_item_per_parameter():
    i = APP.index('self.msg_q.put(("show-inspector"')
    j = APP.index("))", i)
    body = APP[i:j]
    items = body.count(",")            # commas after the tag == items - 1 + tag
    assert items == len(_params(APP, "_open_inspector_gui")), (
        "the queued tuple and the GUI signature have drifted apart"
    )


def test_every_kwarg_the_pipeline_sends_is_accepted():
    accepted = set(_params(APP, "_show_inspector_cb"))
    tree = ast.parse(PIPE)
    calls = 0
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Name) and f.id == "show_inspector_cb":
            calls += 1
            for kw in n.keywords:
                assert kw.arg in accepted, (
                    f"pipeline passes '{kw.arg}' but _show_inspector_cb does "
                    f"not accept it"
                )
    assert calls >= 1, "no show_inspector_cb call found to check"


def test_every_kwarg_app_passes_to_the_inspector_is_accepted():
    accepted = set(_params(INSP, "__init__", cls="DataInspectorWindow"))
    tree = ast.parse(APP)
    checked = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and n.func.id == "DataInspectorWindow":
            checked = True
            for kw in n.keywords:
                if kw.arg is None:
                    continue
                assert kw.arg in accepted, (
                    f"app passes '{kw.arg}' but DataInspectorWindow does not "
                    f"accept it"
                )
    assert checked, "no DataInspectorWindow call found to check"


def test_the_newest_argument_is_present_at_every_hop():
    name = "ptp_windows_by_type"
    assert name in _params(APP, "_show_inspector_cb")
    assert name in _params(APP, "_open_inspector_gui")
    assert name in _params(INSP, "__init__", cls="DataInspectorWindow")
    assert f"{name}=_ptp_ms_by_type" in PIPE


def test_an_inspector_failure_does_not_hang_the_analysis():
    """
    The worker thread blocks on `while self._last_outlier_result is None`. If
    opening the window raises, nothing ever sets that, so the analysis stops
    dead with no window and no message -- which is indistinguishable, from the
    analyst's side, from the tool having frozen.

    The handler must report the failure and release the worker so the run
    completes with automatic markers.
    """
    a = APP.index('elif msg == "show-inspector":')
    b = APP.index('elif msg == "bidsify-convert-done":', a)
    body = APP[a:b]
    assert "try:" in body and "except Exception" in body
    assert "self._last_outlier_result = {}" in body, (
        "the worker must be released or the analysis hangs"
    )
    assert "format_exc()" in body, "the traceback must reach the console"
    assert "self._log_gui(" in body, "the failure must reach the in-app log"
