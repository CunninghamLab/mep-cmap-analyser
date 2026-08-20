"""
The onset detector must be told where the stimulus sits in the trace it is
given.

`pre_ms` is what locates t=0 for dispatch_onset. It was cfg.pre_ms
unconditionally, and the docstring said it had to be -- true while one epoch
served the whole file, wrong once epochs became per stimulus type.
pipeline_extract_segments cuts with window_samples(cfg, stim_type), so a type
given a 100 ms epoch against a file-wide 20 ms has its stimulus at 100 ms while
cfg.pre_ms says 20.

The docstring predicted the symptom exactly: "returns None even when a clear
MEP is present". The preview, which trims to the same per-type window, found no
onset on any trial and then reported that the run would find none either.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = (ROOT / "mep_cmap" / "pipeline.py").read_text(encoding="utf-8")


def _function(name):
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found")


def test_the_dispatcher_can_be_told_the_stimulus_type():
    body = _function("_detect_onset_dispatch")
    assert "stim_type" in body
    assert "resolve_window(cfg, stim_type)" in body


def test_it_falls_back_to_the_file_wide_pre():
    """For a caller whose trace really is cut to the file-wide window."""
    body = _function("_detect_onset_dispatch")
    assert "else cfg.pre_ms" in body


def test_the_resolved_pre_is_what_is_passed():
    body = _function("_detect_onset_dispatch")
    assert "pre_ms=_pre_ms" in body
    assert "pre_ms=cfg.pre_ms," not in body


@pytest.mark.parametrize("caller", ["pipeline_detect_onsets"])
def test_the_detection_pass_names_its_type(caller):
    """Both calls in here know the stim type; neither may omit it."""
    body = _function(caller)
    n_calls = body.count("_detect_onset_dispatch(")
    n_typed = body.count("stim_type=stim_type")
    assert n_calls >= 2
    assert n_typed == n_calls, (
        f"{n_calls} dispatch call(s) but {n_typed} pass stim_type")


def test_no_call_site_omits_the_type():
    """The guard. Every caller in this module has a stim type to hand, so a
    call without one is an oversight rather than a considered fallback."""
    tree = ast.parse(SRC)
    bare = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Name)
                and fn.id == "_detect_onset_dispatch"):
            continue
        if not any(k.arg == "stim_type" for k in node.keywords):
            bare.append(ast.unparse(node)[:80])
    assert bare == [], f"dispatch call(s) without stim_type: {bare}"


def test_the_resolution_matches_how_segments_are_cut():
    """resolve_window and window_samples must agree, or the detector is told
    one thing and the cutter does another."""
    from mep_cmap.pipeline import (PipelineConfig, resolve_window,
                                   window_samples)
    cfg = PipelineConfig()
    cfg.pre_ms, cfg.post_ms = 20.0, 300.0
    cfg.window_map = {"A": (100.0, 300.0)}
    fs = 2000.0
    pre_ms = resolve_window(cfg, "A")[0]
    assert pre_ms == 100.0
    assert int(pre_ms * fs / 1000) == window_samples(cfg, "A", fs)[0]


def test_a_type_without_its_own_window_uses_the_file_wide_one():
    from mep_cmap.pipeline import PipelineConfig, resolve_window
    cfg = PipelineConfig()
    cfg.pre_ms, cfg.post_ms = 20.0, 300.0
    cfg.window_map = {"A": (100.0, 300.0)}
    assert resolve_window(cfg, "B")[0] == 20.0
