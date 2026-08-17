"""
Per-stimulus-type epoch windows.

The epoch a response needs is a property of the response. A cortical silent
period wants several hundred milliseconds after the pulse; an M-wave wants a
few tens. Forcing both to share one window means either truncating the first or
carrying an order of magnitude of unnecessary samples through every trial of
the second.

The governing property throughout is that an empty window_map reproduces the
single-window behaviour exactly. There is one code path and the shared window
is its degenerate case, which is what makes the change reviewable rather than a
rewrite.
"""

import ast
import pathlib

import numpy as np
import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
APP = (PKG / "app.py").read_text(encoding="utf-8")
PIPE = (PKG / "pipeline.py").read_text(encoding="utf-8")
INSP = (PKG / "inspector.py").read_text(encoding="utf-8")


def _cfg(**kw):
    from mep_cmap.pipeline import PipelineConfig
    base = dict(pre_ms=100.0, post_ms=400.0)
    base.update(kw)
    return PipelineConfig(**base)


# ── resolution ───────────────────────────────────────────────────────────────

def test_a_type_not_in_the_map_uses_the_file_wide_window():
    from mep_cmap.pipeline import resolve_window
    assert resolve_window(_cfg(), "A") == (100.0, 400.0)


def test_an_empty_map_is_the_current_behaviour():
    """The safety property the whole change rests on."""
    from mep_cmap.pipeline import resolve_window
    cfg = _cfg(window_map={})
    for stim in ("A", "B", "anything"):
        assert resolve_window(cfg, stim) == (cfg.pre_ms, cfg.post_ms)


def test_a_mapped_type_uses_its_own_window():
    from mep_cmap.pipeline import resolve_window
    cfg = _cfg(window_map={"cSP": (100.0, 500.0), "Mwave": (20.0, 50.0)})
    assert resolve_window(cfg, "cSP") == (100.0, 500.0)
    assert resolve_window(cfg, "Mwave") == (20.0, 50.0)
    assert resolve_window(cfg, "other") == (100.0, 400.0)


def test_one_blank_side_falls_back_for_that_side_only():
    """The table has two boxes and an analyst may fill in one.

    Coercing a blank to zero would epoch a type over no time at all.
    """
    from mep_cmap.pipeline import resolve_window
    cfg = _cfg(window_map={"A": (None, 500.0), "B": (25.0, None)})
    assert resolve_window(cfg, "A") == (100.0, 500.0)
    assert resolve_window(cfg, "B") == (25.0, 400.0)


def test_sample_counts_follow_the_window():
    from mep_cmap.pipeline import window_samples
    cfg = _cfg(window_map={"short": (20.0, 50.0)})
    assert window_samples(cfg, "short", 1000.0) == (20, 50)
    assert window_samples(cfg, "long", 1000.0) == (100, 400)


def test_the_axis_spans_that_type_and_no_other():
    from mep_cmap.pipeline import time_axis_for
    cfg = _cfg(window_map={"short": (20.0, 50.0)})
    short = time_axis_for(cfg, "short", 1000.0)
    default = time_axis_for(cfg, "other", 1000.0)
    assert short[0] == pytest.approx(-20.0)
    assert len(short) == 70
    assert default[0] == pytest.approx(-100.0)
    assert len(default) == 500


# ── extraction ───────────────────────────────────────────────────────────────

def _synthetic(fs=1000.0, n_s=60):
    rng = np.random.default_rng(0)
    emg = rng.standard_normal(int(fs * n_s)) * 0.01
    time = np.arange(emg.size) / fs
    stim = {"short": [5.0, 10.0, 15.0], "long": [25.0, 30.0, 35.0]}
    return time, emg, stim


def test_each_type_is_cut_to_its_own_length():
    from mep_cmap.pipeline import pipeline_extract_segments
    time, emg, stim = _synthetic()
    cfg = _cfg(window_map={"short": (20.0, 50.0), "long": (100.0, 500.0)})
    segs = pipeline_extract_segments(time, emg, stim, list(stim), 1000.0, cfg)
    assert len(segs["short"][0][0]) == 70
    assert len(segs["long"][0][0]) == 600


def test_without_a_map_every_type_is_cut_the_same():
    from mep_cmap.pipeline import pipeline_extract_segments
    time, emg, stim = _synthetic()
    segs = pipeline_extract_segments(time, emg, stim, list(stim), 1000.0, _cfg())
    lengths = {st: len(v[0][0]) for st, v in segs.items()}
    assert set(lengths.values()) == {500}


def test_the_window_is_resolved_inside_the_per_type_loop():
    """Above the loop it could only ever be one window for the file."""
    body = PIPE[PIPE.index("def pipeline_extract_segments"):]
    body = body[:body.index("\ndef ", 10)]
    loop = body.index("for stim_type in stim_types")
    assert "window_samples(cfg, stim_type, fs)" in body[loop:]
    assert "samples_before  = int(cfg.pre_ms" not in body


def test_the_delay_still_applies_per_type():
    """The window is new; the delay is not, and both are per type."""
    from mep_cmap.pipeline import pipeline_extract_segments
    time, emg, stim = _synthetic()
    cfg = _cfg(window_map={"short": (20.0, 50.0)},
               delay_ms_map={"short": -2.0})
    segs = pipeline_extract_segments(time, emg, stim, list(stim), 1000.0, cfg)
    assert len(segs["short"][0][0]) == 70


# ── the figures ──────────────────────────────────────────────────────────────

def test_trace_stats_carries_an_axis_per_entry():
    """One shared axis cannot describe types of differing length."""
    assert "mean_ptp, _axis(stim_type)" in PIPE
    assert "mean_ptp, t_axis" in PIPE


def test_the_plot_generator_takes_no_shared_axis():
    tree = ast.parse(PIPE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and \
                node.name == "pipeline_generate_plots":
            args = [a.arg for a in node.args.args]
            assert "time_axis" not in args
            assert "plot_included" not in args
            return
    raise AssertionError("pipeline_generate_plots not found")


# ── the removed control ──────────────────────────────────────────────────────

def test_plot_included_is_gone_from_the_application():
    """It selected which types appeared on one figure and nothing else.

    Not the analysis, not the per-type figures, not any CSV. A control that
    changes one image was not worth a column on a table where the epoch window
    now needs two.
    """
    for src, name in ((APP, "app.py"), (PIPE, "pipeline.py")):
        code = "\n".join(l for l in src.split("\n")
                         if not l.strip().startswith("#"))
        assert "self.plot_included" not in code, name
        assert "_lab_entry_include" not in code, name


def test_the_table_columns_line_up_with_its_headers():
    """A row that grew a widget without a heading silently mislabels the rest."""
    import re
    body = APP[APP.index("def _build_labels_tab"):]
    body = body[:body.index("\n    def ", 10)]
    used = sorted({int(m) for m in re.findall(r"column=(\d+)", body)})
    headers = APP[APP.index('headers = ["Stim"'):]
    headers = headers[:headers.index("]") + 1]
    assert used == list(range(len(ast.literal_eval(headers.split("=", 1)[1].strip()))))


def test_the_window_columns_sit_beside_the_other_timing_ones():
    headers = APP[APP.index('headers = ["Stim"'):]
    headers = ast.literal_eval(
        headers[:headers.index("]") + 1].split("=", 1)[1].strip())
    for label in ("Gap (ms)", "Delay (ms)", "Pre (ms)", "Post (ms)"):
        assert label in headers
    i = headers.index("Delay (ms)")
    assert headers[i + 1] == "Pre (ms)" and headers[i + 2] == "Post (ms)"


# ── state plumbing ───────────────────────────────────────────────────────────

def test_the_window_is_stored_per_channel():
    """Two channels may be different muscles with different response windows.

    test_per_channel_setup checks this list against what the tab harvests, so
    a map on the table that is missing here fails there.
    """
    keys = APP[APP.index("_chan_settings_keys = ("):]
    keys = keys[:keys.index(")")]
    assert '"window_map"' in keys


def test_the_window_reaches_the_run():
    a = APP.index("def _snapshot_analysis_params")
    b = APP.index("\n    def ", a + 10)
    assert "window_map" in APP[a:b]
    a = APP.index("def _analysis_worker")
    b = APP.index("\n    def ", a + 10)
    assert 'window_map           = _own("window_map", {})' in APP[a:b]


def test_a_blank_box_is_stored_as_none_not_zero():
    a = APP.index("def _harvest_labels_tab")
    b = APP.index("\n    def ", a + 10)
    body = APP[a:b]
    assert "def _opt_ms" in body
    assert "return None" in body
    assert "_pre is not None or _post is not None" in body, \
        "a row with both boxes empty should not enter the map at all"


def test_an_older_session_loads_without_a_window_map():
    """Sessions predating this carry plot_included and no window_map."""
    assert 'sess.get("window_map") or {}' in APP


# ── the inspector ────────────────────────────────────────────────────────────

def test_the_inspector_holds_an_axis_per_type():
    assert "_axes_by_type" in INSP
    body = INSP[INSP.index("def _select_axis"):]
    body = body[:body.index("\n    def ", 10)]
    assert "self.cur_type" in body and "self.t = axis" in body


def test_changing_event_type_reselects_the_axis():
    """self.t converts every index to a latency -- markers, the AUC window,
    the reported values. Left on the first type's axis, every one of those is
    wrong for every other type."""
    src = INSP[INSP.index("self.cur_type, self.cur_idx = self.dd_event.get(), 0"):]
    assert "_select_axis()" in src[:200]


def test_a_single_array_still_works():
    """Callers that pass one axis must behave exactly as before."""
    body = INSP[INSP.index("if isinstance(time_axis, dict):"):]
    body = body[:body.index("self.meta")]
    assert "self._axes_by_type = {}" in body
    assert "self.t = time_axis" in body


def test_the_axes_are_derived_not_passed():
    """No new parameter on the payload chain.

    _show_inspector_cb and _open_inspector_gui are unpacked positionally
    through three hops, so a parameter for the windows would have had to be
    added to all of them. The segment lengths and fs are already there and the
    step is 1000/fs, so each type's axis follows from its own length.
    """
    a = APP.index("def _axes_by_type")
    b = APP.index("\n    def ", a + 10)
    body = APP[a:b]
    assert "len(_segs[0])" in body
    assert "1000.0 / float(fs)" in body


def test_the_derived_axis_matches_the_pipeline_axis():
    """The two are computed independently and must agree."""
    from mep_cmap.pipeline import time_axis_for

    fs = 1000.0
    cfg = _cfg(window_map={"A": (100.0, 400.0)})
    pipeline_axis = time_axis_for(cfg, "A", fs)
    n = len(pipeline_axis)
    derived = np.arange(n) * (1000.0 / fs) - 100.0
    assert np.allclose(pipeline_axis, derived)


# ── every segment loop must resolve the window the same way ──────────────────

def test_the_preview_cuts_each_type_to_its_own_window():
    """It had its own loop and kept cutting every type to one window.

    The preview exists to show what the run will produce; offering trials of a
    length the analysis will not produce is the one failure that makes it
    worse than having none. For a type given a longer window it showed a
    response truncated exactly where the analysis measures it whole.
    """
    prev = (PKG / "preview.py").read_text(encoding="utf-8")
    a = prev.index("def _preview_cut")
    b = prev.index("\n    def ", a + 10)
    body = prev[a:b]
    assert "window_samples(" in body, \
        "the preview must resolve the window, not assume one"
    loop = body.index("for stim_type, idxs in chosen.items():")
    assert body.index("window_samples(") > loop, \
        "resolving above the loop is one window for every type again"
    assert "samples_after  = int(post_ms" not in body


def test_no_segment_loop_hardcodes_the_file_wide_window():
    """Walks every module rather than naming the ones already known.

    Three separate loops cut epochs -- the analysis, the inspector segments and
    the preview -- and each was written at a different time. A fourth would be
    written the same way.
    """
    offenders = []
    for path in sorted(PKG.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        code = "\n".join(l for l in src.split("\n")
                         if not l.strip().startswith("#"))
        for bad in ("int(cfg.post_ms    * fs / 1000)",
                    "int(cfg.post_ms * fs / 1000)"):
            if bad in code:
                offenders.append(f"{path.name}: {bad.strip()}")
    assert not offenders, (
        "these resolve the post window without consulting window_map: "
        + ", ".join(offenders))


def test_the_preview_takes_its_window_from_the_run_snapshot():
    """Not from the Tk variables directly, which the run may not be using."""
    prev = (PKG / "preview.py").read_text(encoding="utf-8")
    assert 'params.get("window_map")' in prev
    assert 'params["pre_ms"]' in prev


# ── the boxes are seeded, and stay linked to the default ─────────────────────

def test_the_window_boxes_are_prefilled_from_tab_1c():
    """An empty box is not self-explanatory.

    Showing the window in force means the row states what will happen rather
    than requiring the analyst to know what blank means.
    """
    a = APP.index("def _build_labels_tab")
    b = APP.index("\n    def ", a + 10)
    body = APP[a:b]
    assert "_default_window_ms()" in body
    assert 'f"{_def_pre:g}"' in body and 'f"{_def_post:g}"' in body


def test_the_default_is_read_defensively():
    """These are IntVars being typed into; .get() on '4' mid-'40' raises."""
    body = APP[APP.index("def _default_window_ms"):]
    body = body[:body.index("\n    def ", 10)]
    assert body.count("except Exception") == 2
    assert "return pre, post" in body


def test_unedited_rows_follow_a_change_to_the_default():
    """Pre-filling would otherwise sever the link silently.

    Change Post on 1c from 400 to 500 and every row would still say 400, and
    the run would use 400 -- a number the analyst had just replaced.
    """
    body = APP[APP.index("def _follow_default_window"):]
    body = body[:body.index("\n    def ", 10)]
    assert "_last_default_pre" in body and "_last_default_post" in body
    assert 'cur == ""' in body, "a blank row should adopt the new default too"


def test_an_edited_row_is_left_alone():
    """A row the analyst changed is a decision, not a stale default."""
    body = APP[APP.index("def _follow_default_window"):]
    body = body[:body.index("\n    def ", 10)]
    assert 'cur == f"{old:g}"' in body, (
        "only rows still showing the previous default may be moved")


def test_the_1c_fields_are_watched():
    assert "_v.trace_add(\"write\", self._follow_default_window)" in APP


def test_following_the_default_moves_only_unedited_rows():
    """The logic itself, independent of Tk."""
    class V:
        def __init__(self, v): self.v = str(v)
        def get(self): return self.v
        def set(self, v): self.v = v

    rows = {"A": V("20"), "B": V(""), "C": V("35")}
    old, new = 20.0, 50.0
    for _stim, v in rows.items():
        cur = v.get().strip()
        if cur == "" or cur == f"{old:g}":
            v.set(f"{new:g}")
    assert rows["A"].get() == "50"
    assert rows["B"].get() == "50"
    assert rows["C"].get() == "35"


# ── the clamp must know about the per-type windows ───────────────────────────

def test_per_type_windows_are_clamped_to_the_epoch():
    """A pre-epoched file contains nothing outside its own epoch.

    A window reaching past it draws its baseline from the previous trial's
    response and reports that as background EMG. The file-wide pair was
    already clamped; adding a per-type column without this made exactly that
    contamination reachable again by a route the clamp did not know about.
    """
    from mep_cmap.pipeline import clamp_config_to_epoch_bounds

    params = dict(pre_ms=100, post_ms=400, prestim_ms=100,
                  window_map={"short": (20.0, 50.0),
                              "greedy": (250.0, 900.0)})
    cfg, changes = clamp_config_to_epoch_bounds(params, (150.0, 500.0))
    assert cfg["window_map"]["short"] == (20.0, 50.0), "a fitting window is untouched"
    assert cfg["window_map"]["greedy"] == (150.0, 500.0)
    named = {c[0] for c in changes}
    assert "window_map[greedy].pre" in named
    assert "window_map[greedy].post" in named


def test_a_blank_side_stays_blank_through_the_clamp():
    """None means 'use the fallback', which the clamp has already handled."""
    from mep_cmap.pipeline import clamp_config_to_epoch_bounds
    cfg, _ = clamp_config_to_epoch_bounds(
        dict(pre_ms=100, post_ms=400, prestim_ms=100,
             window_map={"partial": (None, 800.0)}),
        (150.0, 500.0))
    assert cfg["window_map"]["partial"] == (None, 500.0)


def test_clamping_is_reported_not_silent():
    from mep_cmap.pipeline import clamp_config_to_epoch_bounds
    _cfgd, changes = clamp_config_to_epoch_bounds(
        dict(pre_ms=100, post_ms=400, prestim_ms=100,
             window_map={"greedy": (10.0, 900.0)}),
        (150.0, 500.0))
    assert any("window_map" in str(c[0]) for c in changes), (
        "a shortened window must appear in the change list the analyst is shown")


# ── one place sets the epoch ─────────────────────────────────────────────────

def test_the_epoch_fields_are_gone_from_tab_1c():
    """Neither was visible-only: both decided what was extracted.

    Three fields on that tab looked like pre-stimulus settings and meant three
    different things.
    """
    assert 'text="Pre-stim visible (ms):"' not in APP
    assert 'text="Post-stim visible (ms):"' not in APP


def test_the_baseline_setting_stays_and_says_where_the_epoch_went():
    """The grey line beside it became the field's own ⓘ.

    Three settings across two tabs look like pre-stimulus settings, so this
    one has to say which it is and where the others live.
    """
    assert '"Pre-stim for analysis (ms):"' in APP
    tree = ast.parse(APP)
    help_d = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and \
                any(getattr(t, "id", "") == "FIELD_HELP" for t in node.targets):
            help_d = ast.literal_eval(node.value)
    assert help_d is not None, "FIELD_HELP not found"
    text = help_d["prestim"]
    assert "tab 1a" in text
    assert "set per type" in text or "per type" in text


def test_the_seed_comes_from_preferences():
    from mep_cmap.preferences import prefs
    pre, post = prefs.default_epoch_ms
    assert pre > 0 and post > 0
    assert "prefs.default_epoch_ms" in APP


def test_the_inspector_view_follows_the_per_type_window():
    """The segments still carry prestim_ms of lead-in for the detectors.

    What changes is where the view opens, so judging an M-wave does not mean
    staring at 80 ms of flat baseline.
    """
    a = APP.index("def _visible_pre_by_type")
    b = APP.index("\n    def ", a + 10)
    body = APP[a:b]
    assert "self.window_map" in body
    assert "float(self.pre_time.get())" in body, "a plain float when unconfigured"
    assert "_visible_pre   = self._visible_pre_by_type()" in APP


def test_the_inspector_accepts_either_shape():
    body = INSP[INSP.index("self._visible_pre_map"):]
    body = body[:body.index("\n        self._analysis_pre") if
                "\n        self._analysis_pre" in body else 400]
    assert "isinstance(visible_pre_ms, dict)" in body
    src = INSP[INSP.index("def _select_axis"):]
    src = src[:src.index("\n    def ", 10)]
    assert "self.visible_pre_ms = float(_vp)" in src, \
        "the x-limit must move with the displayed type"


# ── the snapshot must describe the screen ────────────────────────────────────

def test_the_snapshot_reads_the_setup_table_first():
    """Editing a row and pressing Run used the previously confirmed values.

    _harvest_labels_tab ran only on Confirm Setup, a channel switch, or
    copy-to-all, and the confirmation was invalidated only when the table was
    REBUILT -- not when a field was edited. So a row changed after the last
    confirm was displayed but not analysed. It showed up first on the epoch
    window, where the difference is a visibly wrong plot, but every map on
    that table behaved the same way.
    """
    body = APP[APP.index("def _snapshot_analysis_params"):]
    body = body[:body.index("\n    def ", 10)]
    assert "_harvest_labels_tab()" in body
    assert body.index("_harvest_labels_tab()") < body.index("params = dict("), \
        "the table must be read before the snapshot is taken"


def test_the_harvest_is_guarded_and_non_fatal():
    """No table exists before a file is opened, and a bad field is not a
    reason to refuse to run."""
    body = APP[APP.index("def _snapshot_analysis_params"):]
    body = body[:body.index("\n    def ", 10)]
    assert '_labels_tab_built' in body
    assert "except Exception" in body


def test_both_run_and_preview_go_through_it():
    """One seam, so neither can drift from the other."""
    run = APP[APP.index("def run_analysis_start"):]
    run = run[:run.index("\n    def ", 10)]
    assert "_snapshot_analysis_params()" in run
    prev = (PKG / "preview.py").read_text(encoding="utf-8")
    assert "_snapshot_analysis_params()" in prev


# ── the map is held twice; both copies must be clamped ───────────────────────

def test_the_per_channel_window_map_is_clamped_too():
    """The analysis reads the per-channel copy in preference to the file-wide
    one, so clamping only the latter left the run epoching past the end of the
    recording while the preview stopped at it.

    On a stitched pre-epoched file the extra samples are mirror-padded guard
    band. They draw as a flat line, which is indistinguishable from a quiet
    trace -- so the disagreement looked like a rendering difference rather
    than a measurement reaching into fabricated data.
    """
    body = APP[APP.index("def _snapshot_analysis_params"):]
    body = body[:body.index("\n    def ", 10)]
    assert "clamp_window_map" in body
    assert 'params.get("chan_settings")' in body
    i, j = body.index("_clamp(params, _bounds)"), body.index("clamp_window_map")
    assert i < j, "the file-wide clamp comes first, then the per-channel copies"


def test_the_helper_clamps_both_sides_and_reports():
    from mep_cmap.pipeline import clamp_window_map
    out, changes = clamp_window_map(
        {"ok": (10.0, 50.0), "greedy": (200.0, 900.0), "half": (None, 700.0)},
        25.0, 100.0)
    assert out["ok"] == (10.0, 50.0)
    assert out["greedy"] == (25.0, 100.0)
    assert out["half"] == (None, 100.0)
    named = {c[0] for c in changes}
    assert "window_map[greedy].pre" in named
    assert "window_map[half].post" in named
    assert not any("ok" in c[0] for c in changes)


def test_a_malformed_entry_is_passed_through_not_dropped():
    """Losing a stimulus type's window silently would be worse than keeping
    one that cannot be clamped."""
    from mep_cmap.pipeline import clamp_window_map
    out, _ = clamp_window_map({"weird": "nonsense"}, 25.0, 100.0)
    assert out["weird"] == "nonsense"


def test_the_channel_is_named_in_the_change_list():
    """Otherwise two channels reporting the same reduction read as a repeat."""
    body = APP[APP.index("def _snapshot_analysis_params"):]
    body = body[:body.index("\n    def ", 10)]
    assert 'f"channel {_ch} {_f}"' in body


# ── traces bound in the constructor fire before the GUI exists ───────────────

def test_the_entry_dicts_exist_from_construction():
    """A trace bound in __init__ reads them.

    Writing to pre_time or post_time before any file was opened -- restoring a
    session, or any other early set -- reached a table that did not exist yet.
    """
    import ast

    tree = ast.parse(APP)
    init = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TMSAnalysisApp":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    init = ast.unparse(fn)
    assert init is not None, "TMSAnalysisApp.__init__ not found"
    assert "self._lab_entry_pre = {}" in init
    assert "self._lab_entry_post = {}" in init


def test_the_follow_handler_survives_an_empty_table():
    body = APP[APP.index("def _follow_default_window"):]
    body = body[:body.index("\n    def ", 10)]
    assert "getattr(self, \"_lab_entry_pre\", None)" in body
    i = body.index("return")
    j = body.index("_default_window_ms()")
    assert i < j, "the guard must come before anything reads the table"


def test_every_constructor_trace_targets_something_that_exists():
    """The general form of the fault.

    test_no_missing_attributes asks whether an attribute is assigned SOMEWHERE
    in the class, which _lab_entry_pre was -- in _build_labels_tab. It cannot
    know that a trace bound in __init__ runs first. Any handler wired up there
    must therefore either guard, or read only what __init__ has already set.
    """
    import ast

    tree = ast.parse(APP)
    init = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TMSAnalysisApp":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    init = fn
    assert init is not None

    handlers = set()
    for n in ast.walk(init):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "trace_add":
            for arg in n.args[1:]:
                src = ast.unparse(arg)
                if src.startswith("self."):
                    handlers.add(src.split(".", 1)[1].split("(")[0])

    for name in handlers:
        body = APP[APP.index(f"def {name}(self"):]
        body = body[:body.index("\n    def ", 10)]
        assert "getattr(self," in body or "hasattr(self," in body, (
            f"{name} is bound as a trace in __init__ and reads state without "
            f"checking it exists yet")
