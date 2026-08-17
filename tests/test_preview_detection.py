"""
Preview detection must show what the run will produce.

A preview that disagrees with the analysis is worse than no preview: it is a
confident picture of settings that are not the ones about to be applied, and
nothing about the output looks wrong. Every test here guards one way the two
could drift apart.
"""

import ast
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = (ROOT / "mep_cmap" / "app.py").read_text(encoding="utf-8")
PREVIEW_SRC = (ROOT / "mep_cmap" / "preview.py").read_text(encoding="utf-8")
PIPELINE_SRC = (ROOT / "mep_cmap" / "pipeline.py").read_text(encoding="utf-8")


def _cfg_fields_read_by(func_name, source):
    """Every `cfg.<field>` the named function reads."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            out = set()
            for n in ast.walk(node):
                if isinstance(n, ast.Attribute) and \
                        isinstance(n.value, ast.Name) and n.value.id == "cfg":
                    out.add(n.attr)
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "getattr" \
                        and len(n.args) > 1 and isinstance(n.args[0], ast.Name) \
                        and n.args[0].id == "cfg":
                    try:
                        out.add(ast.literal_eval(n.args[1]))
                    except Exception:
                        out.add("<dynamic>")
            return out
    raise AssertionError(f"{func_name} not found")


def test_preview_supplies_exactly_the_filter_fields():
    """The preview's config must cover what the filter stage reads -- no more.

    The obvious shortcut, building the config by filtering the params snapshot
    against PipelineConfig field names, is wrong: the snapshot says `min_amp`
    and `enable_out_review` where the fields are `min_peak_amplitude` and
    `enable_outlier_review`, so a name filter silently substitutes defaults.
    Naming the fields avoids that but can go stale, so this test holds the list
    to the filter stage. Adding a filter parameter fails here rather than
    quietly making the preview filter differently from the run.
    """
    from mep_cmap.preview import FILTER_CFG_FIELDS

    read = _cfg_fields_read_by("pipeline_apply_filters", PIPELINE_SRC)
    assert "<dynamic>" not in read, \
        "pipeline_apply_filters reads cfg dynamically; the preview cannot " \
        "guarantee parity by naming fields"
    assert set(FILTER_CFG_FIELDS) == read, (
        f"missing from preview: {sorted(read - set(FILTER_CFG_FIELDS))}; "
        f"surplus: {sorted(set(FILTER_CFG_FIELDS) - read)}")


def test_every_filter_field_exists_on_pipeline_config():
    from dataclasses import fields as _fields

    from mep_cmap.pipeline import PipelineConfig
    from mep_cmap.preview import FILTER_CFG_FIELDS

    names = {f.name for f in _fields(PipelineConfig)}
    assert set(FILTER_CFG_FIELDS) <= names


def test_filter_fields_all_present_in_the_params_snapshot():
    """Every field the preview needs must exist in the run's own snapshot.

    If one is missing the preview would raise KeyError -- loud, but only when
    someone previews. Catch it here instead.
    """
    from mep_cmap.preview import FILTER_CFG_FIELDS

    tree = ast.parse(APP)
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and \
                node.name == "_snapshot_analysis_params":
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "dict":
                    ks = {k.arg for k in n.keywords if k.arg}
                    if len(ks) > len(keys):
                        keys = ks
    assert keys, "_snapshot_analysis_params does not build a params dict"
    missing = set(FILTER_CFG_FIELDS) - keys
    assert not missing, f"snapshot lacks: {sorted(missing)}"
    for extra in ("prestim_ms", "post_ms", "delay_ms_map", "input_path",
                  "channel_idx", "marker_choice"):
        assert extra in keys, f"snapshot lacks {extra}, which the preview reads"


# ── trial selection ──────────────────────────────────────────────────────────

def test_selection_is_evenly_spaced_and_includes_the_endpoints():
    from mep_cmap.preview import select_preview_trials

    picked = select_preview_trials(100, 5)
    assert picked[0] == 0 and picked[-1] == 99, \
        "first and last trial must be judged; drift shows up there first"
    gaps = np.diff(picked)
    assert gaps.max() - gaps.min() <= 1


@pytest.mark.parametrize("n,k", [(0, 8), (8, 0), (3, 8), (1, 1), (50, 1)])
def test_selection_degenerate_cases(n, k):
    from mep_cmap.preview import select_preview_trials

    picked = select_preview_trials(n, k)
    assert len(picked) == len(set(picked))
    assert all(0 <= i < n for i in picked)
    assert picked == sorted(picked)
    if n and k:
        assert picked


def test_selection_never_exceeds_what_was_asked_for():
    from mep_cmap.preview import select_preview_trials

    for n in (5, 17, 40, 500):
        for k in (1, 2, 8, 24):
            assert len(select_preview_trials(n, k)) <= max(k, 0) or n < k


# ── the preview must not write ───────────────────────────────────────────────

def _method_source(name):
    tree = ast.parse(APP)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(APP, node)
    raise AssertionError(f"{name} not found")


def test_the_preview_opener_writes_nothing_back():
    """The preview call site must not persist the inspector's metadata.

    The review path writes inspector.meta into segments_metadata, the
    per-channel map and _last_outlier_result, then autosaves the session.
    Reaching any of that from a preview would seed manual marker edits into an
    analysis the analyst never ran.
    """
    src = _method_source("_open_inspector_preview")
    for forbidden in ("segments_metadata", "_chan_segment_meta",
                      "_last_outlier_result", "_autosave_session"):
        assert forbidden not in src, \
            f"_open_inspector_preview must not touch {forbidden}"


def test_the_preview_opener_starts_from_empty_metadata():
    """Empty metadata is what makes every marker freshly detected.

    Seeded metadata would show saved edits from an earlier session as though
    the current settings had produced them.
    """
    tree = ast.parse(_method_source("_open_inspector_preview"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and \
                getattr(node.func, "id", "") == "DataInspectorWindow":
            kw = {k.arg: k.value for k in node.keywords}
            assert "metadata_dict" in kw
            assert isinstance(kw["metadata_dict"], ast.Dict) and \
                not kw["metadata_dict"].keys, "preview must seed no metadata"
            return
    raise AssertionError("_open_inspector_preview never opens the inspector")


def test_the_two_inspector_call_sites_pass_the_same_settings():
    """Review and preview must configure the inspector identically.

    They are separate call sites on purpose -- _open_inspector_gui sits on the
    pipeline's positionally-unpacked payload, so a preview flag there would
    have to be threaded through two more hops. The cost of that choice is
    duplication, and this is what stops the duplication drifting: a detection
    setting added to review but forgotten in preview would make the preview
    quietly show different markers from the run.
    """
    tree = ast.parse(APP)
    sites = {}
    for fn in ("_open_inspector_gui", "_open_inspector_preview"):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn:
                for n in ast.walk(node):
                    if isinstance(n, ast.Call) and \
                            getattr(n.func, "id", "") == "DataInspectorWindow":
                        sites[fn] = {k.arg for k in n.keywords if k.arg}
    assert len(sites) == 2, f"expected two call sites, found {sorted(sites)}"
    review, preview = sites["_open_inspector_gui"], sites["_open_inspector_preview"]
    assert review == preview, (
        f"only in review: {sorted(review - preview)}; "
        f"only in preview: {sorted(preview - review)}")


def test_preview_module_writes_nothing():
    """No file, figure or session writing anywhere in the preview path."""
    for forbidden in ("to_csv", "savefig", "print_figure", "np.savez",
                      "_autosave_session", "makedirs", "open("):
        assert forbidden not in PREVIEW_SRC, \
            f"preview.py must not write; found {forbidden!r}"


def test_preview_module_uses_the_read_only_opener():
    """preview.py must not reach the write-back path by mistake."""
    assert "_open_inspector_gui" not in PREVIEW_SRC
    tree = ast.parse(PREVIEW_SRC)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and \
                getattr(node.func, "attr", "") == "_open_inspector_preview":
            found = True
    assert found, "preview.py never opens the inspector"


# ── parity with the pipeline's own segmentation ──────────────────────────────

def test_preview_segments_match_the_pipeline_for_the_same_trial():
    """Byte-identical windows, including the event delay.

    The pipeline re-extracts its inspector segments from the filtered trace
    with prestim_ms/post_ms and the per-type delay. The preview does the same;
    if the two ever differ, the preview shows a different epoch from the one
    the analysis measures -- which is how markers came to sit ~1.6-2.0 ms early
    in condition C.
    """
    from mep_cmap.preview import select_preview_trials

    rng = np.random.default_rng(0)
    fs = 4000
    emg = rng.standard_normal(fs * 30).astype(float)
    time = np.arange(emg.size) / fs
    stim = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0]
    prestim_ms, post_ms, delay_ms = 100.0, 300.0, -1.8

    sb = int(prestim_ms * fs / 1000)
    sa = int(post_ms * fs / 1000)
    shift = int(round(delay_ms * fs / 1000.0))

    def pipeline_style(t0):
        ix = int(np.argmin(np.abs(time - t0))) + shift
        seg = emg[max(0, ix - sb): ix + sa]
        return seg if len(seg) == sb + sa else None

    for i in select_preview_trials(len(stim), 3):
        expected = pipeline_style(stim[i])
        ix = int(np.argmin(np.abs(time - stim[i]))) + shift
        got = emg[max(0, ix - sb): ix + sa]
        assert expected is not None
        assert np.array_equal(expected, got)


def test_preview_respects_the_epoch_clamp_by_reusing_the_snapshot():
    """Clamping happens in the snapshot, so the preview inherits it.

    An unclamped window on a pre-epoched file draws its baseline from the
    previous trial's response -- silent contamination. The preview must not
    have its own window logic that could skip the clamp.
    """
    snap = _method_source("_snapshot_analysis_params")
    assert "clamp_config_to_epoch_bounds" in snap
    tree = ast.parse(PREVIEW_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and \
                getattr(node.func, "attr", "") == "_snapshot_analysis_params":
            break
    else:
        raise AssertionError("preview must take its settings from the snapshot")
    assert "get_epoch_bounds" not in PREVIEW_SRC, \
        "preview must inherit the clamp, not re-derive bounds itself"


def test_the_preference_is_a_default_not_a_cap():
    """Selecting every trial must be possible.

    Trial count barely affects cost: the read and filter dominate and happen
    regardless, and the inspector draws one trial at a time. A cap here would
    stop an analyst looking at a run they have reason to doubt.
    """
    from mep_cmap import preview

    assert not hasattr(preview, "MAX_PREVIEW_TRIALS"), \
        "the cap was removed deliberately"
    assert preview.default_tick_count() >= 1


def test_select_all_offers_every_trial():
    from mep_cmap.preview import select_preview_trials

    assert select_preview_trials(300, 300) == list(range(300))


def test_the_dialog_preselects_the_even_spread():
    """The default shown must be the spread, not the first n."""
    import ast as _ast

    tree = _ast.parse(PREVIEW_SRC)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef) and node.name == "_preview_choose":
            src = _ast.unparse(node)
            assert "select_preview_trials" in src
            assert "TrialSelectDialog" in src
            return
    raise AssertionError("_preview_choose not found")


def test_cancelling_the_dialog_opens_nothing():
    import ast as _ast

    tree = _ast.parse(PREVIEW_SRC)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef) and node.name == "_preview_choose":
            src = _ast.unparse(node)
            i = src.index("dlg.result")
            head = src[:src.index("_preview_cut")]
            assert "if not dlg.result" in head and "return" in head[i:], \
                "a cancelled dialog must return before any trial is cut"
            return
    raise AssertionError("_preview_choose not found")


def test_the_chooser_keeps_each_stim_types_selection():
    """exportselection=False, or picking in one list clears the others.

    Tk hands the X selection to the most recent listbox, which silently
    deselects every other one -- so a two-condition file would preview only
    whichever list was touched last.
    """
    assert "exportselection=False" in PREVIEW_SRC


def test_the_chooser_reports_indices_not_labels():
    """Trials are shown 1-based and returned 0-based."""
    import ast as _ast

    tree = _ast.parse(PREVIEW_SRC)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef) and node.name == "TrialSelectDialog":
            for fn in node.body:
                if isinstance(fn, _ast.FunctionDef) and fn.name == "selection":
                    assert "curselection" in _ast.unparse(fn)
                    assert "int(i)" in _ast.unparse(fn)
                    return
    raise AssertionError("TrialSelectDialog.selection not found")


# ── read-only mode ───────────────────────────────────────────────────────────

INSPECTOR_SRC = (ROOT / "mep_cmap" / "inspector.py").read_text(encoding="utf-8")


def test_preview_opens_the_inspector_read_only():
    src = _method_source("_open_inspector_preview")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and \
                getattr(node.func, "id", "") == "DataInspectorWindow":
            kw = {k.arg: k.value for k in node.keywords}
            assert "read_only" in kw and kw["read_only"].value is True
            return
    raise AssertionError("preview does not construct the inspector")


def test_review_stays_editable():
    """The flag is additive; review behaviour must be unchanged."""
    src = _method_source("_open_inspector_gui")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and \
                getattr(node.func, "id", "") == "DataInspectorWindow":
            kw = {k.arg: k.value for k in node.keywords}
            assert kw["read_only"].value is False
            return
    raise AssertionError("review does not construct the inspector")


def test_read_only_defaults_to_false_everywhere():
    """A caller that says nothing gets the editable window it always got."""
    tree = ast.parse(INSPECTOR_SRC)
    checked = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            names = [a.arg for a in node.args.args]
            if "read_only" not in names:
                continue
            i = names.index("read_only") - (len(names) - len(node.args.defaults))
            assert node.args.defaults[i].value is False
            checked += 1
    assert checked == 3, (
        f"expected read_only on DataInspectorWindow, DraggablePoint and "
        f"DraggableLine; found {checked}")


def test_both_marker_classes_ignore_the_mouse_when_read_only():
    """Drawn but fixed. The guard must be the first thing _on_press does.

    Placing it after the toolbar check would still work today, but the point of
    a preview is that a marker cannot be moved at all -- not that most routes
    to moving it are closed.
    """
    tree = ast.parse(INSPECTOR_SRC)
    seen = set()
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        if cls.name not in ("DraggablePoint", "DraggableLine"):
            continue
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "_on_press":
                first = fn.body[0]
                assert isinstance(first, ast.If), \
                    f"{cls.name}._on_press does not open with a guard"
                assert "read_only" in ast.unparse(first.test)
                assert isinstance(first.body[0], ast.Return)
                seen.add(cls.name)
    assert seen == {"DraggablePoint", "DraggableLine"}, seen


def test_the_markers_are_still_drawn_in_read_only_mode():
    """Fixed, not absent -- the markers are what the preview is for."""
    tree = ast.parse(INSPECTOR_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and \
                getattr(node.func, "id", "") in ("DraggablePoint", "DraggableLine"):
            kw = {k.arg for k in node.keywords if k.arg}
            assert "read_only" in kw, \
                "a marker is created without being told the mode"
    assert "if self.read_only" not in INSPECTOR_SRC.split("def _plot")[1][:6000], \
        "plotting must not branch on read_only; the markers are drawn either way"


def test_read_only_never_reaches_detection():
    """The flag is a widget concern. Detection must not see it.

    If a detector ever branched on read_only, the preview would stop being a
    preview of the analysis and become its own algorithm.
    """
    tree = ast.parse(INSPECTOR_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
                "_condition_template", "_update_meta", "_set_exclude"):
            assert "read_only" not in ast.unparse(node), \
                f"{node.name} branches on read_only"
    for detector in ("dispatch_onset", "detect_csp_bootstrap", "detect_mep_offset"):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and \
                    getattr(node.func, "id", "") == detector:
                kw = {k.arg for k in node.keywords if k.arg}
                assert "read_only" not in kw


def test_the_close_button_does_not_promise_to_save():
    tree = ast.parse(INSPECTOR_SRC)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_close_preview" in names
    src = [ast.unparse(n) for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name == "_close_preview"][0]
    for forbidden in ("note_box.get", "_update_meta", "self.meta["):
        assert forbidden not in src, \
            f"_close_preview must commit nothing; found {forbidden!r}"


def test_even_spread_does_not_reapply_the_remembered_selection():
    """The button must recompute, not replay what the dialog opened with.

    It was bound to the `preselect` argument, which _preview_choose sets to the
    previous preview's selection when the file has not changed. After a Select
    all, pressing Even spread put every trial back -- the button looked broken
    while doing exactly what it was told.
    """
    import ast as _ast

    tree = _ast.parse(PREVIEW_SRC)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef) and node.name == "TrialSelectDialog":
            init = [f for f in node.body
                    if isinstance(f, _ast.FunctionDef) and f.name == "__init__"][0]
            src = _ast.unparse(init)
            assert "command=self._select_spread" in src, \
                "Even spread must call the recomputing handler"
            assert "self._apply(preselect)" in src, \
                "the dialog should still OPEN on the preselection"
            i = src.index("Even spread")
            j = src.index("pack", i)
            assert "preselect" not in src[i:j], \
                "the button is bound to preselect again"

            spread = [f for f in node.body
                      if isinstance(f, _ast.FunctionDef)
                      and f.name == "_select_spread"][0]
            body = _ast.unparse(spread)
            assert "select_preview_trials" in body and "self._counts" in body
            assert "preselect" not in body
            return
    raise AssertionError("TrialSelectDialog not found")


def test_spread_equals_all_on_short_files_and_is_explained():
    """k over fewer than k trials is every trial -- correct, but confusing."""
    from mep_cmap.preview import select_preview_trials

    assert select_preview_trials(5, 8) == [0, 1, 2, 3, 4]
    assert "fewer trials than the spread size" in PREVIEW_SRC, \
        "the count line should say why the spread selected everything"


def test_the_count_line_reports_the_total_available():
    """'8 selected' alone cannot be judged; '8 of 120' can."""
    import ast as _ast

    tree = _ast.parse(PREVIEW_SRC)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef) and node.name == "_refresh_count":
            src = _ast.unparse(node)
            assert "available" in src and "self._counts" in src
            return
    raise AssertionError("_refresh_count not found")


# ── the preview must load the way the run loads ──────────────────────────────

def test_every_pipeline_load_file_call_passes_the_event_sources():
    """A caller that omits them silently reads the file's own markers.

    The preview did exactly that: a configured threshold produced 163 events
    for the run and the chooser offered the 100 embedded markers, so the two
    disagreed about which trials existed. Compiling and every other test
    passed, because they checked the pipeline and the worker rather than each
    caller.

    This walks every call instead of naming the ones already known.
    """
    import ast
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
    offenders = []
    for path in sorted(pkg.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and \
                    getattr(node.func, "id", "") == "pipeline_load_file":
                kw = {k.arg for k in node.keywords if k.arg}
                if "sources" not in kw:
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "these calls would read the file's markers instead of the configured "
        "sources: " + ", ".join(offenders))


def test_the_preview_builds_sources_the_way_the_worker_does():
    """Same key, same reconstruction — so the two cannot pick different ones."""
    import ast
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
    app = (pkg / "app.py").read_text(encoding="utf-8")
    prev = (pkg / "preview.py").read_text(encoding="utf-8")

    for src in (app, prev):
        assert '(params.get("event_sources") or {}).get(' in src
    a = app.index("def _analysis_worker")
    b = app.index("\n    def ", a + 10)
    assert ".get(_ch)" in app[a:b], "the worker keys on the channel it is running"
    assert '.get(\n            params["channel_idx"])' in prev or \
           '.get(params["channel_idx"])' in prev, \
        "the preview keys on the channel it is previewing"


def test_merge_warnings_reach_the_preview_log():
    import pathlib

    prev = (pathlib.Path(__file__).resolve().parent.parent
            / "mep_cmap" / "preview.py").read_text(encoding="utf-8")
    assert "warn=lambda m: self.log(" in prev, \
        "a source misconfiguration must be as visible in the preview as in a run"
