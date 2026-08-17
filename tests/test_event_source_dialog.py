"""
The event-source dialogue.

A threshold level is not checkable by reading it: two volts is right or wrong
depending on the trigger's amplitude, its baseline, and whether the pulse rings,
none of which is visible from the box it was typed into. So the dialogue's real
content is the preview -- the trace with the level drawn across it and every
detected crossing marked -- and a count that changes as the level does.

Tk cannot run here, so the dialogue's construction is checked by reading it and
its decimation is exercised directly.
"""

import pathlib

import numpy as np
import pytest

from mep_cmap.event_sources import decimate_for_preview

DLG = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "event_source_dialog.py").read_text(encoding="utf-8")
PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
APP = (PKG / "app.py").read_text(encoding="utf-8")
DLG_NOW = (PKG / "event_source_dialog.py").read_text(encoding="utf-8")


def _code_of(name, src=None):
    """A function's body as code, with its docstring removed.

    Grepping a method's raw text matches prose as readily as statements, and
    these docstrings name the very things being asserted absent -- three tests
    in this file have already passed or failed on a word in a comment rather
    than on what the code does.
    """
    import ast

    tree = ast.parse(src if src is not None else DLG_NOW)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                body = body[1:]
            return "\n".join(ast.unparse(n) for n in body)
    raise AssertionError(f"{name} not found")


def _params_of(name, src=None):
    import ast

    tree = ast.parse(src if src is not None else DLG_NOW)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [a.arg for a in node.args.args]
    raise AssertionError(f"{name} not found")

FS = 5000.0


# ── The preview must show what the detector sees ─────────────────────────────

def _recording_with_spikes(n_spikes=200, dur_s=2000.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.01, int(dur_s * FS))
    pos = sorted(rng.integers(0, x.size - 10, n_spikes))
    for i in pos:
        x[i:i + 3] = 5.0
    return x


def test_decimation_preserves_the_transients_the_detector_finds():
    """
    A stimulus trigger is a one-sample spike. Plain subsampling drew a flat
    line on a two-thousand-second recording while the detector found two
    hundred events -- a preview that would have led the analyst to set a level
    against a trace showing none of the pulses.
    """
    x = _recording_with_spikes()
    t, lo, hi = decimate_for_preview(x, FS, max_points=4000)

    kept = int((hi > 2.5).sum())
    assert kept >= 190, f"only {kept} of 200 pulses survived decimation"

    step = x.size // 4000
    naive = int((x[::step] > 2.5).sum())
    assert naive < 10, (
        "the fixture no longer demonstrates the problem: plain subsampling "
        f"kept {naive} pulses"
    )


def test_decimation_returns_a_band_not_a_line():
    x = _recording_with_spikes()
    t, lo, hi = decimate_for_preview(x, FS, max_points=1000)
    assert len(t) == len(lo) == len(hi)
    assert np.any(hi > lo), "min and max are identical; nothing was summarised"


def test_a_short_recording_is_not_decimated():
    x = np.arange(100, dtype=float)
    t, lo, hi = decimate_for_preview(x, FS, max_points=4000)
    assert np.allclose(lo, x) and np.allclose(hi, x)


def test_the_tail_is_not_dropped():
    """A recording whose length is not a multiple of the step still ends where
    it ends; losing the last block would hide events at the end of a session."""
    x = np.zeros(10_007)
    x[-3:] = 5.0
    t, lo, hi = decimate_for_preview(x, FS, max_points=100)
    assert hi.max() == 5.0, "the final partial block was discarded"


def test_an_empty_signal_does_not_raise():
    t, lo, hi = decimate_for_preview([], FS)
    assert len(t) == 0


# ── The dialogue ─────────────────────────────────────────────────────────────

def test_the_preview_draws_the_level_and_the_crossings():
    assert "axhline(src.level" in DLG
    assert "detect_threshold_crossings(" in DLG
    assert "fill_between" in DLG, "the trace must be a min/max band"


def test_the_count_updates_with_the_level():
    a = DLG.index("def _update_preview")
    b = DLG.index("\n    def ", a + 10)
    body = DLG[a:b]
    assert "count_var.set(" in body
    assert "event(s) detected" in body


def test_editing_a_field_refreshes_the_preview():
    assert 'trace_add("write"' in DLG
    a = DLG.index("def _on_edit")
    b = DLG.index("\n    def ", a + 10)
    assert "_update_preview()" in DLG[a:b]


def test_a_half_typed_number_is_not_rejected():
    """
    The analyst is mid-keystroke. Rejecting an unparseable value would fight
    the typing; the preview simply does not update until it parses.
    """
    a = DLG.index("def _apply_edits")
    b = DLG.index("\n    def ", a + 10)
    body = DLG[a:b]
    assert "except (TypeError, ValueError)" in body
    assert "pass" in body


def test_the_channel_is_read_once_not_per_keystroke():
    """Re-reading a long recording on every character would make it unusable."""
    a = DLG.index("def _channel_data")
    b = DLG.index("\n    def ", a + 10)
    assert "self._cache" in DLG[a:b]


def test_an_unreadable_channel_is_reported_in_the_preview():
    a = DLG.index("def _update_preview")
    b = DLG.index("\n    def ", a + 10)
    assert "Could not read" in DLG[a:b]


def test_the_interval_kind_says_nothing_is_detected():
    """The times are asserted; no part of the recording can confirm them."""
    assert "Nothing is detected" in DLG


# ── Both entry points ────────────────────────────────────────────────────────

def test_the_dialogue_is_reachable_from_tab_1a_and_channel_assignment():
    """
    One dialogue, two entry points: the first setup of a file is when the
    question arises, and it must be revisable afterwards without reopening the
    file.
    """
    assert APP.count("self._open_event_sources") >= 3, (
        "expected a definition plus a button in tab 1a and in each channel "
        "dialogue"
    )
    # Anchor on the BUTTON, not the phrase: the same words appear in the
    # run-gate warning, and searching from the start of the file finds that.
    a = APP.index('command=self._copy_setup_to_all_channels')
    assert "Event sources" in APP[a:a + 400], "no button on tab 1a"


def test_choosing_sources_rebuilds_the_stimulus_table():
    a = APP.index("def _apply_event_sources")
    b = APP.index("\n    def ", a + 10)
    body = APP[a:b]
    assert "extract_events(" in body
    assert "_build_labels_tab(" in body
    assert "warnings" in body, "merge warnings must reach the analyst"


def test_sources_are_saved_with_the_session_and_cleared_with_the_file():
    """Sources are per channel now, so the session stores a mapping.

    JSON has no integer keys, hence the str() on the way out and the int() on
    the way back.
    """
    assert '"event_sources": {str(_c): [_s.to_dict() for _s in _lst]' in APP
    assert 'sess.get("event_sources")' in APP
    a = APP.index("def _reset_state_for_new_file")
    b = APP.index("\n    def ", a + 10)
    assert "self.event_sources = {}" in APP[a:b], (
        "sources describe one recording's channels; they cannot carry over"
    )


def test_an_old_flat_session_still_loads():
    """Sessions written before sources were per channel are a plain list.

    Applying that list to every selected channel reproduces what the session
    actually did, which is what restoring one should mean.
    """
    a = APP.index('sess.get("event_sources")')
    seg = APP[a - 400:a + 900]
    assert "isinstance(_raw, list)" in seg, "the flat shape must still load"
    assert "_analysis_channel_indices()" in seg


def test_sources_reach_the_analysis():
    """The gap this phase closes.

    extract_events was called only to rebuild tab 1a, so a configured
    threshold changed what the interface displayed and nothing else: the run
    went back to reading the file's own markers by name, and the analysis
    measured events the analyst was never shown.
    """
    a = APP.index("def _snapshot_analysis_params")
    b = APP.index("\n    def ", a + 10)
    assert "event_sources" in APP[a:b], (
        "the snapshot must carry the sources or the run cannot see them")

    a = APP.index("def _analysis_worker")
    b = APP.index("\n    def ", a + 10)
    body = APP[a:b]
    assert "event_sources        = _own_sources" in body
    assert '(params.get("event_sources") or {}).get(_ch)' in body, (
        "each channel must get its own sources, not the first channel's")


def test_the_pipeline_re_derives_rather_than_being_handed_timestamps():
    """Configuration crosses the boundary, not resolved event times.

    A run should be reproducible from the recording plus its configuration.
    Passing a list of timestamps would make the derivative depend on GUI state
    that nothing downstream could check against the file.
    """
    pipe = (PKG / "pipeline.py").read_text(encoding="utf-8")
    a = pipe.index("def pipeline_load_file")
    b = pipe.index("\ndef ", a + 10)
    body = pipe[a:b]
    assert "extract_events(file_path, sources" in body
    assert "extract_stim_times(file_path, marker_name)" in body, (
        "with no sources the marker path must be untouched")


def test_the_sidecar_records_how_stimuli_were_identified():
    bids = (PKG / "bids.py").read_text(encoding="utf-8")
    pipe = (PKG / "pipeline.py").read_text(encoding="utf-8")
    assert 'd["event_sources"]' in bids
    assert "event_sources=[_s.to_dict()" in pipe, (
        "a run whose events came from a threshold is not reproducible from "
        "its outputs unless the sidecar says so")


# ── It must look alive the moment it opens ───────────────────────────────────

def test_the_dialogue_never_opens_with_nothing_selected():
    """
    An empty list meant an empty editor and an empty preview, so the dialogue
    looked broken until the analyst guessed that Add came first. A file with no
    configured sources uses its own markers; that is what the first row now
    describes -- the existing behaviour made visible and editable rather than
    implied by an empty list.
    """
    a = DLG.index("if not self._sources:")
    b = DLG.index("self._load_selected()", a)
    body = DLG[a:b]
    assert 'EventSource(' in body
    assert 'kind="embedded"' in body
    assert "self.listbox.selection_set(0)" in DLG


def test_there_is_an_explicit_preview_control():
    assert 'text="Preview"' in DLG
    assert "def _preview_now" in DLG
    a = DLG.index("def _preview_now")
    b = DLG.index("\n    def ", a + 10)
    body = DLG[a:b]
    assert "_apply_edits()" in body, "it must use what is currently typed"
    assert "force=True" in body


def test_selection_changes_force_the_draw_rather_than_scheduling_it():
    """
    draw_idle only schedules. After Add, the scheduled draw had not run by the
    time the analyst looked, so a newly added source appeared to have no
    preview until its row was clicked again.
    """
    assert "self.canvas.draw() if force else self.canvas.draw_idle()" in DLG
    for site in ("def _add", "def _remove"):
        a = DLG.index(site)
        b = DLG.index("\n    def ", a + 10)
        assert "force=True" in DLG[a:b], f"{site} does not force the redraw"
    a = DLG.index('self.listbox.bind("<<ListboxSelect>>"')
    assert "force=True" in DLG[a:a + 160]


def test_removing_the_last_source_leaves_a_valid_selection():
    """Otherwise the editor empties and the dialogue looks broken again."""
    a = DLG.index("def _remove")
    b = DLG.index("\n    def ", a + 10)
    body = DLG[a:b]
    assert "min(i, len(self._sources) - 1)" in body


def test_merge_warnings_reach_the_run_log():
    """The run must not be quieter than the dialogue.

    merge_event_sources reports two sources claiming the same stimulus type,
    and events from different sources landing near-simultaneously. Both are
    usually misconfigurations, and swallowing them in the analysis path would
    hide the reason a trial count looks wrong.
    """
    pipe = (PKG / "pipeline.py").read_text(encoding="utf-8")
    a = pipe.index("def pipeline_load_file")
    b = pipe.index("\ndef ", a + 10)
    body = pipe[a:b]
    assert "warn=None" in body
    assert "_warnings" in body and "warn(_w)" in body
    assert "warn=lambda m: log_callback(" in pipe, \
        "run_pipeline must give pipeline_load_file somewhere to report"


def test_per_channel_sources_are_independent():
    """A weak trigger on one electrode needs its own level.

    The same stimuli at a third of the amplitude are invisible at the level
    that works elsewhere -- which is the case that made per-channel sources
    necessary rather than a convenience.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "es_probe", PKG / "event_sources.py")
    es = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(es)

    fs = 5000.0
    trig = np.zeros(int(fs * 20))
    for t in (2.0, 5.0, 8.0, 11.0, 14.0, 17.0):
        trig[int(t * fs):int(t * fs) + 5] = 5.0
    weak = trig * 0.3

    assert len(es.detect_threshold_crossings(trig, fs, 2.5, "rising", 50.0)) == 6
    assert len(es.detect_threshold_crossings(weak, fs, 2.5, "rising", 50.0)) == 0
    assert len(es.detect_threshold_crossings(weak, fs, 0.75, "rising", 50.0)) == 6


# ── switching kind must not carry an impossible channel ──────────────────────

def test_switching_kind_resets_a_channel_the_new_kind_cannot_use():
    """Embedded and threshold share v_channel but not their option lists.

    A readonly Combobox displays whatever its variable holds, even a value
    absent from its values, so switching to Trigger channel showed the
    embedded source's name over the analogue list -- and _apply_edits wrote
    that impossible name back, leaving the preview empty until the analyst
    happened to reselect.
    """
    a = DLG_NOW.index("def _kind_changed")
    b = DLG_NOW.index("\n    def ", a + 10)
    body = DLG_NOW[a:b]
    assert "_options_for(" in body
    assert "self.v_channel.set(" in body
    i = body.index("v_channel.set")
    j = body.index("_build_fields")
    assert i < j, "the channel must be corrected before the fields are rebuilt"


def test_the_option_lists_are_per_kind():
    a = DLG_NOW.index("def _options_for")
    b = DLG_NOW.index("\n    def ", a + 10)
    body = DLG_NOW[a:b]
    assert '"embedded"' in body and '"analogue"' in body


def test_edit_traces_are_bound_once():
    """They were added inside _build_fields, which runs on every rebuild.

    Each kind switch added another callback, so after a few one keystroke
    re-read the channel several times over.
    """
    a = DLG_NOW.index("def _build_fields")
    b = DLG_NOW.index("\n    def ", a + 10)
    assert "trace_add" not in DLG_NOW[a:b], \
        "traces must not be added where the fields are rebuilt"
    assert "_traces_bound" in DLG_NOW, "binding must be idempotent"


# ── every kind previews ──────────────────────────────────────────────────────

def test_no_kind_falls_through_to_placeholder_text():
    for dead in ("No preview for this kind of source",
                 "The file's own events need no level"):
        assert dead not in DLG_NOW, \
            "every source kind draws a signal with its events now"


def test_the_preview_resolves_events_the_way_the_run_does():
    """Same call as pipeline_load_file, so the picture matches the analysis."""
    a = DLG_NOW.index("def _resolve_events")
    b = DLG_NOW.index("\n    def ", a + 10)
    assert "extract_events(" in DLG_NOW[a:b]
    pipe = (PKG / "pipeline.py").read_text(encoding="utf-8")
    assert "extract_events(file_path, sources" in pipe


def test_threshold_still_previews_from_the_cached_array():
    """Typing in the level box must not reopen the recording each keystroke."""
    a = DLG_NOW.index("def _update_preview")
    b = DLG_NOW.index("\n    def ", a + 10)
    body = DLG_NOW[a:b]
    assert "detect_threshold_crossings(" in body
    assert "extract_events(" not in body, \
        "the level box must read the cached array, not reopen the file"
    assert body.index("detect_threshold_crossings(") < body.index(
        "_resolve_events("), "the threshold branch comes first"


def test_a_display_channel_is_offered_for_the_kinds_that_name_none():
    body = _code_of("_overview_channel")
    assert 'src.kind == \'threshold\'' in body or \
           'src.kind == "threshold"' in body, \
        "the overview stays on the channel being thresholded"
    assert "v_display" in _code_of("_display_channel")
    assert "Show against:" in DLG_NOW


# ── the container must stay a mapping ────────────────────────────────────────

def test_every_assignment_to_event_sources_is_a_mapping():
    """One missed site turns the per-channel dict back into a list.

    _open_event_sources kept `self.event_sources = dlg.result`, so configuring
    any source replaced the dict with a list and the next Preview or Run died
    on .items(). Compiling and the other tests both passed, because they
    checked the snapshot and the worker rather than the site that writes.

    This walks every assignment instead of naming the ones already known.
    """
    import ast

    tree = ast.parse(APP)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            # self.event_sources = <value>
            if isinstance(tgt, ast.Attribute) and tgt.attr == "event_sources" \
                    and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                v = node.value
                ok = (isinstance(v, ast.Dict)
                      or isinstance(v, ast.DictComp)
                      or (isinstance(v, ast.IfExp)
                          and isinstance(v.body, (ast.Dict, ast.DictComp))))
                if not ok:
                    offenders.append((node.lineno, ast.unparse(node)[:80]))
    assert not offenders, (
        "self.event_sources must always be assigned a mapping "
        "{channel_idx: [EventSource]}; found: " + str(offenders))


def test_the_dialogue_edits_one_channel():
    a = APP.index("def _open_event_sources")
    b = APP.index("\n    def ", a + 10)
    body = APP[a:b]
    assert "self.event_sources[_ch] = dlg.result" in body, \
        "the dialogue configures the current channel, not the whole file"
    assert "self.event_sources.get(_ch, [])" in body, \
        "and it must open on that channel's existing sources"
    assert "dlg.copy_to_all" in body, "copy-to-all must be honoured"


def test_the_labels_tab_previews_the_configured_channel():
    a = APP.index("def _apply_event_sources")
    b = APP.index("\n    def ", a + 10)
    assert "self.event_sources.get(self.channel_idx" in APP[a:b], \
        "tab 1a must show the channel being configured, not a flat list"


# ── stepping through single events ───────────────────────────────────────────

def test_there_are_two_axes_not_one():
    """Overview and detail together, not behind a toggle.

    A whole recording of marks answers "did I catch them all" and nothing
    else: at 176 events over 72 seconds the trace is a picket fence and no
    single mark can be judged. Both questions are live while a level is being
    chosen.
    """
    assert "add_subplot(211)" in DLG_NOW and "add_subplot(212)" in DLG_NOW
    assert "self.ax_detail" in DLG_NOW


def test_the_detail_view_reads_raw_samples():
    """Not a slice of the decimated overview.

    decimate_for_preview collapses the recording to a few thousand min/max
    columns, so a 200 ms window covers a handful of them. The result would be
    a blocky trace that still looks like data -- wrong in the way that is hard
    to notice.
    """
    a = DLG_NOW.index("def _draw_detail")
    b = DLG_NOW.index("\n    def ", a + 10)
    body = DLG_NOW[a:b]
    assert "decimate_for_preview(" not in body, \
        "the detail window must not be built from the decimated overview"
    assert "sig[a:b]" in body, "the detail window must index the raw array"


def test_navigation_clamps_at_both_ends():
    a = DLG_NOW.index("def _event_step")
    b = DLG_NOW.index("\n    def ", a + 10)
    body = DLG_NOW[a:b]
    assert "max(" in body and "min(" in body, "stepping must not run off either end"


def test_arrow_keys_do_nothing_while_typing():
    """The channel and edge dropdowns consume Left and Right when focused.

    An unconditional binding would change the source instead of the event, so
    arrowing away from a dropdown would silently reconfigure the thing being
    previewed.
    """
    a = DLG_NOW.index("def _on_nav_key")
    b = DLG_NOW.index("\n    def ", a + 10)
    body = DLG_NOW[a:b]
    assert "focus_get()" in body
    for widget in ("tk.Entry", "ttk.Combobox", "tk.Listbox"):
        assert widget in body
    assert 'self.top.bind("<Left>"' in DLG_NOW
    assert 'self.top.bind("<Home>"' in DLG_NOW


def test_changing_source_resets_the_position():
    a = DLG_NOW.index("def _load_selected")
    b = DLG_NOW.index("\n    def ", a + 10)
    assert "self._cur_event = 0" in DLG_NOW[a:b], \
        "a different source has a different event list"


def test_adjusting_a_level_keeps_your_place():
    """Clamped, not reset: retuning a level should not lose where you were."""
    a = DLG_NOW.index("def _update_preview")
    b = DLG_NOW.index("\n    def ", a + 10)
    body = DLG_NOW[a:b]
    assert "self._cur_event = min(self._cur_event" in body


def test_the_detail_width_defaults_to_the_analysis_window():
    a = APP.index("def _open_event_sources")
    b = APP.index("\n    def ", a + 10)
    body = APP[a:b]
    assert "window_ms=" in body
    assert "pre_time.get()" in body and "post_time.get()" in body


def test_an_event_outside_the_recording_is_reported_not_blank():
    """Some formats carry markers past the end of their waveform.

    Clamping the slice bounds gives an empty window; drawing nothing would
    leave blank axes under a position label still claiming to show that event.
    """
    a = DLG_NOW.index("def _draw_detail")
    b = DLG_NOW.index("\n    def ", a + 10)
    body = DLG_NOW[a:b]
    assert "lies outside the recording" in body
    i = body.index("if b <= a:")
    j = body.index("return", i)
    assert "where_var.set" in body[i:j], \
        "the position label must not go stale"
    assert "min(len(sig)" in body, "both bounds must be clamped to the array"


# ── the two panes answer different questions ─────────────────────────────────

def test_show_against_drives_the_detail_view_even_for_a_threshold():
    """The dropdown was inert in the case it exists for.

    _display_channel returned src.channel whenever the source was a threshold,
    so both panes drew the trigger and choosing an EMG channel did nothing.
    Setting a level is answered on the trigger; whether an event lands on a
    response can only be answered somewhere else.
    """
    assert _params_of("_display_channel") == ["self"], \
        "the detail channel must not depend on the source at all"
    body = _code_of("_display_channel")
    assert "v_display" in body
    assert "src" not in body


def test_the_overview_keeps_the_thresholded_channel():
    """A level drawn over a different signal would mean nothing."""
    body = _code_of("_overview_channel")
    assert "src.kind" in body and "src.channel" in body


def test_the_detail_reads_its_own_channel_when_they_differ():
    a = DLG_NOW.index("def _update_preview")
    b = DLG_NOW.index("\n    def ", a + 10)
    body = DLG_NOW[a:b]
    assert "_det != shown" in body, \
        "the detail pane must fetch the chosen channel, not reuse the overview's"
    assert "_channel_data(_det)" in body


def test_both_axes_say_which_channel_they_show():
    assert "Whole recording — " in DLG_NOW
    assert "This event — " in DLG_NOW


def test_changing_the_display_channel_redraws():
    assert 'self.v_display.trace_add("write"' in DLG_NOW
    a = DLG_NOW.index('self.v_display.trace_add("write"')
    assert "_update_preview()" in DLG_NOW[a:a + 120], \
        "a full refresh, since the detail source is chosen there"


# ── anything that shows events must honour the configuration ─────────────────

def test_the_range_picker_uses_the_configured_sources():
    """It drew the file's own comments after a source had been chosen.

    The recording that exposed this carries 162 'Trigger' comments and 6
    'Start Task'. Choosing Trigger and then being shown Start Task is not a
    cosmetic difference: it is a different set of stimuli, presented as though
    it were the one just configured.
    """
    body = _code_of("_crop_selector", src=APP)
    assert "_configured_events(" in body
    assert "extract_stim_times(" not in body, \
        "the range picker must not read markers behind the configuration"


def test_the_helper_falls_back_to_markers_when_nothing_is_configured():
    """Every file behaves as it always did until someone sets a source."""
    body = _code_of("_configured_events", src=APP)
    assert "if not sources:" in body
    assert "extract_events(" in body
    assert "self.channel_idx" in body, "sources are per channel"


def test_the_helper_reports_rather_than_hides_a_broken_source():
    body = _code_of("_configured_events", src=APP)
    assert "except Exception" in body
    assert "self.log(" in body, \
        "falling back silently would hide that the configuration was ignored"


# ── the Event sources button must not discard the channel choice ─────────────

def test_the_assignment_dialogue_saves_before_opening_event_sources():
    """It called dlg.destroy(), which is Cancel by another name.

    The ticked channels were discarded, channel_idx stayed on whichever
    channel was current before the file opened, and the marker choice was
    never set. Sources ticked for Channel 3 were filed against Channel 1, the
    range picker drew Channel 1, and the marker dropdown kept the load-time
    discovery — three symptoms from one missing call.
    """
    import re

    # every Event sources button in app.py
    for m in re.finditer(r'text="Event sources\\u2026"', APP):
        window = APP[m.start():m.start() + 400]
        assert "dlg.destroy(),\n" not in window, \
            "the button must commit the selection, not discard it"
        assert "command=_" in window, \
            "it should call a named handler that saves first"


def test_the_generic_handler_applies_the_choice_before_configuring():
    body = _code_of("_to_event_sources", src=APP)
    assert "_save()" in body
    assert "_apply_choice()" in body
    assert body.index("_save()") < body.index("_open_event_sources"), \
        "the channel must be current before its sources are configured"


def test_the_spike2_handler_defers_until_the_channels_exist():
    """Its indices do not exist until the dropdown is populated."""
    body = _code_of("_smr_to_event_sources", src=APP)
    assert "_ok()" in body
    assert "_want_event_sources_after_load" in body
    assert "_open_event_sources" not in body, \
        "opening immediately would file sources against a stale channel"
    assert "_want_event_sources_after_load" in APP.split(
        "_update_marker_dropdown()")[1][:600], \
        "the deferred open must run after the channel selection is applied"


def test_choosing_no_channel_is_refused_rather_than_defaulted():
    body = _code_of("_to_event_sources", src=APP)
    assert "showwarning" in body, \
        "sources belong to a channel; guessing one would file them wrongly"


# ── the load flow must not overwrite a configured selection ──────────────────

def test_the_load_flow_rebuild_honours_configured_sources():
    """It rebuilt tab 1a from every marker found, after the dialogue had run.

    The assignment dialogue happens partway through _browse_file_path, so a
    source chosen there built tab 1a from the right events and this later
    rebuild replaced them with all of them. On a LabChart file carrying 162
    'Trigger' comments and 6 'Start Task', choosing Trigger still left both
    rows on the tab -- each configurable, each analysed.
    """
    body = _code_of("_browse_file_path", src=APP)
    assert "_configured_events(" in body, \
        "the rebuild must consult the configuration"
    i = body.index("_configured_events(")
    j = body.index("_build_labels_tab(")
    assert i < j, "the configured set must be resolved before the tab is built"


def test_the_marker_dropdown_follows_the_configured_sources():
    """It named 'Start Task' while the analysis was to use 'Trigger'."""
    body = _code_of("_browse_file_path", src=APP)
    assert "self.available_markers = sorted(_cfg_events)" in body
    assert "self.marker_choice.set(" in body


def test_an_unconfigured_file_still_shows_everything_it_carries():
    """Discovery is the right behaviour until someone chooses otherwise."""
    body = _code_of("_browse_file_path", src=APP)
    assert "if _cfg_sources:" in body, \
        "the override must be conditional on sources actually being set"


# ── the delay scan must measure against the events being analysed ────────────

def test_the_delay_scan_uses_the_configured_events():
    """A delay is the offset between an event and the stimulus artefact.

    Measured against the file's markers while a threshold source is
    configured, it measures the wrong thing and proposes it with the same
    confidence as the right one -- and every latency in the file shifts by
    whatever it proposes.
    """
    body = _code_of("_detect_event_delays", src=APP)
    assert "_configured_events(" in body
    assert "extract_stim_times(" not in body, \
        "the scan must not read markers behind the configuration"


def test_the_delay_scan_does_not_scan_non_stimulus_markers():
    """162 'Trigger' comments beside 6 'Start Task'.

    Scanning both reported a delay for a type with no row to fill in and
    counted it in "Scanned N type(s)".
    """
    body = _code_of("_detect_event_delays", src=APP)
    i = body.index("_configured_events(")
    j = body.index("scan_event_delays(")
    assert i < j, "the event set must be narrowed before it is scanned"


def test_the_remaining_direct_reads_are_discovery_only():
    """Whatever still calls extract_stim_times must run before configuration.

    Four calls remain in app.py. Each populates the marker list the Event
    sources dialogue itself offers, so routing them through the configuration
    would be circular. This pins the count so a fifth cannot appear unnoticed.
    """
    import ast

    tree = ast.parse(APP)
    sites = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            for m in ast.walk(n):
                if isinstance(m, ast.Call) and (
                        getattr(m.func, "id", "") in ("extract_stim_times", "_io_est")
                        or getattr(m.func, "attr", "") == "extract_stim_times"):
                    sites.append((m.lineno, n.name))
    names = {fn for _, fn in sites}
    assert names <= {"_browse_file_path"}, (
        "extract_stim_times outside the load flow reads events behind the "
        f"configuration: {sorted(names - {'_browse_file_path'})}")
    # The count is deliberately NOT pinned. It rises by one for every format
    # that gains a load branch -- signal_mat took it from four to five -- so a
    # fixed number would churn without saying anything the location check
    # above does not already say. The invariant that matters is WHERE these
    # calls are: discovery belongs to the load flow, and a read anywhere else
    # is one that ignores the configuration.
    assert len(sites) >= 1, "the discovery reads have disappeared entirely"


# ── Add must not reintroduce a label the analyst just excluded ───────────────

def test_add_skips_labels_an_existing_source_already_names():
    """Defaulting to embedded[0] undid the choice that had just been made.

    On a file whose labels sort as ['Start Task', 'Trigger'], setting the
    first source to Trigger and pressing Add appended a second pointing at
    Start Task. Both merge, so the excluded type came back as a row on tab 1a.
    The merge does not object: it only warns when two sources claim the SAME
    type, and these are different ones.
    """
    body = _code_of("_add")
    assert "taken" in body and "free" in body
    assert "src.channel for src in self._sources" in body


def test_add_still_works_when_every_label_is_taken():
    """A second threshold source on a file with one comment type is legal."""
    body = _code_of("_add")
    assert "if free else" in body, "it must fall back rather than refuse"


def test_the_dialogue_reports_what_all_sources_together_produce():
    """The count line describes the selected row only.

    A second source quietly widening the analysis was invisible until tab 1a
    had already been rebuilt with the extra type on it.
    """
    body = _code_of("_refresh_total")
    assert "extract_events(" in body
    assert "self._sources" in body, "the summary must cover every source"
    assert "total_var" in DLG_NOW
    assert "_refresh_total()" in _code_of("_update_preview")


def test_the_summary_uses_the_same_call_as_the_analysis():
    """Otherwise it could promise types the run will not produce."""
    body = _code_of("_refresh_total")
    assert "from .io import extract_events" in body


def test_the_add_button_says_it_adds_another():
    """'Add' was read as 'confirm this selection into the list'.

    It appends a second source, which on a file with two comment types brings
    the excluded one back as a row on tab 1a. The selected row is already
    live; OK accepts it.
    """
    assert 'text="Add another"' in DLG_NOW
    assert 'text="Add"' not in DLG_NOW


def test_the_help_says_one_source_is_usually_enough():
    assert "One source is usually enough" in DLG_NOW
    assert "every source adds its own row" in DLG_NOW


# ── the default source must not narrow anything ──────────────────────────────

def test_the_default_source_takes_every_marker():
    """It seeded channel=embedded[0], which is a change of behaviour.

    On a recording whose labels sort as ['Start Task', 'Trigger'], simply
    opening the dialogue restricted the analysis to 'Start Task' -- and the
    only way back was to delete that row.
    """
    body = _code_of("__init__")
    i = body.index("if not self._sources")
    seg = body[i:i + 400]
    assert "channel=''" in seg or 'channel=""' in seg
    assert "embedded[0]" not in seg


def test_every_marker_is_an_explicit_option():
    """An empty string in a dropdown reads as an empty dropdown."""
    assert "ALL_LABELS" in DLG_NOW
    body = _code_of("_build_fields")
    assert "[ALL_LABELS] +" in body


def test_the_display_label_round_trips_to_an_empty_channel():
    assert "'' if _ch == ALL_LABELS else _ch" in DLG_NOW or \
           '"" if _ch == ALL_LABELS else _ch' in DLG_NOW
    body = _code_of("_load_selected")
    assert "ALL_LABELS" in body, "an empty channel must display as the option"


# ── no early return may leave a pane describing the last selection ───────────

def test_the_panes_are_cleared_together():
    """Clearing only the overview left the summary, position label and
    single-event view describing the previous selection -- which reads as the
    dialogue not updating until something is clicked."""
    body = _code_of("_clear_panes")
    for what in ("self.ax.clear()", "self.ax_detail.clear()",
                 "self.count_var.set('')", "self.where_var.set('')",
                 "self._refresh_total()", "self._event_times = []"):
        assert what in body, f"_clear_panes must reset {what}"


def test_no_early_return_clears_only_the_overview():
    import ast

    for fn in ("_update_preview", "_load_selected"):
        tree = ast.parse(_code_of(fn))
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                seg = ast.unparse(node)
                if "return" in seg and "canvas.draw" in seg:
                    assert "_clear_panes" in seg, (
                        f"{fn} has a return path that leaves panes stale")


def test_removing_the_last_source_keeps_a_selection():
    """An empty list is the same instruction as one source reading everything.

    Leaving it empty gave a dialogue with no selection, no fields, and panes
    still showing whatever was there before.
    """
    body = _code_of("_remove")
    assert "if not self._sources:" in body
    assert "EventSource(kind='embedded', channel='')" in body or \
           'EventSource(kind="embedded", channel="")' in body


def test_show_against_starts_on_the_channel_being_configured():
    body = _code_of("__init__")
    assert "channel_name if channel_name in _an" in body


def test_a_chosen_marker_source_narrows_the_analysis():
    """Picking 'Trigger' in Channel Assignment left every label on tab 1a."""
    body = _code_of("_browse_file_path", src=APP)
    assert "stim_types_found = {_mk}" in body
    i = body.index("_mk in stim_types_found")
    assert "len(stim_types_found) > 1" in body[i:i + 200], \
        "narrowing to the only type there is would be a no-op with a log line"


# ── jumping to an event by number ────────────────────────────────────────────

def test_a_typed_event_number_jumps_there():
    """160+ events makes stepping an unreasonable way to reach the last one."""
    body = _code_of("_event_goto")
    assert "self.v_goto.get()" in body
    assert "self._cur_event = max(0, min(n - 1" in body
    assert "_update_preview()" in body


def test_the_jump_clamps_rather_than_refusing():
    """On 162 events, typing 200 means the last one."""
    body = _code_of("_event_goto")
    assert "min(n - 1, len(self._event_times) - 1)" in body
    assert "showwarning" not in body and "showerror" not in body


def test_a_non_number_is_ignored_not_raised():
    body = _code_of("_event_goto")
    assert "except (TypeError, ValueError)" in body
    assert "_sync_goto()" in body


def test_the_jump_is_on_enter_not_every_keystroke():
    """A trace would send you to event 1 on the way to typing 16."""
    body = _code_of("__init__")
    assert "_goto_entry.bind('<Return>'" in body or \
           '_goto_entry.bind("<Return>"' in body
    assert "self.v_goto.trace_add" not in DLG_NOW


def test_the_box_shows_the_position_but_not_while_being_typed_into():
    """The preview refreshes on a level-box trace, so this runs at any moment.

    Overwriting the box mid-number would fight the typing.
    """
    body = _code_of("_sync_goto")
    assert "focus_get() is self._goto_entry" in body
    assert "return" in body


def test_the_position_is_synced_everywhere_it_can_change():
    for fn in ("_draw_detail", "_clear_panes"):
        assert "_sync_goto()" in _code_of(fn), f"{fn} must keep the box honest"


def test_first_and_last_are_on_screen():
    """Home and End already worked; nothing announced them."""
    body = _code_of("__init__")
    assert "_event_jump('first')" in body or '_event_jump("first")' in body
    assert "_event_jump('last')" in body or '_event_jump("last")' in body
