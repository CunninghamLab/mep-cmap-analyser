"""
The analysis runs once per selected channel.

Each pass is independent and uses that channel's own tab 1a settings, so an
iSP on one channel and a contralateral MEP on another are measured with their
own latency profiles. The pipeline is unchanged -- it still receives one flat
set of maps per run and knows nothing about there being other channels.

Sequential rather than concurrent, with the Data Inspector opening in turn:
that also means a channel cannot be selected and then forgotten.
"""

import ast
import pathlib

APP = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "app.py").read_text(encoding="utf-8")
PIPE = (pathlib.Path(__file__).resolve().parent.parent
        / "mep_cmap" / "pipeline.py").read_text(encoding="utf-8")


def _worker_body():
    a = APP.index("# -------- run the heavy pipeline, once per channel")
    b = APP.index('"✅ Analysis complete!"', a)
    return APP[a:b]


def test_the_pipeline_is_called_once_per_selected_channel():
    body = _worker_body()
    assert "for _pass, _ch in enumerate(_chan_list" in body
    a = body.index("for _pass, _ch in enumerate")
    assert body.index("run_pipeline(") > a, "the call must be inside the loop"


def test_each_pass_uses_that_channels_own_setup():
    """
    The whole point of per-channel settings. Falling back to the top-level
    values keeps a single-channel run behaving exactly as before.
    """
    body = _worker_body()
    assert "def _own(key, default=None):" in body
    for key in ("label_map", "gap_ms_map", "delay_ms_map", "latency_map",
                "csp_types", "reference_map"):
        assert f'_own("{key}"' in body, f"{key} is not taken per channel"


def test_the_channel_index_passed_is_the_loop_variable():
    """Using params["channel_idx"] would analyse the same channel N times."""
    body = _worker_body()
    assert "channel_idx          = _ch," in body
    assert 'channel_idx          = params["channel_idx"]' not in body


def test_a_single_channel_run_is_unchanged():
    """Defaults to the selected channel, and no channel token in the filename."""
    body = _worker_body()
    assert '_chan_list = params.get("analysis_channels") or [params["channel_idx"]]' in body
    assert "multi_channel        = len(_chan_list) > 1," in body


def test_outputs_are_tagged_with_the_channel_only_when_several_are_run():
    """
    A single-channel analysis must keep the filenames it has always had, so
    existing derivatives, the group merge and any scripts pointing at them go
    on working.
    """
    a = PIPE.index("if channel_label and multi_channel:")
    seg = PIPE[a:a + 300]
    assert "_channel-" in seg
    assert "_sanitise_bids_label" in seg


def test_the_channel_column_is_written_regardless():
    """
    A column that appears only sometimes is worse to work with than one that
    is always there: a single-channel file joining a multi-channel dataset
    must still say which channel it holds.
    """
    a = PIPE.index("def _tag_channel(df):")
    b = PIPE.index("if latency_manual:", a)
    body = PIPE[a:b]
    assert '"Channel"' in body
    assert "multi_channel" not in body, (
        "the column must not be conditional on a multi-channel run"
    )


def test_the_writer_takes_the_label_rather_than_relying_on_scope():
    """
    It was first written inside pipeline_write_outputs referring to a name
    that only exists in run_pipeline -- valid Python, NameError at runtime.
    """
    tree = ast.parse(PIPE)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "pipeline_write_outputs")
    args = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    assert "channel_label" in args


def test_both_modules_still_parse():
    ast.parse(APP)
    ast.parse(PIPE)


def test_the_pipeline_call_is_reachable_in_the_loop_body():
    """
    Indenting the call to move it inside the loop put it INSIDE the helper
    defined just above, after that helper's return -- unreachable dead code.
    The loop then ran once per channel, called nothing, and reported success:
    no error, no output, an empty derivatives folder and a green log.

    A call that is syntactically fine but nested one level too deep is
    invisible to the compiler, to pyflakes and to every behavioural test,
    because nothing raises. This checks the shape directly.
    """
    tree = ast.parse(APP)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        for ch in ast.walk(node):
            if (isinstance(ch, ast.Call) and isinstance(ch.func, ast.Name)
                    and ch.func.id == "run_pipeline"):
                nested = any(
                    isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.Lambda))
                    and d.lineno > node.lineno
                    and d.lineno <= ch.lineno <= (d.end_lineno or 0)
                    for d in ast.walk(node))
                found.append((ch.lineno, nested))

    assert found, "run_pipeline is not called inside any loop"
    for lineno, nested in found:
        assert not nested, (
            f"run_pipeline at line {lineno} sits inside a nested function "
            f"within the loop, so it never executes"
        )


def test_no_statement_follows_a_return_in_the_per_channel_helper():
    """The specific shape of the failure: dead code after a return."""
    tree = ast.parse(APP)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_own":
            for i, st in enumerate(node.body):
                if isinstance(st, ast.Return):
                    assert i == len(node.body) - 1, (
                        "statements follow the return in _own; they cannot run"
                    )


# ── Marker edits belong to one channel ───────────────────────────────────────

def test_each_pass_receives_only_its_own_channels_edits():
    """
    A marker position is an index into ONE channel's waveform. Applied to
    another it means nothing: on a real recording, EMG 1's peak-to-peak marker
    indices landed on EMG 2's trace and produced NEGATIVE peak-to-peak values,
    because the stored "max" sample sat below the stored "min" one. Every
    offset also came back marked "manual" when none had been set by hand.
    """
    body = _worker_body()
    assert 'params.get("chan_segment_meta", {}).get(_ch, {})' in body
    assert "dict(self.segments_metadata)" not in body, (
        "the shared map must not be passed to every channel"
    )


def test_the_inspectors_edits_are_stored_against_the_channel_reviewed():
    assert "self._review_channel_idx = _ch" in APP
    assert "_chan_segment_meta[getattr(self, \"_review_channel_idx\"" in APP


def test_sessions_written_before_this_still_load():
    """
    A flat map belongs to whichever channel was analysed at the time. It is
    restored as the current channel's and left out of the others -- applying
    it to a second channel is the bug.
    """
    a = APP.index('_per_chan = sess.get("chan_segment_meta")')
    b = APP.index("try: self.toggle_bandpass_fields()", a)
    body = APP[a:b]
    assert "else:" in body
    assert 'sess.get("segments_metadata")' in body


def test_peak_to_peak_can_never_be_written_negative():
    """
    It is a magnitude. A negative value propagates into normalisation and
    z-scores as though it were meaningful -- the CSV that exposed this had
    Z_PTP_Within computed from PTP values of -0.12 and -0.29 mV.
    """
    a = PIPE.index('if "ptp_max_idx" in meta and "ptp_min_idx" in meta:')
    seg = PIPE[a:a + 1200]
    assert "if ptp < 0:" in seg
    assert "ptp = abs(ptp)" in seg


def test_the_inspector_seed_uses_only_the_reviewed_channels_edits():
    """
    self.segments_metadata holds whichever channel was reviewed LAST. In a
    multi-channel run the second Inspector opened showing the first's marker
    positions -- indices into a different waveform.

    The analysis path had already been given the right per-channel edits; this
    display path had not. So the review disagreed with the numbers being
    computed, and any marker left untouched was saved back as the wrong
    channel's.
    """
    a = APP.index("_seed = {k: dict(v) for k, v in (auto_meta or {}).items()}")
    b = APP.index("_det_params = self._current_detection_params()", a)
    body = APP[a:b]
    assert "_chan_segment_meta" in body
    assert "_review_channel_idx" in body


def test_a_single_channel_run_still_sees_its_saved_edits():
    """
    Falling back to the flat map keeps single-channel review working, and
    keeps sessions written before edits were stored per channel usable.
    """
    a = APP.index("_seed = {k: dict(v) for k, v in (auto_meta or {}).items()}")
    b = APP.index("_det_params = self._current_detection_params()", a)
    body = APP[a:b]
    assert "if _saved is None:" in body
    assert "_multi_channel_run" in body, (
        "the flat map may be used only when one channel is analysed; in a "
        "multi-channel run there is no way to tell which channel produced it"
    )


def test_an_unattributed_metadata_map_is_not_assigned_to_a_guessed_channel():
    """
    A session written before edits were stored per channel carries one flat
    map with no record of which channel produced it. Assigning it to
    "whatever is selected now" is how EMG 1's marker indices ended up on
    EMG 2's waveform -- and the guess is invisible, because the result looks
    like a detector fault rather than a bookkeeping one.
    """
    a = APP.index('self.segments_metadata = _unpack_meta(sess.get("segments_metadata"))')
    seg = APP[max(0, a - 700):a + 200]
    assert "_chan_segment_meta[self.channel_idx] = dict(" not in seg, (
        "the flat map is still being attributed to the current channel"
    )


def test_per_channel_edits_are_cleared_with_the_flat_map():
    """
    Every site that clears segments_metadata for a new file must clear the
    per-channel copies too, or edits stored against a channel INDEX in the
    previous file are applied to whatever channel shares that index in this
    one -- the same trap as the per-channel 1a settings.
    """
    import re

    flat = [APP[:m.start()].count("\n") + 1
            for m in re.finditer(r"self\.segments_metadata = \{\}", APP)]
    per = [APP[:m.start()].count("\n") + 1
           for m in re.finditer(r"self\._chan_segment_meta = \{\}", APP)]
    # A generous window: the two are declared together in __init__ but with a
    # comment block between them, so "adjacent" is not line-adjacent.
    for ln in flat:
        assert any(abs(ln - p) <= 20 for p in per), (
            f"segments_metadata cleared at line {ln} without clearing the "
            f"per-channel store nearby; edits stored against a channel index "
            f"in the previous file would be applied to this one"
        )
