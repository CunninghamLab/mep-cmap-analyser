"""
The Inspector must measure amplitude over the window the analysis used.

Amplitude window anchoring places the window per stimulus type, from that
type's median onset. The Inspector seeded its peak markers from the file-wide
window in tab 1c and knew nothing about anchoring, so with anchoring enabled
the review searched a different interval from the analysis. The visible symptom
is a peak marker that ignores the largest excursion in the response, because
that excursion falls outside the window the Inspector happened to use -- and
the peak-to-peak shown during review is then not the one in the results file.
"""

import ast
import pathlib

SRC = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "inspector.py").read_text(encoding="utf-8")
APP = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "app.py").read_text(encoding="utf-8")
PIPE = (pathlib.Path(__file__).resolve().parent.parent
        / "mep_cmap" / "pipeline.py").read_text(encoding="utf-8")


def test_inspector_accepts_per_type_windows():
    assert "ptp_windows_by_type=None," in SRC
    assert "self.ptp_windows_by_type  = dict(ptp_windows_by_type or {})" in SRC


def test_peak_search_uses_the_per_type_window():
    a = SRC.index("# … but if the user defined a PTP window")
    b = SRC.index("dt_ms  = self.t[1] - self.t[0]", a)
    body = SRC[a:b]
    assert "self._ptp_window_ms()" in body, (
        "the peak search must use the analysis window for this stimulus type"
    )
    assert "(self.t >= self.ptp_start_ms)" not in body, (
        "the file-wide window is still being used directly"
    )


def test_the_file_wide_window_remains_the_fallback():
    a = SRC.index("def _ptp_window_ms")
    b = SRC.index("\n    def ", a + 10)
    body = SRC[a:b]
    assert "return self.ptp_start_ms, self.ptp_end_ms" in body
    assert "self.cur_type if stim_type is None" in body


def test_a_degenerate_window_falls_back_rather_than_being_used():
    """An inverted or unparseable pair must not silently produce an empty mask."""
    a = SRC.index("def _ptp_window_ms")
    b = SRC.index("\n    def ", a + 10)
    body = SRC[a:b]
    assert "if b > a:" in body
    assert "except Exception:" in body


def test_app_forwards_the_windows():
    assert "ptp_windows_by_type=None):" in APP
    assert "ptp_windows_by_type = ptp_windows_by_type," in APP


def test_pipeline_sends_the_windows_it_measured_with():
    assert "ptp_windows_by_type=_ptp_ms_by_type)" in PIPE
    a = PIPE.index("_ptp_ms_by_type = {}")
    b = PIPE.index("segments_metadata = show_inspector_cb", a)
    body = PIPE[a:b]
    assert "ptp_window_by_type" in body
    # converted from sample indices to ms relative to the stimulus
    assert "samples_before" in body and "1000.0 / fs" in body


def test_all_three_modules_still_parse():
    for src in (SRC, APP, PIPE):
        ast.parse(src)


def test_the_window_is_read_by_index_not_by_unpacking():
    """
    ptp_window_for_stim_type returns THREE values -- (start_idx, end_idx,
    ms_pair_or_None). Destructuring its result to a pair raised "too many
    values to unpack" only once an analysis actually reached the inspector,
    after every window had already been computed and logged.
    """
    a = PIPE.index("_ptp_ms_by_type = {}")
    b = PIPE.index("segments_metadata = show_inspector_cb", a)
    body = PIPE[a:b]
    assert "for _st, _win in" in body, "the result is being destructured again"
    assert "_win[0]" in body and "_win[1]" in body


def test_the_conversion_matches_what_the_function_returns():
    """
    Exercise it rather than trusting the shape: the third element is already a
    millisecond pair, so it doubles as a check on the index arithmetic.
    """
    import sys
    import types as _t

    for _n in ('tkinter', 'tkinter.ttk', 'tkinter.filedialog',
               'tkinter.messagebox', 'tkinter.font', 'tkinter.simpledialog',
               'tkinter.colorchooser', 'tkinter.scrolledtext'):
        if _n not in sys.modules:
            _m = _t.ModuleType(_n)
            _m.__getattr__ = lambda x: _t.SimpleNamespace()
            sys.modules[_n] = _m

    from mep_cmap.pipeline import PipelineConfig, ptp_window_for_stim_type

    cfg = PipelineConfig(pre_ms=20, post_ms=400, ptp_start=10, ptp_end=50,
                         ptp_anchor=True, ptp_anchor_min_trials=4,
                         ptp_anchor_pre_ms=2.0, ptp_anchor_duration_ms=40.0)
    sb, fs = 100, 5000.0
    win = ptp_window_for_stim_type(
        "A", {i: 3.8 for i in range(5)}, fs, cfg,
        sb + 50, sb + 250, sb, log_callback=lambda *a: None)

    assert len(win) == 3, (
        "the consumer in run_pipeline reads win[0] and win[1]; if the arity "
        "changes that consumer must change with it"
    )
    derived = ((win[0] - sb) * 1000.0 / fs, (win[1] - sb) * 1000.0 / fs)
    assert win[2] is not None
    assert abs(derived[0] - win[2][0]) < 1e-6
    assert abs(derived[1] - win[2][1]) < 1e-6
