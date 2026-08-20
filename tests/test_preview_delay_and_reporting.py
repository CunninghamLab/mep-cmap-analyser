"""
Three things the tool knew and did not say.

  * the preview cut its segments without the event delay, found no onsets, and
    reported that the analysis would find none either -- while the run detected
    all 21
  * a value left over from a corrected vocabulary went into *_nibs.tsv unremarked
  * every MEP's area read as 0.000 mV·s
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREVIEW = (ROOT / "mep_cmap" / "preview.py").read_text(encoding="utf-8")
INSPECTOR = (ROOT / "mep_cmap" / "inspector.py").read_text(encoding="utf-8")


def _function(name, src):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found")


# ── the preview cuts what the run cuts ───────────────────────────────────────

def test_the_preview_reads_the_delay_per_channel():
    """delay_ms_map is per channel, held in chan_settings. The flat copy
    belongs to whichever channel was last harvested, and was empty here -- so
    the preview cut with no delay, the response sat 17.5 ms late, outside the
    latency window, and no onset was found on any trial."""
    body = _function("_preview_cut", PREVIEW)
    assert "params.get('delay_ms_map')" not in body
    assert "_pv_own('delay_ms_map'" in body or '_pv_own("delay_ms_map"' in body


def test_the_preview_reads_the_window_per_channel():
    """Same map, same fault: cutting to the wrong epoch offers trials the run
    will not produce."""
    body = _function("_preview_cut", PREVIEW)
    assert "params.get('window_map')" not in body
    assert "_pv_own('window_map'" in body or '_pv_own("window_map"' in body


def test_the_flat_map_is_still_the_fallback():
    """A channel with no snapshot yet must not lose its settings entirely."""
    body = _function("_preview_cut", PREVIEW)
    assert "params.get(key, default)" in body


# ── a value outside its vocabulary is named ──────────────────────────────────

def test_a_stale_enum_value_is_reported():
    from mep_cmap.bidsify import _vocabulary_problems
    from mep_cmap.bids_schema import load_schema
    from mep_cmap import stim_params as sp

    schema = load_schema()
    sets = [sp.StimParamSet("MEP", nibs_type="TMS",
                            values={"CurrentDirection": "AP"})]
    msgs = _vocabulary_problems(sets, schema)
    assert len(msgs) == 1
    assert "CurrentDirection" in msgs[0]
    assert "'AP'" in msgs[0]
    assert "co-cl" in msgs[0], "the allowed values must be listed"
    assert "MEP" in msgs[0], "the parameter set must be named"


def test_a_valid_value_is_not_reported():
    from mep_cmap.bidsify import _vocabulary_problems
    from mep_cmap.bids_schema import load_schema
    from mep_cmap import stim_params as sp

    schema = load_schema()
    sets = [sp.StimParamSet("MEP", nibs_type="TMS",
                            values={"CurrentDirection": "co-cl"})]
    assert _vocabulary_problems(sets, schema) == []


def test_a_free_text_field_is_not_policed():
    from mep_cmap.bidsify import _vocabulary_problems
    from mep_cmap.bids_schema import load_schema
    from mep_cmap import stim_params as sp

    schema = load_schema()
    sets = [sp.StimParamSet("MEP", nibs_type="TMS",
                            values={"TargetRegion": "M1 hand hotspot"})]
    assert _vocabulary_problems(sets, schema) == []


def test_a_blank_is_not_reported():
    from mep_cmap.bidsify import _vocabulary_problems
    from mep_cmap.bids_schema import load_schema
    from mep_cmap import stim_params as sp

    schema = load_schema()
    for blank in ("", "   ", None):
        sets = [sp.StimParamSet("MEP", nibs_type="TMS",
                                values={"CurrentDirection": blank})]
        assert _vocabulary_problems(sets, schema) == []


def test_it_is_reported_not_corrected():
    """PA cannot be translated into a winding direction: they describe
    different things, so correcting it would invent a measurement."""
    from mep_cmap.bidsify import _vocabulary_problems
    from mep_cmap.bids_schema import load_schema
    from mep_cmap import stim_params as sp

    schema = load_schema()
    s = sp.StimParamSet("MEP", nibs_type="TMS",
                        values={"CurrentDirection": "AP"})
    _vocabulary_problems([s], schema)
    assert s.values["CurrentDirection"] == "AP"


def test_the_conversion_reports_them():
    body = _function("write_nibs_sidecar", (ROOT / "mep_cmap" / "bidsify.py")
                     .read_text(encoding="utf-8"))
    assert "_vocabulary_problems" in body


# ── an area a human can read ─────────────────────────────────────────────────

def test_the_area_is_shown_in_microvolt_seconds():
    """A 0.1 mV response over 15 ms is about 3e-4 mV·s: every MEP in range
    rounded to 0.000 at three decimals."""
    body = _function("_refresh_status", INSPECTOR)
    assert "auc_val * 1000.0" in body
    assert "mV\u00b7s" not in body


def test_the_results_file_keeps_millivolt_seconds():
    """Display only. Changing the stored unit would move every published
    value and break comparison with earlier results."""
    from mep_cmap.pipeline import LAT_COLS
    assert "AUC(mV*s)" in LAT_COLS
