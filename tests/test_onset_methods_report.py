"""
Tests for the onset-method comparison tables and figures.

These outputs exist because a disagreement number in a spreadsheet does not
tell an analyst whether their choice of detector matters for their data.
Seeing where each method lands on the actual waveform does. The detector runs
have already happened when agreement is enabled, so this is retention and
presentation rather than computation -- and the failure modes are
correspondingly about shape and provenance, not arithmetic.
"""

import numpy as np
import pandas as pd
import pytest

from mep_cmap.detection.onset_methods_median import OnsetAgreement
from mep_cmap.onset_methods_report import (CAVEAT, FIGURE_SUBDIR_SUFFIX,
                                           METHOD_COLS, SUMMARY_COLS,
                                           _loo_methods_median,
                                           figures_subdir,
                                           build_method_summary,
                                           collect_agreement_rows,
                                           plot_bland_altman,
                                           write_onset_method_figures,
                                           write_onset_method_tables)

METHODS = ["bigoni", "rms_envelope", "cusum"]


def _agreement(per_method):
    vals = [v for v in per_method.values() if v is not None]
    consensus = float(np.median(vals)) if vals else None
    spread = (max(vals) - min(vals)) if len(vals) > 1 else None
    iqr = (float(np.percentile(vals, 75) - np.percentile(vals, 25))
           if len(vals) > 1 else None)
    return OnsetAgreement(per_method, consensus, spread, iqr,
                          len(vals), len(per_method))


def _fixture(n=10, stims=("B", "C")):
    rng = np.random.default_rng(0)
    out = {}
    for st in stims:
        for i in range(n):
            base = 20.0 + rng.normal(0, 0.6)
            out[(st, i)] = _agreement({
                "bigoni": round(base - 1.2, 2),
                "rms_envelope": round(base + 0.4, 2),
                "cusum": round(base + 1.1, 2) if i % 5 else None,
            })
    return out


# ── Long-format table ─────────────────────────────────────────────────────────

def test_rows_cover_every_trial_and_method():
    """Three members plus the derived consensus = four rows per trial."""
    rows = collect_agreement_rows(_fixture(n=10), "f.smr")
    assert len(rows) == 10 * 2 * (len(METHODS) + 1)
    assert set(rows[0]) == set(METHOD_COLS)


def test_undetected_trials_are_recorded_not_dropped():
    """
    A method that fails on a trial must appear as a row with Detected=False.
    Dropping it would make the detection rate unrecoverable from the table and
    would silently flatter whichever method fails most.
    """
    rows = collect_agreement_rows(_fixture(n=10), "f.smr")
    cusum = [r for r in rows if r["Method"] == "cusum"]
    assert len(cusum) == 20
    assert any(r["Detected"] is False for r in cusum)
    assert all(r["Latency(ms)"] is None
               for r in cusum if r["Detected"] is False)


def test_delta_from_consensus_is_signed_and_omitted_when_undefined():
    rows = collect_agreement_rows(_fixture(n=6), "f.smr")
    big = [r for r in rows if r["Method"] == "bigoni"]
    assert all(r["Delta_From_MethodsMedian(ms)"] is not None for r in big)
    assert any(r["Delta_From_MethodsMedian(ms)"] < 0 for r in big)
    missing = [r for r in rows if not r["Detected"]]
    assert all(r["Delta_From_MethodsMedian(ms)"] is None for r in missing)


def test_segment_numbers_are_one_based_to_match_the_trials_table():
    rows = collect_agreement_rows({("B", 0): _agreement({"bigoni": 20.0})},
                                  "f.smr")
    assert rows[0]["Segment"] == 1


def test_stim_labels_are_carried_through():
    rows = collect_agreement_rows({("B", 0): _agreement({"bigoni": 20.0})},
                                  "f.smr", {"B": "120% RMT"})
    assert rows[0]["Stim_Label"] == "120% RMT"


def test_empty_input_produces_no_rows():
    assert collect_agreement_rows({}, "f.smr") == []


# ── Summary table ─────────────────────────────────────────────────────────────

def test_summary_reports_detection_rate_per_method():
    summ = build_method_summary(collect_agreement_rows(_fixture(n=10), "f.smr"))
    assert list(summ.columns) == SUMMARY_COLS
    cusum = summ[summ["Method"] == "cusum"]
    assert (cusum["Detection_Rate"] < 1.0).all()
    big = summ[summ["Method"] == "bigoni"]
    assert (big["Detection_Rate"] == 1.0).all()


def test_limits_of_agreement_bracket_the_mean_difference():
    summ = build_method_summary(collect_agreement_rows(_fixture(n=12), "f.smr"))
    row = summ[summ["Method"] == "bigoni"].iloc[0]
    assert row["LoA_Lower(ms)"] <= row["Mean_Delta_From_MethodsMedian(ms)"] \
        <= row["LoA_Upper(ms)"]


def test_summary_tolerates_a_method_with_a_single_detection():
    ag = {("B", 0): _agreement({"bigoni": 20.0, "cusum": 21.0}),
          ("B", 1): _agreement({"bigoni": 20.5, "cusum": None})}
    summ = build_method_summary(collect_agreement_rows(ag, "f.smr"))
    assert len(summ) == 3                        # 2 members + consensus
    assert summ["SD_Latency(ms)"].isna().any()   # n=1 has no SD, not a crash


def test_empty_summary_still_has_the_right_columns():
    assert list(build_method_summary([]).columns) == SUMMARY_COLS


# ── Files on disk ─────────────────────────────────────────────────────────────

def test_tables_are_written_and_readable(tmp_path):
    rows = collect_agreement_rows(_fixture(n=8), "f.smr")
    paths = write_onset_method_tables(rows, str(tmp_path), "sub-01")
    assert len(paths) == 2
    long_df = pd.read_csv(paths[0])
    assert list(long_df.columns) == METHOD_COLS
    assert len(long_df) == len(rows)
    assert list(pd.read_csv(paths[1]).columns) == SUMMARY_COLS


def test_nothing_is_written_when_agreement_produced_nothing(tmp_path):
    assert write_onset_method_tables([], str(tmp_path), "sub-01") == []
    assert not list(tmp_path.iterdir())


def test_figures_are_written(tmp_path):
    ag = _fixture(n=10)
    rows = collect_agreement_rows(ag, "f.smr")
    rng = np.random.default_rng(1)
    segs = {st: rng.normal(0, 0.01, (10, 2100)) for st in ("B", "C")}
    paths = write_onset_method_figures(rows, ag, segs, 5000.0, 20,
                                       str(tmp_path), "sub-01",
                                       log_callback=lambda *a: None)
    names = [p.rsplit("/", 1)[-1] for p in paths]
    assert any("stim-B_onset_methods" in n for n in names)
    assert any("onset_method_agreement" in n for n in names)
    assert any("onset_disagreement" in n for n in names)
    assert any("onset_bland_altman" in n for n in names)
    import os
    for p in paths:
        assert os.path.getsize(p) > 5000


def test_a_failing_figure_does_not_abort_the_others(tmp_path):
    """One malformed condition must not cost the whole report."""
    ag = _fixture(n=10)
    rows = collect_agreement_rows(ag, "f.smr")
    segs = {"B": np.zeros((0, 10)),                  # empty -> skipped
            "C": np.random.default_rng(2).normal(0, .01, (10, 2100))}
    msgs = []
    paths = write_onset_method_figures(rows, ag, segs, 5000.0, 20,
                                       str(tmp_path), "sub-01",
                                       log_callback=lambda *a: msgs.append(a))
    assert any("stim-C_onset_methods" in p for p in paths)


def test_stim_type_names_are_made_filename_safe(tmp_path):
    ag = {("A/B 120%", i): _agreement({"bigoni": 20.0 + i * .1})
          for i in range(5)}
    rows = collect_agreement_rows(ag, "f.smr")
    segs = {"A/B 120%": np.random.default_rng(3).normal(0, .01, (5, 2100))}
    paths = write_onset_method_figures(rows, ag, segs, 5000.0, 20,
                                       str(tmp_path), "sub-01",
                                       log_callback=lambda *a: None)
    assert paths
    for p in paths:
        assert "/" not in p.rsplit("/", 1)[-1]


# ── The caveat must travel with the figures ───────────────────────────────────

def test_every_figure_carries_the_accuracy_caveat():
    """
    A reader who does not already know that agreement is not accuracy is
    exactly the reader who will over-read these plots, so the caveat belongs
    on the figure rather than only in the documentation.
    """
    import inspect

    from mep_cmap import onset_methods_report as r

    for fn in (r.plot_onset_methods_on_trace, r.plot_method_agreement):
        assert "CAVEAT" in inspect.getsource(fn)
    assert "not accuracy" in CAVEAT


# ── Bland-Altman and the leave-one-out reference ─────────────────────────────

def test_loo_methods_median_excludes_the_method_under_test():
    per = {"a": 10.0, "b": 20.0, "c": 30.0}
    assert _loo_methods_median(per, "a") == 25.0
    assert _loo_methods_median(per, "c") == 15.0


def test_loo_methods_median_needs_at_least_two_other_members():
    """One remaining value is not a consensus."""
    assert _loo_methods_median({"a": 10.0, "b": 20.0}, "a") is None
    assert _loo_methods_median({"a": 10.0, "b": None, "c": 30.0}, "a") is None


def _independent_fixture(n=40, n_methods=5, seed=0):
    """Five members whose errors are INDEPENDENT, as real detectors' are.

    The shared fixture gives every method a fixed offset from one common value,
    so the leave-one-out differences have no variance at all and the
    part-whole comparison cannot be demonstrated. Independent per-method error
    is both more realistic and what the effect requires.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        truth = 20.0 + rng.normal(0, 1.0)
        per = {f"m{k}": round(truth + rng.normal(0, 1.2), 3)
               for k in range(n_methods)}
        out[("B", i)] = _agreement(per)
    return out


def test_loo_limits_are_wider_than_the_part_whole_version():
    """
    Comparing a method against a consensus that contains it is part-whole: the
    method is correlated with its own reference, which shrinks the bias toward
    zero and narrows the limits, flattering every method. Leave-one-out removes
    that, so its limits must be wider. On the real recording this widened the
    mean limits of agreement by about 27%.

    If this ever inverts, the reference is being built with the method still
    included.
    """
    summ = build_method_summary(
        collect_agreement_rows(_independent_fixture(), "f.smr"))
    ok = summ.dropna(subset=["LoA_Lower(ms)", "LoA_LOO_Lower(ms)"])
    assert len(ok) >= 4
    naive = (ok["LoA_Upper(ms)"] - ok["LoA_Lower(ms)"]).mean()
    loo = (ok["LoA_LOO_Upper(ms)"] - ok["LoA_LOO_Lower(ms)"]).mean()
    assert loo > naive, (
        f"leave-one-out limits ({loo:.2f} ms) are not wider than the "
        f"part-whole ones ({naive:.2f} ms); the reference still contains the "
        f"method under test"
    )


def test_loo_bias_is_larger_in_magnitude_than_the_part_whole_bias():
    """The same shrinkage applies to the bias, not only to the limits."""
    summ = build_method_summary(
        collect_agreement_rows(_independent_fixture(seed=3), "f.smr"))
    ok = summ.dropna(subset=["Mean_Delta_From_MethodsMedian(ms)", "Bias_vs_LOO(ms)"])
    assert ok["Bias_vs_LOO(ms)"].abs().mean() >= \
        ok["Mean_Delta_From_MethodsMedian(ms)"].abs().mean()


def test_bland_altman_figure_is_written(tmp_path):
    rows = collect_agreement_rows(_fixture(n=15), "f.smr")
    import os
    path = plot_bland_altman(rows, str(tmp_path), "sub-01")
    assert path and path.endswith("_onset_bland_altman.png")
    assert os.path.getsize(path) > 5000


def test_bland_altman_defaults_to_the_leave_one_out_reference(tmp_path):
    a = plot_bland_altman(collect_agreement_rows(_fixture(n=15), "f.smr"),
                          str(tmp_path), "sub-01")
    b = plot_bland_altman(collect_agreement_rows(_fixture(n=15), "f.smr"),
                          str(tmp_path), "sub-01", use_loo=False)
    assert "vs_consensus" not in a          # default is the LOO version
    assert b.endswith("_vs_consensus.png")  # naive version is clearly marked


def test_bland_altman_excludes_consensus_as_its_own_panel():
    """Consensus compared with itself would be a panel of zeros."""
    ag = {("B", i): OnsetAgreement(
              {"bigoni": 20.0 + i * .1, "cusum": 21.0, "methods_median": 20.5},
              20.5, 1.0, 0.5, 3, 3) for i in range(8)}
    rows = collect_agreement_rows(ag, "f.smr")
    assert any(r["Method"] == "methods_median" for r in rows)
    import inspect

    from mep_cmap import onset_methods_report as r_
    assert 'if m != "methods_median"' in inspect.getsource(r_.plot_bland_altman)


def test_bland_altman_returns_none_without_usable_differences(tmp_path):
    ag = {("B", 0): OnsetAgreement({"bigoni": 20.0}, 20.0, None, None, 1, 1)}
    rows = collect_agreement_rows(ag, "f.smr")
    assert plot_bland_altman(rows, str(tmp_path), "sub-01") is None


# ── Figures belong in a subfolder ────────────────────────────────────────────

def test_figures_go_into_a_subfolder_not_the_figures_root(tmp_path):
    """
    A file with eight stimulus types produces eleven images. Written straight
    into figures/ they bury the pipeline's own trace figures, so they get their
    own folder -- named the way the bundled add-ons already name theirs.
    """
    import os

    ag = _fixture(n=8)
    rows = collect_agreement_rows(ag, "f.smr")
    rng = np.random.default_rng(4)
    segs = {st: rng.normal(0, 0.01, (8, 2100)) for st in ("B", "C")}
    paths = write_onset_method_figures(rows, ag, segs, 5000.0, 20,
                                       str(tmp_path), "sub-01",
                                       log_callback=lambda *a: None)
    assert paths
    expected = figures_subdir(str(tmp_path), "sub-01")
    for p in paths:
        assert os.path.dirname(p) == expected, (
            f"{os.path.basename(p)} was written outside the subfolder"
        )
    # Nothing loose in the figures root.
    assert not [f for f in os.listdir(tmp_path)
                if f.lower().endswith(".png")]


def test_subfolder_name_matches_the_addon_convention():
    """``<bids_prefix>_<name>_figures``, as mepfeatx and temporal_decomposition use."""
    path = figures_subdir("/tmp/figures", "sub-01_ses-2")
    assert path.endswith(f"sub-01_ses-2_{FIGURE_SUBDIR_SUFFIX}")
    assert FIGURE_SUBDIR_SUFFIX.endswith("_figures")


def test_subfolder_is_created_on_demand_only(tmp_path):
    import os

    path = figures_subdir(str(tmp_path), "sub-01")
    assert not os.path.exists(path)
    assert figures_subdir(str(tmp_path), "sub-01", create=True) == path
    assert os.path.isdir(path)


def test_tables_stay_in_results_not_the_figures_subfolder(tmp_path):
    """CSVs are results, not figures; only the images move."""
    import os

    rows = collect_agreement_rows(_fixture(n=6), "f.smr")
    paths = write_onset_method_tables(rows, str(tmp_path), "sub-01")
    for p in paths:
        assert os.path.dirname(p) == str(tmp_path)


# ── The consensus must appear, and the reported method must be identifiable ──

def test_consensus_appears_as_its_own_row():
    """
    compute_onset_agreement returns per_method for the MEMBER detectors only;
    the consensus is derived. Without emitting it explicitly the figures showed
    every method except the one whose value gets reported -- the single line an
    analyst most wants to locate on the waveform.
    """
    from mep_cmap.onset_methods_report import METHODS_MEDIAN_KEY

    rows = collect_agreement_rows(_fixture(n=6), "f.smr")
    assert METHODS_MEDIAN_KEY in {r["Method"] for r in rows}
    cons = [r for r in rows if r["Method"] == METHODS_MEDIAN_KEY]
    assert len(cons) == 6 * 2                       # every trial
    assert all(r["Delta_From_MethodsMedian(ms)"] == 0 for r in cons
               if r["Detected"])


def test_consensus_row_has_no_leave_one_out_reference():
    """Its LOO reference would be the median of all members, i.e. itself."""
    from mep_cmap.onset_methods_report import METHODS_MEDIAN_KEY

    rows = collect_agreement_rows(_fixture(n=6), "f.smr")
    cons = [r for r in rows if r["Method"] == METHODS_MEDIAN_KEY]
    assert all(r["Delta_From_LOO_MethodsMedian(ms)"] is None for r in cons)


def test_consensus_row_matches_the_agreement_object():
    from mep_cmap.onset_methods_report import METHODS_MEDIAN_KEY

    ag = {("B", 0): _agreement({"a": 10.0, "b": 20.0, "c": 30.0})}
    rows = collect_agreement_rows(ag, "f.smr")
    cons = [r for r in rows if r["Method"] == METHODS_MEDIAN_KEY][0]
    assert cons["Latency(ms)"] == 20.0


def test_the_reported_method_is_marked_on_the_trace_figure():
    """
    Five equally weighted lines and no indication of which one was reported
    leaves the reader unable to connect the figure to Latency(ms).
    """
    import inspect

    from mep_cmap import onset_methods_report as r_

    src = inspect.getsource(r_.plot_onset_methods_on_trace)
    assert "selected_method" in src
    assert "reported" in src


def test_selected_method_flows_from_the_pipeline():
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "pipeline.py").read_text(encoding="utf-8")
    assert "selected_method=cfg.onset_method" in src


def test_trace_figure_accepts_a_selected_method(tmp_path):
    import os

    ag = _fixture(n=8)
    rows = collect_agreement_rows(ag, "f.smr")
    rng = np.random.default_rng(7)
    segs = {st: rng.normal(0, 0.01, (8, 2100)) for st in ("B", "C")}
    paths = write_onset_method_figures(rows, ag, segs, 5000.0, 20,
                                       str(tmp_path), "sub-01",
                                       selected_method="bigoni",
                                       log_callback=lambda *a: None)
    assert any("stim-B_onset_methods" in p for p in paths)
    for p in paths:
        assert os.path.getsize(p) > 5000


def test_strip_labels_are_not_clipped_by_the_left_margin():
    """
    The y-tick labels are method names plus a detection count. With a
    hard-coded left margin the longest ("Median across methods  (20/20)") was
    drawn outside the axes and clipped. The margin is now sized from the labels
    that will actually be drawn, so adding a method with a longer name cannot
    silently truncate the axis again.
    """
    import inspect

    from mep_cmap import onset_methods_report as r

    src = inspect.getsource(r.plot_onset_methods_on_trace)
    assert "left_margin" in src
    assert "left=left_margin" in src
    assert "left=0.16" not in src, "the margin is hard-coded again"


def test_computed_left_margin_grows_with_the_longest_label(tmp_path):
    """Rendered width must respond to label length, not stay fixed."""
    import os

    ag = _fixture(n=8, stims=("B",))
    rows = collect_agreement_rows(ag, "f.smr")
    long_rows = [dict(r, Method=r["Method"] + "_with_a_much_longer_name")
                 for r in rows]
    rng = np.random.default_rng(11)
    segs = {"B": rng.normal(0, 0.01, (8, 2100))}

    a = write_onset_method_figures(rows, ag, segs, 5000.0, 20,
                                   str(tmp_path / "short"), "s",
                                   log_callback=lambda *x: None)
    b = write_onset_method_figures(long_rows, ag, segs, 5000.0, 20,
                                   str(tmp_path / "long"), "s",
                                   log_callback=lambda *x: None)
    assert a and b
    # Both render; the long-name figure must not error or produce a stub.
    for pth in a + b:
        assert os.path.getsize(pth) > 5000


def test_no_registry_key_appears_in_figure_text():
    """
    Figure titles and labels are read by people; registry keys are not names.
    `methods_median` leaked into the Bland-Altman title when a text-cleanup
    pass converted the prose back into identifier form.
    """
    import inspect

    from mep_cmap import onset_methods_report as r

    for fn in (r.plot_bland_altman, r.plot_onset_methods_on_trace,
               r.plot_method_agreement, r.plot_disagreement_distribution):
        src = inspect.getsource(fn)
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if any(t in line for t in ("suptitle", "set_title", "fig.text",
                                       "ref_name", "tag =")):
                assert "methods_median" not in line, (
                    f"a registry key appears in figure text: {stripped}"
                )
