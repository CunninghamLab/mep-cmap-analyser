"""
Tests for the temporal_decomposition add-on and the Stage 2 add-on sidecar join.

The add-on never re-detects onsets: it reuses the pipeline's onsets from
``<prefix>_trials.csv``, joined to the waveform stack by ``Segment`` (1-based,
indexing ``segs_all``). Most of what can go wrong here is alignment, so the
fixtures build a bundle in exactly the layout ``pipeline_write_segments_bundle``
produces and the tests assert against it.
"""

import os

import numpy as np
import pandas as pd
import pytest

from mep_cmap import addons as A
from mep_cmap.stage2 import _s2_join_addon_sidecars

FS, PRE_MS, POST_MS = 2000.0, 20.0, 100.0
PREFIX, FNAME = "sub-01_ses-01_task-tms", "rec01"
STIMS = ("TMS_120", "TMS_140")
N_TRIALS = 6
ONSET, DUR = 20.0, 22.0


def _time_axis():
    n_pre, n_post = int(PRE_MS * FS / 1000), int(POST_MS * FS / 1000)
    return (np.arange(n_pre + n_post) - n_pre) / FS * 1000.0


def _burst(t_ms, onset, amp, bg, rng):
    x = rng.normal(0.0, bg, t_ms.size)
    m = (t_ms >= onset) & (t_ms <= onset + DUR)
    tau = t_ms[m] - onset
    x[m] += amp * np.sin(2 * np.pi * tau / 5.0) * np.exp(-tau / 9.0)
    return x


@pytest.fixture
def bundle(tmp_path):
    """A synthetic results bundle + per-trial table, with deliberate edge cases:

      * (TMS_120, Segment 1) — onset written as the literal 'Not Detected'
      * (TMS_120, Segment 2) — MEP offset 12 ms after onset, so the clamp bites
      * (TMS_140, Segment 3) — blank cSP offset, as for a resting MEP
    """
    res = tmp_path / "ses-01" / "results"
    res.mkdir(parents=True)
    rng = np.random.default_rng(11)
    t_ms = _time_axis()

    arrays, rows = {}, []
    for s_i, stim in enumerate(STIMS):
        stack = np.vstack([
            _burst(t_ms, ONSET, 1.0 + 0.6 * s_i, 0.02, rng)
            for _ in range(N_TRIALS)
        ])
        arrays[f"wav_{s_i}"] = stack.astype(np.float32)
        arrays[f"out_{s_i}"] = np.zeros(N_TRIALS, bool)
        arrays[f"tidx_{s_i}"] = np.arange(N_TRIALS)
        arrays[f"stime_{s_i}"] = np.arange(N_TRIALS, dtype=float)
        for i in range(N_TRIALS):
            lat, off = ONSET, ONSET + DUR
            if stim == "TMS_120" and i == 0:
                lat = "Not Detected"
            if stim == "TMS_120" and i == 1:
                off = ONSET + 12.0
            if stim == "TMS_140" and i == 2:
                off = ""
            rows.append({"File": FNAME, "StimType": stim, "Segment": i + 1,
                         "PTP(mV)": 1.0, "Latency(ms)": lat,
                         "cSP_MEP_Offset(ms)": off,
                         "Outlier_Decision": "Not flagged", "Manual_Note": ""})

    arrays["manifest_file"] = np.asarray([FNAME] * len(STIMS))
    arrays["manifest_stim"] = np.asarray(list(STIMS))
    arrays["manifest_unit"] = np.asarray(["mV"] * len(STIMS))
    arrays["manifest_fs"] = np.asarray([FS] * len(STIMS), float)
    arrays["manifest_pre_ms"] = np.asarray([PRE_MS] * len(STIMS), float)
    arrays["manifest_post_ms"] = np.asarray([POST_MS] * len(STIMS), float)

    npz = res / f"{PREFIX}_segments.npz"
    np.savez_compressed(npz, **arrays)
    trials = res / f"{PREFIX}_trials.csv"
    pd.DataFrame(rows).to_csv(trials, index=False)
    return {"npz": str(npz), "trials": str(trials), "results": str(res)}


@pytest.fixture
def entry():
    found = A.discover_all("single_file", log=lambda _m: None)
    hits = [f for f in found if f["name"] == "temporal_decomposition"]
    assert hits, "temporal_decomposition add-on was not discovered"
    return hits[0]


def _run(entry, bundle, **cfg):
    cfg.setdefault("td_figures", False)
    ctx = A.load_contexts(bundle["npz"], config=cfg, log=lambda _m: None)[0]
    res = A.run_addon(entry, ctx)
    return res, ctx


def _addon_out(bundle, name):
    """Where an add-on's own output lands: results/add-ons/, not results/.

    Add-ons write to context.addons_dir. The bundle here is written flat, as a
    study analysed before the family layout existed would be, so this also
    covers the mixed case: core files flat, new add-on output foldered.
    """
    return os.path.join(bundle["results"], "add-ons", f"{PREFIX}_{name}.csv")


def _frame(bundle):
    return pd.read_csv(_addon_out(bundle, "temporal_decomposition"))


def _bin_cols(df):
    return [c for c in df.columns
            if c.startswith("TD_Bin_") and c.endswith("(mV*s)")]


# ── contract ─────────────────────────────────────────────────────────────────

def test_addon_satisfies_contract(entry):
    mod = entry["module"]
    for const in ("ADDON_NAME", "ADDON_DESCRIPTION", "ADDON_VERSION",
                  "ADDON_AUTHOR"):
        assert getattr(mod, const, None), f"missing {const}"
    assert entry["scope"] == "single_file"
    assert callable(mod.run)
    keys = {s["key"] for s in entry["settings"]}
    assert {"td_bin_ms", "td_window_ms", "td_boundary_ms",
            "td_clamp_to_offset"} <= keys


def test_writes_only_new_files(entry, bundle):
    before = {p: os.path.getmtime(os.path.join(bundle["results"], p))
              for p in os.listdir(bundle["results"])}
    res, _ = _run(entry, bundle)
    assert res["ok"], res["error"]
    for p, mtime in before.items():
        assert os.path.getmtime(os.path.join(bundle["results"], p)) == mtime, \
            f"add-on modified pre-existing file {p}"


def test_runs_with_empty_config(entry, bundle):
    res, _ = _run(entry, bundle)
    assert res["ok"], res["error"]
    assert _frame(bundle).shape[0] == len(STIMS) * N_TRIALS


# ── alignment and edge cases ─────────────────────────────────────────────────

def test_missing_onset_is_rejected_not_guessed(entry, bundle):
    _run(entry, bundle)
    df = _frame(bundle)
    row = df[(df.StimType == "TMS_120") & (df.Segment == 1)].iloc[0]
    assert not row["TD_Valid"]
    assert "onset" in str(row["TD_Reject_Reason"]).lower()
    assert df.loc[row.name, _bin_cols(df)].isna().all()


def test_clamp_truncates_at_mep_offset(entry, bundle):
    _run(entry, bundle, td_clamp_to_offset=True)
    df = _frame(bundle)
    row = df[(df.StimType == "TMS_120") & (df.Segment == 2)].iloc[0]
    assert row["TD_Clamped_By"] == "MEP offset"
    assert row["TD_Window_End(ms)"] == pytest.approx(12.0)
    assert row["TD_Late_Duration(ms)"] == pytest.approx(4.0)
    assert row["TD_N_Bins_Valid"] == 6


def test_clamp_can_be_disabled(entry, bundle):
    _run(entry, bundle, td_clamp_to_offset=False)
    df = _frame(bundle)
    row = df[(df.StimType == "TMS_120") & (df.Segment == 2)].iloc[0]
    assert row["TD_Window_End(ms)"] == pytest.approx(20.0)
    assert row["TD_N_Bins_Valid"] == 10


def test_blank_mep_offset_leaves_window_intact(entry, bundle):
    _run(entry, bundle)
    df = _frame(bundle)
    row = df[(df.StimType == "TMS_140") & (df.Segment == 3)].iloc[0]
    assert row["TD_Window_End(ms)"] == pytest.approx(20.0)
    assert pd.isna(row["TD_Clamped_By"]) or row["TD_Clamped_By"] == ""


# ── arithmetic ───────────────────────────────────────────────────────────────

def test_bins_sum_to_total_and_phases_partition_it(entry, bundle):
    _run(entry, bundle)
    df = _frame(bundle)
    ok = df[df.TD_Valid == True]                      # noqa: E712
    assert len(ok) == len(STIMS) * N_TRIALS - 1
    assert np.allclose(df.loc[ok.index, _bin_cols(df)].sum(axis=1),
                       ok["TD_Total_Area(mV*s)"], atol=1e-12)
    assert np.allclose(ok["TD_Early_Area(mV*s)"] + ok["TD_Late_Area(mV*s)"],
                       ok["TD_Total_Area(mV*s)"], atol=1e-12)
    assert ok["TD_Early_Fraction"].between(0.0, 1.0).all()


def test_area_matches_independent_trapezoid(entry, bundle):
    _run(entry, bundle)
    _, ctx = _run(entry, bundle)
    df = _frame(bundle)
    row = df[(df.StimType == "TMS_140") & (df.Segment == 1)].iloc[0]

    seg = np.asarray(ctx.segments["TMS_140"], dtype=float)[0]
    t_ms = np.asarray(ctx.time_ms, dtype=float)
    on = float(row["TD_Onset(ms)"])
    base = float(np.mean(np.abs(seg[t_ms <= -2.0])))

    m = (t_ms > on) & (t_ms < on + 8.0)
    nodes = np.concatenate(([on], t_ms[m], [on + 8.0]))
    vals = np.interp(nodes, t_ms, np.abs(seg))
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    expected = float(trapz(vals, nodes / 1000.0)) - base * 8.0 / 1000.0

    assert row["TD_Early_Area(mV*s)"] == pytest.approx(expected, abs=1e-9)


def test_baseline_correction_reduces_area(entry, bundle):
    _run(entry, bundle, td_baseline_correct=False)
    raw = _frame(bundle)["TD_Total_Area(mV*s)"].sum()
    _run(entry, bundle, td_baseline_correct=True)
    corrected = _frame(bundle)["TD_Total_Area(mV*s)"].sum()
    assert corrected < raw


@pytest.mark.parametrize("requested,expected", [
    (7.0, 8.0),    # tie between 6 and 8 must round UP, never shorten early
    (8.0, 8.0),
    (6.0, 6.0),
    (9.0, 10.0),
    (0.1, 2.0),    # clamped to at least one bin
    (19.9, 18.0),  # clamped so the late phase is never empty
])
def test_boundary_snaps_to_bin_edge(entry, bundle, requested, expected):
    _run(entry, bundle, td_boundary_ms=requested, td_bin_ms=2.0,
         td_window_ms=20.0)
    df = _frame(bundle)
    assert df["TD_Boundary(ms)"].dropna().unique().tolist() == [expected]


def test_boundary_outside_window_fails_cleanly(entry, bundle):
    res, _ = _run(entry, bundle, td_boundary_ms=25.0, td_window_ms=20.0)
    assert not res["ok"]
    assert "boundary" in res["error"].lower()


def test_missing_trials_table_fails_loudly(entry, bundle):
    os.remove(bundle["trials"])
    res, _ = _run(entry, bundle)
    assert not res["ok"]
    assert "onset" in res["error"].lower()


# ── Stage 2 sidecar join ─────────────────────────────────────────────────────

def test_sidecar_join_is_additive_and_nondestructive(entry, bundle):
    _run(entry, bundle)
    core = pd.read_csv(bundle["trials"])
    notes = []
    joined = _s2_join_addon_sidecars(core, bundle["trials"], notes.append)

    assert len(joined) == len(core)
    assert joined[core.columns].equals(core), "core columns were altered"
    assert list(joined[core.columns].dtypes) == list(core.dtypes)
    assert any(c.startswith("TD_") for c in joined.columns)
    assert notes and "temporal_decomposition" in notes[0]
    # the literal 'Not Detected' sentinel must survive the join
    assert "Not Detected" in joined["Latency(ms)"].astype(str).values


def test_sidecar_join_aligns_by_segment_not_row_order(entry, bundle):
    _run(entry, bundle)
    core = pd.read_csv(bundle["trials"])
    shuffled = core.sample(frac=1.0, random_state=3).reset_index(drop=True)
    joined = _s2_join_addon_sidecars(shuffled, bundle["trials"], lambda _m: None)
    side = pd.read_csv(_addon_out(bundle, "temporal_decomposition"))
    ref = side.set_index(["StimType", "Segment"])["TD_Early_Area(mV*s)"]
    for _, r in joined.iterrows():
        want = ref.loc[(r["StimType"], r["Segment"])]
        got = r["TD_Early_Area(mV*s)"]
        assert (pd.isna(want) and pd.isna(got)) or want == pytest.approx(got)


def test_sidecar_join_ignores_core_outputs(entry, bundle):
    _run(entry, bundle)
    pd.DataFrame({"StimType": ["TMS_120"], "Segment": [1],
                  "Mean_PTP(mV)": [9.9]}).to_csv(
        os.path.join(bundle["results"], f"{PREFIX}_summary.csv"), index=False)
    joined = _s2_join_addon_sidecars(pd.read_csv(bundle["trials"]),
                                     bundle["trials"], lambda _m: None)
    assert "Mean_PTP(mV)" not in joined.columns


def test_sidecar_join_skips_tables_without_join_keys(entry, bundle):
    # MEPFeatX emits (File, StimType, Trial) — no Segment, so it cannot be
    # joined per trial and must be left alone rather than merged incorrectly.
    pd.DataFrame({"File": [PREFIX], "StimType": ["TMS_120"], "Trial": [0],
                  "nTurns": [3]}).to_csv(
        os.path.join(bundle["results"], f"{PREFIX}_mepfeatx.csv"), index=False)
    joined = _s2_join_addon_sidecars(pd.read_csv(bundle["trials"]),
                                     bundle["trials"], lambda _m: None)
    assert "nTurns" not in joined.columns


def test_sidecar_join_does_not_fan_out_on_duplicates(entry, bundle):
    _run(entry, bundle)
    side_path = _addon_out(bundle, "temporal_decomposition")
    side = pd.read_csv(side_path)
    pd.concat([side, side]).to_csv(side_path, index=False)
    core = pd.read_csv(bundle["trials"])
    joined = _s2_join_addon_sidecars(core, bundle["trials"], lambda _m: None)
    assert len(joined) == len(core)


def test_sidecar_join_is_idempotent(entry, bundle):
    _run(entry, bundle)
    core = pd.read_csv(bundle["trials"])
    once = _s2_join_addon_sidecars(core, bundle["trials"], lambda _m: None)
    twice = _s2_join_addon_sidecars(once, bundle["trials"], lambda _m: None)
    assert list(once.columns) == list(twice.columns)


def test_sidecar_join_namespaces_colliding_columns(entry, bundle):
    # MEPFeatX emits its own 'Latency(ms)' from a different algorithm than the
    # core pipeline's. It must be kept under a namespaced name, never dropped
    # and never allowed to overwrite the core column.
    pd.DataFrame({"File": [FNAME] * 2, "StimType": ["TMS_120"] * 2,
                  "Segment": [1, 2], "Latency(ms)": [21.5, 22.5],
                  "nTurns": [3, 4]}).to_csv(
        os.path.join(bundle["results"], f"{PREFIX}_mepfeatx.csv"), index=False)
    core = pd.read_csv(bundle["trials"])
    notes = []
    joined = _s2_join_addon_sidecars(core, bundle["trials"], notes.append)

    assert "mepfeatx_Latency(ms)" in joined.columns
    assert "nTurns" in joined.columns
    assert joined["Latency(ms)"].equals(core["Latency(ms)"])
    row = joined[(joined.StimType == "TMS_120") & (joined.Segment == 1)].iloc[0]
    assert row["mepfeatx_Latency(ms)"] == pytest.approx(21.5)
    assert any("namespaced" in n for n in notes)


def test_sidecar_join_skips_when_keys_match_nothing(entry, bundle):
    # A sidecar carrying the BIDS prefix in File instead of the source file name
    # matches no trials. It must be skipped loudly, not appended as all-NaN.
    pd.DataFrame({"File": [PREFIX], "StimType": ["TMS_120"], "Segment": [1],
                  "someMetric": [1.0]}).to_csv(
        os.path.join(bundle["results"], f"{PREFIX}_bogus.csv"), index=False)
    notes = []
    joined = _s2_join_addon_sidecars(pd.read_csv(bundle["trials"]),
                                     bundle["trials"], notes.append)
    assert "someMetric" not in joined.columns
    assert any("matched no trials" in n for n in notes)


def test_mepfeatx_emits_joinable_keys(bundle):
    found = A.discover_all("single_file", log=lambda _m: None)
    mfx = [f for f in found if f["name"] == "mepfeatx"]
    if not mfx:
        pytest.skip("mepfeatx add-on not present")
    ctx = A.load_contexts(bundle["npz"],
                          config={"mepfeatx_plot": False}, log=lambda _m: None)[0]
    res = A.run_addon(mfx[0], ctx)
    assert res["ok"], res["error"]

    side = pd.read_csv(_addon_out(bundle, "mepfeatx"))
    core = pd.read_csv(bundle["trials"])
    assert {"File", "StimType", "Segment"}.issubset(side.columns)
    assert "Trial" in side.columns                      # 0-based index retained
    assert (side["Segment"] == side["Trial"] + 1).all()
    assert set(side["File"].unique()) == set(core["File"].unique())

    joined = _s2_join_addon_sidecars(core, bundle["trials"], lambda _m: None)
    assert len(joined) == len(core)
    assert any(c.endswith("nTurns") for c in joined.columns)
    assert joined[core.columns].equals(core)


def test_late_at_baseline_is_flagged(entry, bundle):
    _run(entry, bundle)
    df = _frame(bundle)
    assert "TD_Late_At_Baseline" in df.columns
    assert "TD_Early_At_Baseline" in df.columns
    ok = df[df.TD_Valid == True]                        # noqa: E712
    flagged = ok["TD_Late_At_Baseline"].astype(bool)
    assert (ok.loc[~flagged, "TD_Early_Fraction"] <= 1.0).all()


def test_baseline_correction_is_reversible_from_outputs(entry, bundle):
    """Uncorrected areas must be recoverable, so no raw duplicate columns are
    needed and a reviewer can inspect them without a re-run."""
    _run(entry, bundle, td_baseline_correct=False)
    raw = _frame(bundle).set_index(["StimType", "Segment"])
    _run(entry, bundle, td_baseline_correct=True)
    cor = _frame(bundle).set_index(["StimType", "Segment"])

    ok = cor[cor.TD_Valid == True]                       # noqa: E712
    rebuilt = (ok["TD_Early_Area(mV*s)"]
               + ok["TD_Baseline_Amp(mV)"] * ok["TD_Early_Duration(ms)"] / 1000.0)
    assert np.allclose(rebuilt, raw.loc[ok.index, "TD_Early_Area(mV*s)"], atol=1e-12)


def test_figures_go_into_their_own_subfolder(entry, bundle):
    # Matches the MEPFeatX convention: figures/<prefix>_<addon>_figures/
    res, _ = _run(entry, bundle, td_figures=True)
    assert res["ok"], res["error"]
    fig_dir = os.path.join(os.path.dirname(bundle["results"]), "figures",
                           f"{PREFIX}_temporal_decomposition_figures")
    assert os.path.isdir(fig_dir)
    pngs = [f for f in os.listdir(fig_dir) if f.endswith(".png")]
    assert len(pngs) == len(STIMS)
    # nothing dumped loose in figures/
    loose = [f for f in os.listdir(os.path.join(
        os.path.dirname(bundle["results"]), "figures")) if f.endswith(".png")]
    assert not loose
    assert all(p in res["paths"] for p in
               [os.path.join(fig_dir, f) for f in pngs])


def test_stim_type_filter_excludes_conditions(entry, bundle):
    # The early/late interpretation is TMS-specific; peripheral-stimulation
    # conditions (M-wave, H-reflex) must be excludable.
    res, _ = _run(entry, bundle, td_stim_types="TMS_140")
    assert res["ok"], res["error"]
    df = _frame(bundle)
    assert set(df["StimType"].unique()) == {"TMS_140"}
    assert len(df) == N_TRIALS


def test_stim_type_filter_accepts_spaces_and_blank(entry, bundle):
    _run(entry, bundle, td_stim_types=" TMS_120 , TMS_140 ")
    assert set(_frame(bundle)["StimType"].unique()) == set(STIMS)
    _run(entry, bundle, td_stim_types="")
    assert set(_frame(bundle)["StimType"].unique()) == set(STIMS)


def test_figure_survives_trials_with_no_onset(entry, bundle):
    # (TMS_120, Segment 1) has no onset; the figure must still render from the
    # remaining trials rather than failing on the None.
    res, _ = _run(entry, bundle, td_figures=True)
    assert res["ok"], res["error"]
    assert any("TMS_120" in p and p.endswith(".png") for p in res["paths"])
