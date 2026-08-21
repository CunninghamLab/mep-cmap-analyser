"""
MEP-CMAP Analyser — ADD-ON TEMPLATE
===================================

Copy this file, rename it, and edit `run()` to build your own analysis add-on.
Drop the finished file into your add-ons folder (set the folder path in
Preferences → Add-ons), open the "Add-ons" tab, and it will appear as a
runnable item. No core files need to be edited — ever.

Contract (all four metadata constants + a `run(context)` function are required):

    ADDON_NAME         short machine-ish name; used to name your output files
    ADDON_DESCRIPTION  one line shown next to the Run button
    ADDON_VERSION      your version string
    ADDON_AUTHOR       you

    run(context) -> list[str]
        Do your analysis and RETURN the list of file paths you wrote.

Golden rules:
  • WRITE NEW FILES ONLY. Never open the core outputs (`*_trials.csv`,
    `*_summary.csv`, figures) for writing. Name your outputs with the prefix:
        f"{context.bids_prefix}_{ADDON_NAME}.csv"
  • Read everything you need from `context`; don't reach into core globals.
  • Raise on real errors — the loader catches and reports them; it will not
    crash the app.

The `context` handed to run():

    context.trials       pandas.DataFrame  — the per-trial table (_trials.csv):
                                             measurements + marker columns.
    context.segments     dict {stim_type: ndarray[n_trials, n_samples]} —
                                             the per-trial waveforms.
    context.fs           float             — sampling rate (Hz). VARIES BY FORMAT
                                             (e.g. 5000 vs 2000) — never assume it.
    context.unit         str | None        — native amplitude unit ('mV', 'µV', or
                                             None if unknown). VARIES BY FORMAT —
                                             use it to label amplitude outputs.
    context.time_ms      1-D ndarray       — time axis in ms relative to stim,
                                             length == n_samples (0 ms = stimulus).

Format-independence: your add-on never sees the source format. Every reader
(Spike2, LabChart, EDF/BDF, BrainVision, AcqKnowledge, CFWB, generic TSV,
Brainsight, …) is normalised to arrays + fs + unit before the bundle is written,
so the same add-on runs on all of them. The only things that vary are `fs` and
`unit` — always take them from `context`, never hardcode a rate or a unit.
    context.config       dict              — analysis settings (prestim_ms,
                                             ptp_start, ptp_end, …).
    context.addons_dir   str               — write your output files here.
    context.results_dir  str               — the results root, to READ what
                                             the pipeline wrote. Not an
                                             output location.
    context.bids_prefix  str               — file-name prefix to use.
    context.log          callable(str)     — write a line to the GUI log.
"""

import os
import numpy as np
import pandas as pd

# ── Required metadata ────────────────────────────────────────────────────────
ADDON_NAME        = "rectified_area"          # -> writes <prefix>_rectified_area.csv
ADDON_DESCRIPTION = "Rectified area under each MEP over the analysis window (example add-on)"
ADDON_VERSION     = "1.0.0"
ADDON_AUTHOR      = "MEP-CMAP Analyser (built-in example)"
ADDON_SCOPE       = "single_file"


def run(context):
    """Compute one simple feature per trial and write it to a new CSV.

    This example measures the rectified area (∫|EMG| dt) of each trial over the
    peak-to-peak analysis window defined in the config. Replace the body with
    your own computation; keep the read-from-context / write-new-file shape.
    """
    fs         = context.fs
    _t_ms_for  = getattr(context, "time_ms_for", None)
    unit       = context.unit or "a.u."      # varies by format; may be None
    cfg        = context.config
    log        = context.log

    # Analysis window (ms relative to stim) — reuse the tool's PTP window.
    win_start_ms = float(cfg.get("ptp_start", 10.0))
    win_end_ms   = float(cfg.get("ptp_end",   50.0))
    dt           = 1.0 / fs

    # The window mask is per stimulus type, since each may be epoched over a
    # different span; built inside the loop below.

    # The join keys Stage 2 needs. Without File and Segment this file is not
    # matchable to the core trials and the group merge skips it, so the example
    # every third-party add-on is copied from must emit them.
    #
    # File is constant for one recording, so any trial row carries it; fall back
    # to the prefix when the add-on runs without a per-trial table beside the
    # bundle. Segment is 1-based and indexes the bundle, matching the core
    # tables; Trial is kept alongside as the 0-based index.
    file_name = context.bids_prefix
    _tr = getattr(context, "trials", None)
    if _tr is not None and getattr(_tr, "empty", True) is False \
            and "File" in _tr.columns and len(_tr["File"]):
        file_name = _tr["File"].iloc[0]

    rows = []
    for stim_type, stack in context.segments.items():
        time_ms = np.asarray(_t_ms_for(stim_type) if _t_ms_for
                             else context.time_ms)
        in_window = (time_ms >= win_start_ms) & (time_ms <= win_end_ms)
        if not in_window.any():
            raise ValueError(
                f"{ADDON_NAME}: analysis window {win_start_ms}-{win_end_ms} ms "
                f"falls outside {stim_type}'s time axis "
                f"({time_ms[0]:.1f} to {time_ms[-1]:.1f} ms)."
            )
        stack = np.asarray(stack, dtype=float)          # [n_trials, n_samples]
        for trial_idx, trace in enumerate(stack):
            windowed = np.abs(trace[in_window])
            area = float(np.trapezoid(windowed, dx=dt)   # NumPy 2.x name
                         if hasattr(np, "trapezoid")
                         else np.trapz(windowed, dx=dt))
            rows.append({
                "File":                       file_name,
                "StimType":                   stim_type,
                "Segment":                    trial_idx + 1,
                "Trial":                      trial_idx,
                f"Rectified_Area({unit}*s)":  round(area, 6),   # unit from context
                # Two numeric columns rather than a "10-50" string: a value like
                # "10-50" is silently turned into a date ("Oct-50") when opened in
                # Excel. Keep add-on outputs numeric / non-date-ambiguous.
                "Window_start(ms)":           win_start_ms,
                "Window_end(ms)":             win_end_ms,
            })

    if not rows:
        log(f"{ADDON_NAME}: no trials found in the results bundle — nothing written.")
        return []

    # WRITE A NEW FILE ONLY — never touch the core outputs.
    out_path = os.path.join(context.addons_dir,
                            f"{context.bids_prefix}_{ADDON_NAME}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log(f"{ADDON_NAME}: wrote {len(rows)} rows → {os.path.basename(out_path)}")
    return [out_path]
