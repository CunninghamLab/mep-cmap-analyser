"""
Group summary — built-in second-level (group-level) MEP-CMAP add-on (example).

A minimal, illustrative group-level add-on. It reads the group-level table that
Second Level ▸ Group Analysis builds (group_level_LME_ready.csv) and writes a
per-condition summary (mean, SD, N) of every numeric metric across the group.

Use it as the template for your own group-level add-ons: read
``context.group_table`` (a pandas DataFrame of every included trial across
sessions/participants, with design columns prepended), do whatever group-level
work you like, and write NEW files into ``context.results_dir`` — never modify
the group CSV itself.
"""

import os
import pandas as pd

ADDON_NAME        = "group_summary"
ADDON_DESCRIPTION = "Per-condition summary (mean, SD, N) of each metric across the group (example add-on)"
ADDON_VERSION     = "1.0.0"
ADDON_AUTHOR      = "MEP-CMAP Analyser (built-in example)"
ADDON_SCOPE       = "group_level"


def run(context):
    df   = context.group_table
    log  = context.log
    metrics = list(context.metric_columns)
    if df is None or df.empty:
        log("group_summary: the group table is empty.")
        return []
    if not metrics:
        log("group_summary: no numeric metric columns were found.")
        return []

    # Prefer to summarise by stimulus condition; fall back to the whole group.
    group_key = "StimType" if "StimType" in df.columns else None
    if group_key:
        summary = df.groupby(group_key)[metrics].agg(["mean", "std", "count"])
        # Flatten the (metric, stat) multi-index into clear "metric_stat" columns.
        summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
        summary = summary.reset_index()
    else:
        log("group_summary: no 'StimType' column — summarising the whole group.")
        summary = df[metrics].agg(["mean", "std", "count"]).T
        summary.columns = [f"{c}" for c in summary.columns]
        summary = summary.reset_index().rename(columns={"index": "Metric"})

    out_path = os.path.join(context.results_dir,
                            f"{context.bids_prefix}_group_summary.csv")
    summary.to_csv(out_path, index=False)
    log(f"group_summary: {len(metrics)} metric(s)"
        + (f" \u00d7 {df[group_key].nunique()} condition(s)" if group_key else "")
        + f" \u2192 {os.path.basename(out_path)}")
    return [out_path]
