"""
mep_cmap.preview
~~~~~~~~~~~~~~~~
PreviewDetectionMixin — "Preview detection": try the current detection
settings on chosen trials before committing to a full run.

Why it exists
-------------
Every detection setting in this application is applied by running the whole
pipeline and reading the result. That makes the settings hard to trust: the
analyst chooses an onset method, an amplitude gate and a latency window, waits
for a run, and only then sees whether the markers landed anywhere sensible.
Adjusting one number means paying for another run. With the number of detection
options this tool now offers, that is the difference between a tool the analyst
reasons about and a tool they poke at.

The preview closes that loop. It loads the file, filters it, offers the trials
it found, and opens the ordinary Data Inspector — read-only — on the ones
chosen.

How it stays honest
-------------------
The Inspector is not a viewer; it calls ``detection.dispatch_onset`` itself for
every trial it draws. So the preview does not reimplement detection, it
supplies segments and lets the same detector run. Three things keep what is
previewed identical to what the run will produce:

  settings   ``_snapshot_analysis_params()``, the same snapshot Run Analysis
             takes, including its pre-epoched clamp.
  filtering  ``pipeline_apply_filters`` with a config carrying the fourteen
             fields that function reads, named explicitly. Building that config
             by filtering the snapshot against PipelineConfig field names would
             be shorter and wrong: the snapshot calls two settings `min_amp`
             and `enable_out_review` where the config fields are
             `min_peak_amplitude` and `enable_outlier_review`, so a name filter
             silently substitutes defaults. `test_preview_detection.py` asserts
             the fourteen still match what the filter stage reads.
  events     the channel's configured event sources, passed to
             ``pipeline_load_file`` exactly as the worker passes them. Without
             this the preview reads the file's own markers and the run reads a
             threshold crossing, and the two show different trials.
  epoching   the same segment loop the pipeline uses for the Inspector,
             including the per-stimulus-type event delay. Omitting the delay
             would preview a different epoch from the one the analysis
             measures -- the exact bug that produced markers ~1.6-2.0 ms early
             in condition C.

Choosing trials
---------------
Which trials to look at is the analyst's decision, not this module's. An
earlier version sampled a fixed set automatically, which is a reasonable
default and a poor constraint: someone who suspects trials 40-52 went wrong had
no way to look at them, and a preview that decides what is worth seeing repeats
the problem it exists to solve.

So the trials are offered, with a default already selected: an even spread
across the recording rather than the first n. The opening trials are where the
participant is freshest and where warm-up artefacts live, which makes them the
least informative sample for judging a detection setting; an even spread also
puts the first and last trial of the session on screen, where drift and fatigue
show up first. Trials are numbered as they are in the recording, so one seen
here can be found again after the run.

There is no cap. Trial count barely affects cost -- reading and filtering the
whole recording is the expensive part and happens regardless, cutting segments
is cheap, and the Inspector draws one trial at a time -- so Select all is a
reasonable thing to do on a long file.

What it deliberately does not do
--------------------------------
Nothing is written: no CSVs, no figures, no session autosave, no marker
metadata. The Inspector is opened read-only, so markers are drawn but fixed:
this is a picture of what the configured detector does, and a marker the
analyst could drag would be an invitation to correct the answer by hand in a
window that saves nothing.

Two things cannot be faithful before a run, and the preview says so in the log
rather than hiding them:

  * per-stimulus-type amplitude window anchoring is computed *from* a run's
    median onset, so anchored types preview with the file-wide window;
  * outlier decisions do not exist yet, so every chosen trial is shown.

This mixin assumes the host provides: self.root, self.log(),
self._validate_analysis_setup(), self._snapshot_analysis_params(),
self._open_inspector_preview(), self.label_map / self.color_map, and the Tk
variables the Inspector payload reads.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from .event_sources import EventSource
from .pipeline import (PipelineConfig, pipeline_apply_filters,
                       pipeline_load_file, window_samples)
from .preferences import prefs


#: Re-exported from pipeline, where it sits beside the function that reads it.
#: One list: the preview and the conditions review pane both filter for
#: display, and two copies of this would drift the moment a filter setting was
#: added to one of them.
from .pipeline import FILTER_CFG_FIELDS  # noqa: E402,F401


def select_preview_trials(n_available: int, k: int) -> list:
    """Indices of *k* trials spread evenly across *n_available*.

    Returns true positions in the recording, ascending, without duplicates.
    Endpoints are included when k >= 2 so the first and last trial of the
    session are always among those offered -- drift and fatigue show up there
    first.
    """
    if n_available <= 0 or k <= 0:
        return []
    if k >= n_available:
        return list(range(n_available))
    if k == 1:
        return [n_available // 2]
    step = (n_available - 1) / (k - 1)
    return sorted({int(round(i * step)) for i in range(k)})


def default_tick_count() -> int:
    """How many trials per stimulus type start out selected."""
    try:
        k = int(prefs.preview_trials_per_type)
    except Exception:
        k = 8
    return max(1, k)


class TrialSelectDialog:
    """Modal trial chooser. ``result`` is {stim_type: [index]} or None.

    One list per stimulus type rather than a type dropdown over a shared list:
    a selection made under one type must survive looking at another, and a
    single list would either lose it or hide it. Types are usually few.
    """

    def __init__(self, master, counts, preselect, label_map=None,
                 spread_k=None):
        # `preselect` is what starts selected, which may be a selection carried
        # over from the last preview. `spread_k` is what the Even spread button
        # computes from, freshly, every time it is pressed. Binding that button
        # to `preselect` instead made it re-apply the remembered selection --
        # so after a Select all it appeared to do nothing at all.
        self.result = None
        self._counts = dict(counts)
        self._spread_k = int(spread_k or default_tick_count())
        self._lists = {}

        self.top = tk.Toplevel(master)
        self.top.title("Preview detection – choose trials")
        self.top.transient(master)
        self.top.grab_set()

        head = ttk.Frame(self.top, padding=(12, 10, 12, 4))
        head.pack(fill="x")
        ttk.Label(
            head, justify="left", wraplength=560,
            text=("Trials are numbered as they are in the recording. The "
                  "default is an even spread across the session, so the first "
                  "and last trial are included — that is where drift shows.")
        ).pack(anchor="w")

        body = ttk.Frame(self.top, padding=(12, 4))
        body.pack(fill="both", expand=True)

        for col, stim in enumerate(sorted(self._counts)):
            n = self._counts[stim]
            frame = ttk.Frame(body)
            frame.grid(row=0, column=col, padx=(0, 12), sticky="nsew")
            body.columnconfigure(col, weight=1)
            label = (label_map or {}).get(stim, stim)
            ttk.Label(frame, text=f"{label}  ({n} trial(s))").pack(anchor="w")

            holder = ttk.Frame(frame)
            holder.pack(fill="both", expand=True)
            bar = ttk.Scrollbar(holder, orient="vertical")
            # exportselection=False: without it, selecting in one list clears
            # the others, because Tk hands the X selection to the newest one.
            box = tk.Listbox(holder, selectmode="extended", height=14,
                             exportselection=False, width=14,
                             yscrollcommand=bar.set)
            bar.config(command=box.yview)
            bar.pack(side="right", fill="y")
            box.pack(side="left", fill="both", expand=True)
            for i in range(n):
                box.insert("end", f"Trial {i + 1}")
            box.bind("<<ListboxSelect>>", lambda e: self._refresh_count())
            self._lists[stim] = box

        body.rowconfigure(0, weight=1)

        self._count_var = tk.StringVar()
        foot = ttk.Frame(self.top, padding=(12, 4, 12, 12))
        foot.pack(fill="x")
        ttk.Label(foot, textvariable=self._count_var).pack(side="left")

        ttk.Button(foot, text="Cancel", command=self._cancel).pack(side="right")
        self._ok = ttk.Button(foot, text="Preview", command=self._accept)
        self._ok.pack(side="right", padx=(0, 6))
        ttk.Button(foot, text="None",
                   command=self._select_none).pack(side="right", padx=(0, 18))
        ttk.Button(foot, text="All",
                   command=self._select_all).pack(side="right", padx=(0, 6))
        ttk.Button(foot, text=f"Even spread ({self._spread_k})",
                   command=self._select_spread).pack(
                       side="right", padx=(0, 6))

        self._apply(preselect)
        self.top.bind("<Return>", lambda e: self._accept())
        self.top.bind("<Escape>", lambda e: self._cancel())
        self.top.protocol("WM_DELETE_WINDOW", self._cancel)
        self._ok.focus_set()

    # ── selection helpers ────────────────────────────────────────────────────

    def _apply(self, chosen):
        for stim, box in self._lists.items():
            box.selection_clear(0, "end")
            for i in (chosen or {}).get(stim, ()):
                if 0 <= i < self._counts[stim]:
                    box.selection_set(i)
            box.see(0)
        self._refresh_count()

    def _select_spread(self):
        """Recompute the spread now, rather than reapplying what was passed in."""
        self._apply({s: select_preview_trials(n, self._spread_k)
                     for s, n in self._counts.items()})

    def _select_all(self):
        self._apply({s: range(n) for s, n in self._counts.items()})

    def _select_none(self):
        self._apply({})

    def selection(self) -> dict:
        return {stim: [int(i) for i in box.curselection()]
                for stim, box in self._lists.items()
                if box.curselection()}

    def _refresh_count(self):
        total = sum(len(v) for v in self.selection().values())
        available = sum(self._counts.values())
        note = ""
        if total and total == available and available <= self._spread_k:
            # Not a fault: an even spread of k over fewer than k trials is
            # every trial. Without this the button looks broken on short files.
            note = "  —  fewer trials than the spread size, so this is all of them"
        self._count_var.set(f"{total} of {available} trial(s) selected{note}")
        self._ok.state(["!disabled"] if total else ["disabled"])

    # ── close ────────────────────────────────────────────────────────────────

    def _accept(self):
        chosen = self.selection()
        if not chosen:
            return
        self.result = chosen
        self.top.destroy()

    def _cancel(self):
        self.result = None
        self.top.destroy()


class PreviewDetectionMixin:
    """First Level ▸ Preview detection."""

    def preview_detection_start(self):
        """Called by the *Preview detection* button (GUI thread)."""
        # The preview writes nothing, so an output folder is not a
        # precondition; every other setup rule still is.
        if not self._validate_analysis_setup(require_derivatives=False):
            return
        if getattr(self, "_preview_running", False):
            return

        params = self._snapshot_analysis_params()
        path = params.get("input_path") or ""
        if not path or not os.path.isfile(path):
            messagebox.showwarning(
                "No file selected",
                "Choose a data file before previewing detection.",
                parent=self.root)
            return

        self._preview_running = True
        self.log(f"🔎 Preview detection — reading {os.path.basename(path)}…")

        # Only the read and filter are threaded; they take seconds and would
        # freeze Tk. Choosing trials and cutting them are cheap and belong on
        # the GUI thread with the dialog.
        def work():
            try:
                loaded = self._preview_load(params)
            except Exception as exc:                # noqa: BLE001 — shown below
                self.root.after(0, lambda e=exc: self._preview_failed(e))
                return
            self.root.after(0, lambda d=loaded: self._preview_choose(d))

        threading.Thread(target=work, daemon=True).start()

    # ── worker ───────────────────────────────────────────────────────────────

    def _preview_load(self, params):
        """Read and filter the recording. Runs off the GUI thread."""
        # This channel's event sources, exactly as _analysis_worker builds
        # them. Omitting them made the preview read the file's own markers
        # while the run used the configured threshold -- so the trial chooser
        # offered one set of trials and the analysis measured another, which is
        # the single thing this feature exists to prevent.
        _src_raw = (params.get("event_sources") or {}).get(
            params["channel_idx"]) or []
        _sources = [EventSource.from_dict(_d) for _d in _src_raw]

        emg, time, fs, unit, stim_times = pipeline_load_file(
            params["input_path"], params["channel_idx"],
            params["marker_choice"],
            crop_ranges=params.get("crop_ranges"),
            crop_start=params.get("crop_start"),
            crop_end=params.get("crop_end"),
            sources=_sources,
            channel_names=params.get("channel_names"),
            # Assigned conditions, when there are any. Passed through the same
            # argument the analysis uses, so both compose the two columns into
            # group keys by the same call -- a preview that grouped trials
            # differently would offer a set the run does not analyse.
            event_rows=params.get("event_rows"),
            warn=lambda m: self.log(f"   ⚠️  {m}"))

        cfg = PipelineConfig(**{f: params[f] for f in FILTER_CFG_FIELDS})
        emg = pipeline_apply_filters(emg, fs, cfg)

        # Only stimuli inside the recording can be cut at all, so the trial
        # numbers offered are the ones that can actually be shown.
        usable = {}
        for stim_type in sorted(stim_times):
            keep = [t for t in stim_times[stim_type]
                    if time.min() <= t <= time.max()]
            if keep:
                usable[stim_type] = keep

        return dict(emg=emg, time=time, fs=fs, unit=unit, usable=usable,
                    params=params)

    # ── GUI thread ───────────────────────────────────────────────────────────

    def _preview_failed(self, exc):
        self._preview_running = False
        self.log(f"❌ Preview detection failed: {exc}")
        messagebox.showerror(
            "Preview detection", f"Could not preview this file:\n\n{exc}",
            parent=self.root)

    def _preview_choose(self, loaded):
        """Offer the trials, then cut and show whatever was chosen."""
        self._preview_running = False
        usable = loaded["usable"]
        if not usable:
            self.log("   ⚠️  No stimuli found inside the recording")
            messagebox.showinfo(
                "Preview detection",
                "No stimulus events fall inside this recording, so there is "
                "nothing to preview. Check the selected marker.",
                parent=self.root)
            return

        counts = {s: len(v) for s, v in usable.items()}
        # Reuse the last selection while the analyst is tuning a setting on the
        # same file. Re-picking the same trials to compare two values of one
        # parameter is exactly the friction this feature exists to remove.
        key = (loaded["params"]["input_path"],
               loaded["params"]["channel_idx"],
               loaded["params"]["marker_choice"])
        remembered = None
        if getattr(self, "_preview_last_key", None) == key:
            remembered = getattr(self, "_preview_last_selection", None)
        preselect = remembered or {
            s: select_preview_trials(n, default_tick_count())
            for s, n in counts.items()}

        dlg = TrialSelectDialog(self.root, counts, preselect,
                                dict(getattr(self, "label_map", {}) or {}),
                                spread_k=default_tick_count())
        self.root.wait_window(dlg.top)
        if not dlg.result:
            self.log("   Preview cancelled")
            return
        self._preview_last_key = key
        self._preview_last_selection = dlg.result

        payload = self._preview_cut(loaded, dlg.result)
        if not payload["segments"]:
            self.log("   ⚠️  No complete trials in the current window — "
                     "nothing to preview")
            messagebox.showinfo(
                "Preview detection",
                "None of the chosen trials fit the current pre/post window, "
                "so there is nothing to show. Check the window settings.",
                parent=self.root)
            return
        self._preview_show(payload)

    def _preview_cut(self, loaded, chosen):
        """Cut the chosen trials. Cheap, so it stays on the GUI thread."""
        emg, time = loaded["emg"], loaded["time"]
        fs = loaded["fs"]
        params = loaded["params"]
        prestim_ms = float(params["prestim_ms"])
        post_ms    = float(params["post_ms"])
        # The window is per stimulus type, exactly as the analysis resolves it.
        # Cutting every type to one window here would have the preview offer
        # trials of a length the run will not produce -- and for a type given a
        # longer window, show a response truncated where the analysis measures
        # it whole.
        #
        # Pre stays prestim_ms, matching what the pipeline hands the Inspector:
        # the review deliberately shows a wider lead-in than the analysis
        # window. Only post varies by type.
        # Per-type maps come from THIS CHANNEL's snapshot, with the flat map as
        # a fallback -- the same rule the detection config below already uses.
        #
        # These two read the flat map only. window_map and delay_ms_map are per
        # channel, held in chan_settings, and the flat copy belongs to whichever
        # channel was last harvested. On this study the delay was 17.5 ms in the
        # snapshot and absent from the flat map, so the preview cut its segments
        # with NO delay while the run cut with it: the response sat at ~50 ms
        # against a 28-45 ms latency window, no onset was found on any trial,
        # and the preview reported "No onsets were pre-detected" while the run
        # detected all 21. A warning that says the opposite of what happens is
        # worse than none.
        _pv_setup = (params.get("chan_settings") or {}).get(
            params.get("channel_idx", getattr(self, "channel_idx", 0))) or {}

        def _pv_own(key, default=None):
            return _pv_setup.get(key, params.get(key, default))

        _wincfg = PipelineConfig(pre_ms=float(params["pre_ms"]),
                                 post_ms=post_ms,
                                 window_map=_pv_own("window_map", {}) or {})
        samples_before = int(prestim_ms * fs / 1000)
        delay_map = _pv_own("delay_ms_map", {}) or {}

        # Two sets are cut: the chosen trials, which the Inspector draws, and
        # EVERY trial, which detection runs over.
        #
        # Onset anchoring takes the median onset of a stimulus type and the
        # amplitude window is derived from it, so detecting over eight of
        # twenty trials gives a different median -- and therefore a different
        # window and a different peak-to-peak -- from the run. Cutting all of
        # them costs one pass over an array already in memory, and it is the
        # difference between the preview being close and being right.
        segments, picked, dropped, every = {}, {}, {}, {}
        for stim_type, idxs in chosen.items():
            samples_after = window_samples(_wincfg, stim_type, fs)[1]
            times = loaded["usable"].get(stim_type, [])
            # The event delay MUST be applied here, exactly as the pipeline
            # applies it when building its own inspector segments.
            shift = int(round(float(delay_map.get(stim_type, 0.0)) * fs / 1000.0))
            segs, kept = [], []
            for i in sorted(idxs):
                if i >= len(times):
                    continue
                ix = int(np.argmin(np.abs(time - times[i]))) + shift
                if ix < 0 or ix >= len(emg):
                    continue
                seg = emg[max(0, ix - samples_before): ix + samples_after]
                if len(seg) == samples_before + samples_after:
                    segs.append(seg)
                    kept.append(i)
            if segs:
                segments[stim_type] = segs
                picked[stim_type] = kept
            dropped[stim_type] = len(idxs) - len(segs)

            _all = []
            for i in range(len(times)):
                ix = int(np.argmin(np.abs(time - times[i]))) + shift
                if ix < 0 or ix >= len(emg):
                    continue
                seg = emg[max(0, ix - samples_before): ix + samples_after]
                if len(seg) == samples_before + samples_after:
                    _all.append(seg)
            if _all:
                every[stim_type] = _all

        return dict(segments=segments, picked=picked, dropped=dropped,
                    every=every,
                    # Carried so the "no onsets" report can name the delay it
                    # actually cut with, which is the commonest reason the
                    # preview and the run disagree.
                    delay_map=dict(delay_map),
                    fs=fs, unit=loaded["unit"], prestim_ms=prestim_ms,
                    post_ms=post_ms)

    def _preview_show(self, payload):
        for stim_type, kept in payload["picked"].items():
            self.log(f"   • {stim_type}: {len(kept)} trial(s) — "
                     + ", ".join(str(i + 1) for i in kept))
            if payload["dropped"].get(stim_type):
                self.log(f"     ({payload['dropped'][stim_type]} chosen "
                         f"trial(s) did not fit the window)")

        # Detect with the analysis's own detector, and seed the Inspector with
        # the result.
        #
        # The Inspector re-detects whatever it is not given, one trial at a
        # time. That is a different computation from the analysis: onset
        # anchoring takes the MEDIAN onset of a stimulus type, and the search
        # window is widened from the latency profile across the sample -- work
        # that needs every trial. Given none of it, the Inspector fell back to
        # the file-wide window and reported the window edge as a latency, so
        # the preview disagreed with the run it was previewing.
        #
        # pipeline_detect_onsets is the sole source of automatic onset values
        # in the analysis. Calling it here is what makes this a rehearsal
        # rather than a second opinion.
        _seed, _ptp_ms = {}, {}
        try:
            from .pipeline import (PipelineConfig, pipeline_detect_onsets,
                                   ptp_window_for_stim_type,
                                   window_samples)
            # Built the way the RUN builds it, for this channel.
            #
            # Filtering the snapshot by PipelineConfig field names looked
            # tidy and was wrong twice over. The snapshot names several
            # settings differently -- min_amp for min_peak_amplitude -- so a
            # name filter silently substitutes defaults; and 38 of the 49
            # detection fields are not top-level keys at all, because the run
            # passes them as one detection_params mapping.
            #
            # latency_map is read from THIS CHANNEL's snapshot, not from the
            # flat map. The flat map belongs to whichever channel was last
            # harvested, and on any other channel it is empty -- which floors
            # every onset at the amplitude window and returns exactly its edge,
            # 10.00 ms on every trial, with a between-trial SD of zero.
            from .app import _detection_config_kwargs
            _params = self._snapshot_analysis_params()
            _setup = (_params.get("chan_settings") or {}).get(
                _params.get("channel_idx", getattr(self, "channel_idx", 0))) or {}

            def _own(key, default=None):
                return _setup.get(key, _params.get(key, default))

            _cfg = PipelineConfig(
                pre_ms=float(_params["pre_ms"]),
                post_ms=float(_params["post_ms"]),
                prestim_ms=float(_params["prestim_ms"]),
                ptp_start=float(_params["ptp_start"]),
                ptp_end=float(_params["ptp_end"]),
                window_map=_own("window_map", {}) or {},
                latency_map=_own("latency_map", {}) or {},
                peak_fraction=_params["peak_fraction"],
                min_peak_amplitude=_params["min_amp"],
                slope_threshold=_params["slope_threshold"],
                onset_method=_params["onset_method"],
                onset_bootstrap_crit=_params["onset_bootstrap_crit"],
                onset_bootstrap_n=_params["onset_bootstrap_n"],
                onset_bigoni_smooth_ms=_params.get("onset_bigoni_smooth_ms", 0.5),
                onset_bigoni_min_run_ms=_params.get("onset_bigoni_min_run_ms", 0.5),
                onset_bigoni_walkback_sd=_params.get("onset_bigoni_walkback_sd", 1.0),
                onset_anchor=_params.get("onset_anchor", False),
                onset_anchor_halfwidth_ms=_params.get("onset_anchor_halfwidth_ms", 8.0),
                # The ptp_anchor* settings are NOT passed here. They are not
                # Tk-backed, so config_detection_kwargs returns them, and
                # naming them as well raises "got multiple values for keyword
                # argument" -- which the try block caught, leaving an empty
                # seed and the Inspector detecting on its own. Which settings
                # that mapping carries depends on the params it is given, so
                # checking it against an empty dict proves nothing.
                **_detection_config_kwargs(_params))
            _fs = payload["fs"]
            # Detect on ANALYSIS-shaped segments, seed in INSPECTOR space.
            #
            # These are two different cuts of the same trial. The preview keeps
            # the full pre-stimulus baseline -- prestim_ms, so the Inspector has
            # it to draw -- while the analysis detects on the shorter epoch,
            # pre_ms. Handed the long ones, the detector found nothing at all on
            # any trial, the seed came back empty, and the Inspector quietly
            # detected each trial itself: the very behaviour this replaced.
            #
            # So each trial is trimmed to the analysis window, detected exactly
            # as the run detects it, and the resulting time converted back to
            # the long segment's index space -- which is what the run does when
            # it seeds the Inspector after a completed analysis.
            _sb_seed = int(round(float(payload["prestim_ms"]) * _fs / 1000))
            # The amplitude window is derived AFTER the onsets, per stimulus
            # type, exactly as the run derives it at Stage 5d.
            #
            # Seeding only the onset left the Inspector to find the peaks in
            # the file-wide window, while the run measures them in a window
            # anchored to that type's median onset. On an M-wave starting at
            # 4 ms with a 10 ms window start, the first phase falls outside it
            # entirely and peak-to-peak is read from whatever is left --
            # a different NUMBER, not merely a marker in a different place.
            _ptp_ms = {}
            for _st, _segs in (payload["segments"] or {}).items():
                if not len(_segs):
                    continue
                # Detect over EVERY trial of this type, not only the ones being
                # shown: the anchor is a median across the sample, so a subset
                # moves it and moves the amplitude window with it.
                _det = (payload.get("every") or {}).get(_st) or _segs
                _before, _after = window_samples(_cfg, _st, _fs)
                _lo = max(0, _sb_seed - _before)
                _ana = np.asarray(_det)[:, _lo: _sb_seed + _after]
                if _ana.shape[1] < _before + 2:
                    continue
                _p0 = _before + int(round(float(_cfg.ptp_start) * _fs / 1000))
                _p1 = _before + int(round(float(_cfg.ptp_end) * _fs / 1000))
                _onsets = pipeline_detect_onsets(
                    _st, _ana, set(), _p0, _p1, _fs, _cfg,
                    log_callback=self.log)
                # Detection is keyed by position within ALL trials; the
                # Inspector numbers only the ones it was given. Map one to the
                # other, or every marker lands on the wrong trial.
                _shown = (payload.get("picked") or {}).get(_st)
                for _disp, _oms in enumerate(
                        [_onsets.get(_i) for _i in
                         (_shown if _shown is not None
                          else range(len(_segs)))]):
                    if _oms is not None:
                        _seed[(_st, _disp)] = {
                            "onset_idx": _sb_seed + int(round(_oms * _fs / 1000))
                        }
                try:
                    _w0, _w1 = ptp_window_for_stim_type(
                        _st, _onsets or {}, _fs, _cfg, _p0, _p1, _before,
                        log_callback=self.log)[:2]
                    # Back to ms about the stimulus, which is what the
                    # Inspector takes.
                    _ptp_ms[_st] = ((_w0 - _before) * 1000.0 / _fs,
                                    (_w1 - _before) * 1000.0 / _fs)
                except Exception:
                    pass
        except Exception as exc:                      # noqa: BLE001 — reported
            import traceback
            _seed = {}
            self.log(f"   ⚠️  Could not pre-detect for the preview "
                     f"({type(exc).__name__}: {exc}); the Inspector will "
                     f"detect each trial on its own, which may not match the "
                     f"run.")
            for _line in traceback.format_exc().strip().splitlines()[-3:]:
                self.log(f"      {_line.strip()}")

        # Say what was seeded, positively.
        #
        # A silent seed is indistinguishable from no seed: the markers simply
        # look wrong, and the only clue is a warning that appears when the
        # detection RAISES but not when it merely returns nothing. Reporting
        # the count either way makes the two cases tell themselves apart.
        if _seed:
            _by_type = {}
            for (_st, _i) in _seed:
                _by_type[_st] = _by_type.get(_st, 0) + 1
            # Out of how many, and how many the analyst has already placed.
            #
            # "A·first 4" reads as a shortfall against a run that reports a
            # latency for every trial, and looks like the preview and the run
            # disagreeing. They need not: the run honours saved manual onsets,
            # which detection neither produces nor counts. Four detected and
            # two placed by hand is six trials with a latency, and no
            # disagreement at all -- but only the numerator was ever shown, so
            # the two cases were indistinguishable.
            _saved = getattr(self, "segments_metadata", None) or {}
            _parts = []
            for _st in sorted(_by_type):
                _n_shown = len((payload.get("picked") or {}).get(_st)
                               or (payload.get("segments") or {}).get(_st) or ())
                # Only the ones detection MISSED. A trial both detected and
                # hand-placed is one trial, and counting it twice would make
                # the total exceed the trials shown.
                _found = {_i for (_s, _i) in _seed if _s == _st}
                _n_manual = sum(
                    1 for (_mst, _mi), _m in _saved.items()
                    if _mst == _st and _mi not in _found
                    and isinstance(_m, dict) and "onset_idx" in _m
                    and not _m.get("onset_auto_failed"))
                _txt = f"{_st} {_by_type[_st]}/{_n_shown}"
                if _n_manual:
                    _txt += f" (+{_n_manual} you placed)"
                _parts.append(_txt)
            self.log("   Onsets pre-detected with the analysis settings: "
                     + ", ".join(_parts))
            if any("you placed" in _p for _p in _parts):
                self.log("      A trial detection missed may still carry an "
                         "onset you placed, and the run uses those. This count "
                         "is detection only, so it can read lower than the "
                         "results without the two disagreeing.")
            if _ptp_ms:
                # Say when a window is the file-wide fallback rather than one
                # anchored to that type's median onset. They look identical --
                # two numbers in ms -- but an anchored window follows the
                # response and the fallback does not, and the difference is
                # exactly what the run reports separately. Anchoring declines
                # when too few trials were detected to give a reliable median,
                # so an unanchored window here usually means the count above is
                # low, and the two lines explain each other.
                _fallback = (float(_cfg.ptp_start), float(_cfg.ptp_end))
                _bits = []
                for k, v in sorted(_ptp_ms.items()):
                    _bits.append(
                        f"{k} {v[0]:.1f}-{v[1]:.1f} ms"
                        + (" (file-wide, not anchored)"
                           if (round(v[0], 1), round(v[1], 1))
                           == (round(_fallback[0], 1), round(_fallback[1], 1))
                           else ""))
                self.log("   Amplitude window per type: " + ", ".join(_bits))
        else:
            # Say what it looked with, not only that it found nothing.
            #
            # "No onsets were pre-detected" names a symptom shared by every
            # possible cause: a latency window that excludes the response, a
            # delay applied here but not by the run (or the reverse), an
            # amplitude gate above the response, a profile that never reached
            # this channel. Distinguishing them meant reading waveforms. The
            # settings actually used are the shortest route to the answer, and
            # the run prints the same three, so the two logs can be compared
            # line for line.
            self.log("   ⚠️  No onsets were pre-detected — the Inspector will "
                     "detect each trial on its own, which may not match the "
                     "run.")
            try:
                _lat = _cfg.latency_map or {}
                _dly = payload.get("delay_map") or {}
                for _st in sorted((payload.get("segments") or {})):
                    _w = _lat.get(_st)
                    self.log(
                        f"      {_st}: latency window "
                        + (f"{_w[0]:.0f}-{_w[1]:.0f} ms" if _w
                           else "NOT SET (falls back to the amplitude window)")
                        + f", delay {float(_dly.get(_st, 0.0)):+.1f} ms"
                        + f", min amplitude {float(_cfg.min_peak_amplitude):.3f} mV")
                self.log("      Compare these with the run's — a difference "
                         "here is why the preview and the analysis disagree.")
            except Exception:               # noqa: BLE001 — diagnostics only
                pass

        self.log("   ℹ️  Outlier decisions do not exist yet — every chosen "
                 "trial is shown. Markers are fixed and nothing is saved.")

        self._open_inspector_preview(
            payload["segments"], payload["fs"],
            payload["prestim_ms"], payload["post_ms"],
            payload["unit"],
            dict(getattr(self, "label_map", {}) or {}),
            dict(getattr(self, "color_map", {}) or {}),
            metadata_dict=_seed, ptp_windows_by_type=_ptp_ms)
        self.log("🔎 Preview closed — no changes were saved")
