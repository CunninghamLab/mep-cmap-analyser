"""
mep_cmap.conditions_tab
~~~~~~~~~~~~~~~~~~~~~~~
Setup ▸ Conditions — assigning a recording's events to named conditions.

The rules live in :mod:`mep_cmap.conditions`, which holds no Tk and is tested
without a display. This module is the surface: a table, a trial selector, and a
pane for looking at what has been selected. Keeping the two apart is what allows
the parsing, splitting and completeness rules to be exercised properly, since
almost none of what can go wrong here is a drawing problem.

The workflow
------------
The table opens populated: one row per stimulus type the event sources found,
each holding all of that type's trials. That state produces exactly the analysis
the recording produced before conditions existed, so the tab is somewhere to go
when a file needs it rather than a step every file must pass through.

From there a row is selected, its trials appear in the list and its epochs in
the review pane, and the row can be split: select ten of twenty ``A`` trials,
name them, and the remaining ten stay behind. The point of doing it here rather
than by typing two trial lists is that a split cannot lose or duplicate a
trial, which two independent assignments can.

Why look at the epochs at all
-----------------------------
Because the question the tab answers is not one the file can answer. Deciding
that trials 1-10 were the baseline block is the analyst's knowledge, and the
only check available is whether the waveforms look like they belong together.
Hence overlay, average, or both: the overlay shows spread and outliers, the
average shows what the condition will contribute, and comparing two conditions
side by side is how a split is judged before it is applied.

No detection markers are drawn here, deliberately. This is raw epochs before
any analysis; onset and amplitude belong to Preview detection, where they are
shown one trial at a time because a dozen sets of markers on one axes cannot be
read.
"""

from __future__ import annotations

import os
import tkinter as tk
from dataclasses import replace
from tkinter import messagebox, simpledialog, ttk

import numpy as np

from . import conditions as C
from .tooltips import attach_info_icon

#: Colours for conditions drawn together. Chosen to stay distinguishable when
#: several are overlaid translucently, which the default cycle does not.
_CYCLE = ("#1F3864", "#B03A2E", "#1E8449", "#7D3C98",
          "#B9770E", "#117864", "#6C3483", "#943126")

#: How many trials may be overlaid before the pane draws only the average.
#: Two hundred translucent lines is neither readable nor quick, and the average
#: is what the analyst is looking for by that point.
OVERLAY_LIMIT = 120

#: How many edits back the undo stack reaches. Deep enough to recover from a
#: run of mistaken splits, shallow enough that it cannot grow without bound in
#: a long session.
UNDO_DEPTH = 50


HELP = {
    "table": (
        "One row per condition. The table opens with a row for each stimulus "
        "type the event sources found, holding all of that type's trials, "
        "which reproduces the analysis this recording would have had before "
        "conditions existed.\n\n"
        "Split a row to separate trials that the recording cannot distinguish "
        "but you can: twenty pulses labelled A may be ten before an "
        "intervention and ten after. The stimulus type stays A — it decides "
        "how the response is detected — and the condition records what the "
        "trial was for."
    ),
    "trials": (
        "Trials of the selected condition, numbered as the trial file and the "
        "Data Inspector number them.\n\n"
        "Select some and use Split to move them into a new condition, or Set "
        "to make them the whole of this one. The review pane follows the "
        "selection."
    ),
    "review": (
        "The selected trials, drawn from the recording as they will be cut.\n\n"
        "Overlay shows spread and outliers; Average shows what the condition "
        "will contribute, with one standard deviation shaded. Selecting two "
        "conditions draws both, in different colours, which is how a split is "
        "checked before it is applied.\n\n"
        "No onset or amplitude markers are drawn: these are raw epochs, and "
        "detection is reviewed in Preview detection where one trial is shown "
        "at a time."
    ),
    "apply": (
        "Writes the conditions to a BIDS events file beside the recording, "
        "with a column for the stimulus type and a column for the condition.\n\n"
        "It is an ordinary events file, not a private format: this tool reads "
        "it back on the next load, and so can anything else that understands "
        "BIDS.\n\n"
        "Every event must belong to a condition or be explicitly excluded. A "
        "trial in neither would simply disappear between the recording and the "
        "analysis, with nothing saying so."
    ),
}


class ConditionsTabMixin:
    """Setup ▸ Conditions."""

    # ── construction ─────────────────────────────────────────────────────────

    def _build_conditions_tab(self, parent):
        self._cond_rows = []          # list[C.ConditionRow]
        self._cond_stim_times = {}    # {stim_type: [onset, ...]}
        self._cond_segments = {}      # {stim_type: ndarray (n_trials, n_samples)}
        self._cond_axis = None        # ms, shared by the drawn epochs
        self._cond_selected_rows = () # table rows currently shown
        self._cond_source_path = None # the recording the table was built from
        self._cond_confirmed_path = None  # the recording last confirmed here
        self._cond_confirming = False # set while handing over to the labels tab
        # Undo/redo stacks of (rows, what-was-done). A snapshot is the list
        # itself rather than a copy that has to be kept honest: ConditionRow is
        # frozen and every editing helper returns a new list.
        self._cond_undo_stack = []
        self._cond_redo_stack = []
        self._cond_list_index = []    # (row index, trial index) per list entry

        head = tk.Frame(parent)
        head.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(head, text="Conditions",
                 font=("TkDefaultFont", 11, "bold")).pack(side="left")
        attach_info_icon(head, HELP["apply"]).pack(side="left", padx=(4, 0))
        self._cond_status = tk.Label(head, text="No recording loaded.", fg="#888")
        self._cond_status.pack(side="left", padx=(12, 0))
        tk.Button(head, text="⟳ Reload from file",
                  command=self._cond_reload).pack(side="right")

        body = tk.Frame(parent)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 4))

        # ── the table ────────────────────────────────────────────────────────
        left = tk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        _lh = tk.Frame(left); _lh.pack(fill="x")
        tk.Label(_lh, text="Conditions table").pack(side="left")
        attach_info_icon(_lh, HELP["table"]).pack(side="left", padx=(3, 0))

        cols = ("stim", "cond", "trials", "n")
        self._cond_tree = ttk.Treeview(left, columns=cols, show="headings",
                                       selectmode="extended", height=12)
        for c, txt, w in (("stim", "Stim", 70), ("cond", "Condition", 130),
                          ("trials", "Trials", 210), ("n", "N", 45)):
            self._cond_tree.heading(c, text=txt)
            self._cond_tree.column(c, width=w,
                                   anchor="w" if c != "n" else "e")
        _tsb = ttk.Scrollbar(left, orient="vertical",
                             command=self._cond_tree.yview)
        self._cond_tree.configure(yscrollcommand=_tsb.set)
        _tsb.pack(side="right", fill="y")
        self._cond_tree.pack(fill="both", expand=True, pady=(2, 4))
        self._cond_tree.bind("<<TreeviewSelect>>",
                             lambda _e: self._cond_on_row_selected())
        self._cond_tree.bind("<Double-1>", lambda _e: self._cond_rename())

        btns = tk.Frame(left); btns.pack(fill="x")
        for txt, cmd in (("Split selection…", self._cond_split),
                         ("Set from selection", self._cond_set_from_selection),
                         ("Rename…", self._cond_rename),
                         ("Auto fill…", self._cond_autofill),
                         ("Exclude", self._cond_toggle_exclude),
                         ("Delete", self._cond_delete)):
            tk.Button(btns, text=txt, command=cmd).pack(side="left", padx=(0, 4))

        hist = tk.Frame(left)
        hist.pack(fill="x", pady=(4, 0))
        self._cond_undo_btn = tk.Button(hist, text="\u21b6 Undo", width=20,
                                        state="disabled",
                                        command=self._cond_undo)
        self._cond_undo_btn.pack(side="left")
        self._cond_redo_btn = tk.Button(hist, text="\u21b7 Redo", width=20,
                                        state="disabled",
                                        command=self._cond_redo)
        self._cond_redo_btn.pack(side="left", padx=(4, 0))
        tk.Label(hist, fg="grey", text="  Ctrl+Z / Ctrl+Y").pack(side="left")

        # ── trials ───────────────────────────────────────────────────────────
        mid = tk.Frame(body)
        mid.pack(side="left", fill="y", padx=(10, 10))
        _mh = tk.Frame(mid); _mh.pack(fill="x")
        tk.Label(_mh, text="Trials").pack(side="left")
        attach_info_icon(_mh, HELP["trials"]).pack(side="left", padx=(3, 0))
        _holder = tk.Frame(mid); _holder.pack(fill="both", expand=True)
        _sb = ttk.Scrollbar(_holder, orient="vertical")
        # exportselection=False, or selecting in the table clears this list:
        # Tk hands the X selection to whichever widget was touched last.
        self._cond_list = tk.Listbox(_holder, selectmode="extended", width=14,
                                     height=16, exportselection=False,
                                     yscrollcommand=_sb.set)
        _sb.config(command=self._cond_list.yview)
        _sb.pack(side="right", fill="y")
        self._cond_list.pack(side="left", fill="both", expand=True, pady=(2, 4))
        self._cond_list.bind("<<ListboxSelect>>",
                            lambda _e: self._cond_draw())
        tk.Button(mid, text="Select all",
                  command=self._cond_select_all_trials).pack(fill="x")

        # Bound here rather than beside the table: both widgets must exist, and
        # the listbox is built after it. Bound to the two widgets rather than
        # the window, so that Ctrl+Z in a dialogue's entry field is still the
        # entry's own undo.
        for _seq, _fn in (("<Control-z>", self._cond_undo),
                          ("<Control-Z>", self._cond_undo),
                          ("<Control-y>", self._cond_redo),
                          ("<Control-Y>", self._cond_redo)):
            for _w in (self._cond_tree, self._cond_list):
                _w.bind(_seq, lambda _e, f=_fn: (f(), "break")[1])

        # ── review ───────────────────────────────────────────────────────────
        right = tk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        _rh = tk.Frame(right); _rh.pack(fill="x")
        tk.Label(_rh, text="Review epochs").pack(side="left")
        attach_info_icon(_rh, HELP["review"]).pack(side="left", padx=(3, 0))
        self._cond_mode = tk.StringVar(value="both")
        for txt, val in (("Overlay", "overlay"), ("Average", "average"),
                         ("Both", "both")):
            tk.Radiobutton(_rh, text=txt, value=val,
                           variable=self._cond_mode,
                           command=self._cond_draw).pack(side="left", padx=(8, 0))
        self._cond_note = tk.Label(right, text="", fg="#1F3864", anchor="w")
        self._cond_note.pack(fill="x", pady=(2, 0))

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self._cond_fig = Figure(figsize=(6.4, 4.2), dpi=100)
        self._cond_ax = self._cond_fig.add_subplot(111)
        self._cond_fig.subplots_adjust(left=0.12, right=0.99, top=0.96,
                                       bottom=0.14)
        self._cond_canvas = FigureCanvasTkAgg(self._cond_fig, master=right)
        self._cond_canvas.get_tk_widget().pack(fill="both", expand=True,
                                               pady=(2, 0))

        # ── apply ────────────────────────────────────────────────────────────
        foot = tk.Frame(parent)
        foot.pack(fill="x", padx=10, pady=(4, 10))
        self._cond_allow_unassigned = tk.BooleanVar(value=False)
        tk.Checkbutton(
            foot, variable=self._cond_allow_unassigned,
            text="Exclude any trial left out of every condition",
            command=self._cond_refresh_status).pack(side="left")
        self._cond_apply_btn = tk.Button(
            foot, text="✔  Confirm events & continue",
            command=self._cond_apply)
        self._cond_apply_btn.pack(side="right")

    # ── loading ──────────────────────────────────────────────────────────────

    def _cond_tab_shown(self):
        """Populate on first sight of a recording, and after the file changes.

        Not on every visit: an assignment already made must survive going away
        to check something and coming back, which is why Reload is a button.
        The table is rebuilt only when it is empty, or when the recording it
        was built from is no longer the one loaded.
        """
        path = (self.file_path.get() if hasattr(self, "file_path") else "") or ""
        if not path:
            return
        if self._cond_rows and getattr(self, "_cond_source_path", None) == path:
            return
        self._cond_source_path = path
        self._cond_reload()

    def _cond_reload(self):
        """Rebuild the table from the recording's own events.

        Discards edits, which is why it is a button rather than something that
        happens on every visit to the tab: returning to check a waveform should
        not silently undo an assignment.
        """
        path = (self.file_path.get() if hasattr(self, "file_path")
                else "") or ""
        if not path or not os.path.isfile(path):
            self._cond_status.config(text="No recording loaded.", fg="#888")
            self._cond_rows, self._cond_stim_times = [], {}
            self._cond_refresh_table()
            return
        try:
            events, _warn = self._configured_events(path)
        except Exception as exc:                      # noqa: BLE001 — reported
            self._cond_status.config(text=f"Could not read events: {exc}",
                                     fg="#B03A2E")
            return
        if not events:
            self._cond_status.config(
                text="No stimulus events found in this recording.", fg="#B03A2E")
            self._cond_rows, self._cond_stim_times = [], {}
            self._cond_refresh_table()
            return

        # Only the stimuli inside the selected range, by the same rule the
        # analysis applies. Conditions are assigned by trial INDEX, so a table
        # built from the whole recording numbers its trials differently from
        # the analysis and every assignment would land on the wrong trial --
        # and the waveforms drawn beside the list are already cropped, so the
        # two halves of this tab would disagree with each other as well.
        from .pipeline import crop_stim_times
        try:
            _p = self._snapshot_analysis_params()
            events = crop_stim_times(events, _p.get("crop_ranges"),
                                     _p.get("crop_start"), _p.get("crop_end"))
        except Exception:
            pass
        if not events:
            self._cond_status.config(
                text="No stimulus events fall inside the selected data range.",
                fg="#B03A2E")
            self._cond_rows, self._cond_stim_times = [], {}
            self._cond_refresh_table()
            return

        self._cond_source_path = path
        self._cond_stim_times = {k: list(v) for k, v in events.items()}
        self._cond_rows = C.rows_from_events(self._cond_stim_times)
        # Undoing past a reload would restore rows belonging to a recording
        # that is no longer open.
        self._cond_undo_stack.clear()
        self._cond_redo_stack.clear()
        self._cond_undo_stack.clear()
        self._cond_redo_stack.clear()
        self._cond_segments, self._cond_axis = {}, None
        _cropped = ""
        try:
            _p = self._snapshot_analysis_params()
            if _p.get("crop_ranges") or _p.get("crop_start") is not None:
                _cropped = "  (within the selected data range)"
        except Exception:
            pass
        self._cond_status.config(
            text=f"{os.path.basename(path)} — "
                 + ", ".join(f"{k} ({len(v)})"
                             for k, v in sorted(events.items())) + _cropped,
            fg="#555")
        self._cond_refresh_table()

    def _cond_load_segments(self):
        """Read and cut the recording once, for the review pane only.

        Separate from the analysis path: this is a look at raw epochs, and the
        window used is the one the analysis would use so that what is judged
        here is what will be measured.
        """
        if self._cond_segments:
            return True
        path = (self.file_path.get() if hasattr(self, "file_path")
                else "") or ""
        if not path or not self._cond_stim_times:
            return False
        try:
            from .pipeline import (PipelineConfig, pipeline_load_file,
                                   window_samples)
            params = self._snapshot_analysis_params()
            emg, time, fs, _unit, _stim = pipeline_load_file(
                params["input_path"], params["channel_idx"],
                params["marker_choice"],
                crop_ranges=params.get("crop_ranges"),
                crop_start=params.get("crop_start"),
                crop_end=params.get("crop_end"),
                sources=None, event_rows=None)
            wincfg = PipelineConfig(pre_ms=float(params["pre_ms"]),
                                    post_ms=float(params["post_ms"]),
                                    window_map=params.get("window_map") or {})
            out = {}
            for stim, times in self._cond_stim_times.items():
                before, after = window_samples(wincfg, stim, fs)
                segs = []
                for t in times:
                    ix = int(np.argmin(np.abs(time - t)))
                    seg = emg[max(0, ix - before): ix + after]
                    segs.append(seg if len(seg) == before + after else None)
                keep = [s for s in segs if s is not None]
                if keep:
                    out[stim] = (np.vstack(keep),
                                 np.arange(len(keep[0])) * 1000.0 / fs
                                 - before * 1000.0 / fs)
            self._cond_segments = out
            return bool(out)
        except Exception as exc:                      # noqa: BLE001 — reported
            self._cond_note.config(
                text=f"Could not read the waveforms: {exc}", fg="#B03A2E")
            return False

    # ── the table ────────────────────────────────────────────────────────────

    def _cond_refresh_table(self):
        sel = set(self._cond_tree.selection()) if self._cond_tree else set()
        self._cond_tree.delete(*self._cond_tree.get_children())
        for i, row in enumerate(self._cond_rows):
            self._cond_tree.insert(
                "", "end", iid=str(i),
                values=(row.stim_type,
                        ("— excluded —" if row.excluded
                         else (row.condition or "")),
                        C.format_trials(row.trials), row.n))
        for iid in sel:
            if self._cond_tree.exists(iid):
                self._cond_tree.selection_add(iid)
        self._cond_refresh_history_buttons()
        self._cond_refresh_status()

    def _cond_refresh_status(self):
        """Say whether the table can be applied, and why not when it cannot."""
        if not self._cond_rows:
            self._cond_apply_btn.config(state="disabled")
            return
        try:
            C.validate(self._cond_rows, self._cond_stim_times,
                       allow_unassigned=bool(self._cond_allow_unassigned.get()))
        except C.ConditionError as exc:
            self._cond_note.config(text=str(exc), fg="#B03A2E")
            self._cond_apply_btn.config(state="disabled")
            return
        self._cond_apply_btn.config(state="normal")
        n = sum(r.n for r in self._cond_rows if not r.excluded)
        self._cond_note.config(
            text=f"{len(self._cond_rows)} condition(s), {n} trial(s) — ready to "
                 f"apply", fg="#1E8449")

    def _cond_selected_indices(self):
        return tuple(int(i) for i in self._cond_tree.selection()
                     if str(i).isdigit())

    def _cond_on_row_selected(self):
        """Fill the trial list from the selected rows and draw them."""
        rows = self._cond_selected_indices()
        self._cond_selected_rows = rows
        self._cond_list.delete(0, "end")
        self._cond_list_index = []
        for ri in rows:
            row = self._cond_rows[ri]
            for t in row.trials:
                self._cond_list.insert("end", f"Trial {t + 1}")
                self._cond_list_index.append((ri, t))
        self._cond_list.selection_set(0, "end")
        self._cond_draw()

    def _cond_select_all_trials(self):
        self._cond_list.selection_set(0, "end")
        self._cond_draw()

    def _cond_current_selection(self):
        """{row index: (trial index, ...)} for what is selected in the list."""
        out = {}
        for i in self._cond_list.curselection():
            ri, t = self._cond_list_index[int(i)]
            out.setdefault(ri, []).append(t)
        return {k: tuple(sorted(v)) for k, v in out.items()}

    # ── history ──────────────────────────────────────────────────────────────

    def _cond_commit(self, new_rows, what):
        """Adopt an edited table, remembering what it replaced.

        Called only once an edit has succeeded, so a refused one -- a split
        that would move every trial, a name containing the separator -- leaves
        nothing behind. An undo that does nothing visible is worse than no
        undo: it teaches the analyst the button is unreliable, and the next
        press is the one that goes too far.
        """
        self._cond_undo_stack.append((list(self._cond_rows), what))
        del self._cond_undo_stack[:-UNDO_DEPTH]
        # A new edit invalidates the redo path: redoing onto a table that has
        # since changed would apply an edit to rows it was never made against.
        self._cond_redo_stack.clear()
        self._cond_rows = list(new_rows)
        self._cond_refresh_table()

    def _cond_undo(self):
        if not self._cond_undo_stack:
            return
        rows, what = self._cond_undo_stack.pop()
        self._cond_redo_stack.append((list(self._cond_rows), what))
        self._cond_rows = rows
        self._cond_refresh_table()
        self.log(f"\u21b6 Undid: {what}")

    def _cond_redo(self):
        if not self._cond_redo_stack:
            return
        rows, what = self._cond_redo_stack.pop()
        self._cond_undo_stack.append((list(self._cond_rows), what))
        self._cond_rows = rows
        self._cond_refresh_table()
        self.log(f"\u21b7 Redid: {what}")

    def _cond_refresh_history_buttons(self):
        """Name what will be undone, rather than offering a bare Undo.

        "Undo split" is a different proposition from "Undo delete", and the
        analyst should not have to remember which came last to know what the
        button will do.
        """
        if self._cond_undo_stack:
            self._cond_undo_btn.config(
                state="normal",
                text=f"\u21b6 Undo {self._cond_undo_stack[-1][1]}")
        else:
            self._cond_undo_btn.config(state="disabled", text="\u21b6 Undo")
        if self._cond_redo_stack:
            self._cond_redo_btn.config(
                state="normal",
                text=f"\u21b7 Redo {self._cond_redo_stack[-1][1]}")
        else:
            self._cond_redo_btn.config(state="disabled", text="\u21b7 Redo")

    # ── editing ──────────────────────────────────────────────────────────────

    def _cond_split(self):
        sel = self._cond_current_selection()
        if len(sel) != 1:
            messagebox.showinfo(
                "Split", "Select trials from one condition to split it.",
                parent=self.root)
            return
        (ri, trials), = sel.items()
        name = simpledialog.askstring(
            "Split condition",
            f"{len(trials)} trial(s) selected.\n\n"
            f"Name for the new condition:", parent=self.root)
        if not name:
            return
        try:
            new_rows = C.split_row(self._cond_rows, ri, trials,
                                   new_condition=name)
        except C.ConditionError as exc:
            messagebox.showwarning("Split", str(exc), parent=self.root)
            return
        self._cond_commit(new_rows, "split")

    def _cond_set_from_selection(self):
        """Make the selection the whole of its row.

        The complement becomes unassigned rather than silently dropped, which
        the status line then reports: removing trials from a condition is a
        decision, and where they went should be visible.
        """
        sel = self._cond_current_selection()
        if len(sel) != 1:
            messagebox.showinfo(
                "Set", "Select trials from one condition.", parent=self.root)
            return
        (ri, trials), = sel.items()
        new_rows = list(self._cond_rows)
        new_rows[ri] = replace(new_rows[ri], trials=trials)
        self._cond_commit(new_rows, "set trials")

    def _cond_rename(self):
        rows = self._cond_selected_indices()
        if not rows:
            return
        current = self._cond_rows[rows[0]].condition
        name = simpledialog.askstring("Condition name", "Name:",
                                      initialvalue=current, parent=self.root)
        if name is None:
            return
        try:
            clean = C.sanitise_name(name)
        except C.ConditionError as exc:
            messagebox.showwarning("Condition name", str(exc), parent=self.root)
            return
        new_rows = list(self._cond_rows)
        for ri in rows:
            new_rows[ri] = replace(new_rows[ri], condition=clean)
        self._cond_commit(new_rows, "rename")

    def _cond_autofill(self):
        rows = self._cond_selected_indices()
        if len(rows) != 1:
            messagebox.showinfo("Auto fill", "Select one condition to divide.",
                                parent=self.root)
            return
        n = simpledialog.askinteger(
            "Auto fill", "Trials per condition:", minvalue=1, parent=self.root)
        if not n:
            return
        stim = self._cond_rows[rows[0]].stim_type
        try:
            new_rows = C.autofill(
                self._cond_rows, rows[0], per_row=n,
                n_available=len(self._cond_stim_times.get(stim) or []))
        except C.ConditionError as exc:
            messagebox.showwarning("Auto fill", str(exc), parent=self.root)
            return
        self._cond_commit(new_rows, "auto fill")

    def _cond_toggle_exclude(self):
        rows = self._cond_selected_indices()
        if not rows:
            return
        new_rows = list(self._cond_rows)
        for ri in rows:
            new_rows[ri] = replace(new_rows[ri],
                                   excluded=not new_rows[ri].excluded)
        self._cond_commit(new_rows, "exclude")

    def _cond_delete(self):
        rows = set(self._cond_selected_indices())
        if not rows:
            return
        self._cond_commit([r for i, r in enumerate(self._cond_rows)
                           if i not in rows], "delete")

    # ── drawing ──────────────────────────────────────────────────────────────

    def _cond_draw(self):
        ax = self._cond_ax
        ax.clear()
        sel = self._cond_current_selection()
        if not sel or not self._cond_load_segments():
            ax.text(0.5, 0.5, "Select a condition to see its epochs.",
                    ha="center", va="center", color="0.5",
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            self._cond_canvas.draw_idle()
            return

        mode = self._cond_mode.get()
        drawn = 0
        for n, (ri, trials) in enumerate(sorted(sel.items())):
            row = self._cond_rows[ri]
            pack = self._cond_segments.get(row.stim_type)
            if pack is None:
                continue
            stack, axis = pack
            usable = [t for t in trials if t < stack.shape[0]]
            if not usable:
                continue
            block = stack[usable, :]
            colour = _CYCLE[n % len(_CYCLE)]
            label = row.describe().split(":")[0]

            # Above the limit only the average is drawn: two hundred
            # translucent lines is neither readable nor quick, and by then the
            # average is what is being looked at anyway.
            if mode in ("overlay", "both") and len(usable) <= OVERLAY_LIMIT:
                for s in block:
                    ax.plot(axis, s, color=colour, alpha=0.18, linewidth=0.6)
            if mode in ("average", "both") or len(usable) > OVERLAY_LIMIT:
                mean = block.mean(axis=0)
                ax.plot(axis, mean, color=colour, linewidth=2.0, label=label)
                if block.shape[0] > 1:
                    sd = block.std(axis=0)
                    ax.fill_between(axis, mean - sd, mean + sd,
                                    color=colour, alpha=0.15, linewidth=0)
            elif mode == "overlay":
                ax.plot([], [], color=colour, linewidth=2.0, label=label)
            drawn += len(usable)

        ax.axvline(0.0, color="0.4", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Time about the stimulus (ms)", fontsize=8)
        ax.set_ylabel(f"EMG ({getattr(self, 'emg_unit', 'mV')})", fontsize=8)
        ax.tick_params(labelsize=8)
        if len(sel) > 1:
            ax.legend(fontsize=7, loc="upper right")
        self._cond_note.config(
            text=f"{drawn} trial(s) from {len(sel)} condition(s)", fg="#1F3864")
        self._cond_canvas.draw_idle()

    # ── apply ────────────────────────────────────────────────────────────────

    def _cond_apply(self):
        """Validate, write the events file, and hand the groups to the analysis."""
        try:
            rows = C.validate(
                self._cond_rows, self._cond_stim_times,
                allow_unassigned=bool(self._cond_allow_unassigned.get()))
        except C.ConditionError as exc:
            messagebox.showwarning("Conditions", str(exc), parent=self.root)
            return
        self._cond_rows = rows

        event_rows = C.to_event_rows(rows, self._cond_stim_times)
        groups, decoded = C.group_events(event_rows)

        path = self.file_path.get()
        try:
            written = write_events_tsv_beside(path, event_rows)
        except Exception as exc:                      # noqa: BLE001 — reported
            messagebox.showerror(
                "Conditions",
                f"The conditions are applied to this session but the events "
                f"file could not be written:\n\n{exc}", parent=self.root)
            written = None

        # Held on the app, and read by _snapshot_analysis_params, so that the
        # analysis and the preview group trials by these conditions.
        self.condition_event_rows = event_rows
        self.condition_map = decoded

        self._cond_confirmed_path = path
        self.log("🏷  Conditions applied: "
                 + ", ".join(f"{k} ({len(v)})" for k, v in sorted(groups.items())))
        if written:
            self.log(f"   → {os.path.basename(written)}")
        self._cond_refresh_status()
        # Rebuild the labels tab from the conditions and go there. The flag
        # tells _build_labels_tab that this is the onward step rather than a
        # fresh file arriving, which would otherwise send the analyst back here.
        self._cond_confirming = True
        try:
            self._build_labels_tab(sorted(groups))
            self.log("   Tab 1a rebuilt from the conditions.")
        except Exception as exc:                      # noqa: BLE001 — reported
            self.log(f"   ⚠️  Could not rebuild the labels tab: {exc}")
        finally:
            self._cond_confirming = False


# ── writing ──────────────────────────────────────────────────────────────────

def events_tsv_path(recording_path: str) -> str:
    """``<entities>_emg.edf`` → ``<entities>_events.tsv``.

    Mirrors the reader's own rule, so that what this writes is what it reads.
    """
    d = os.path.dirname(recording_path)
    stem = os.path.splitext(os.path.basename(recording_path))[0]
    for suffix in ("_emg", "_eeg"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return os.path.join(d, stem + "_events.tsv")


def write_events_tsv_beside(recording_path: str, event_rows) -> str:
    """Write a BIDS events file next to the recording, and its sidecar.

    An ordinary events file rather than a private format: this tool reads it
    back on the next load through a path that already existed, and so can
    anything else that understands BIDS. The sidecar exists because ``condition``
    is not a BIDS-defined column, and an undocumented extra column is a column
    only its author can interpret.
    """
    import csv
    import json

    path = events_tsv_path(recording_path)
    cols = ("onset", "duration", "trial_type", "condition")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for row in event_rows:
            w.writerow({k: row.get(k, C.NA) for k in cols})

    side = os.path.splitext(path)[0] + ".json"
    with open(side, "w", encoding="utf-8") as fh:
        json.dump({
            "onset": {"Description": "Onset of the stimulus, in seconds from "
                                     "the start of the recording.",
                      "Units": "s"},
            "duration": {"Description": "Duration of the stimulus. Zero for an "
                                        "instantaneous event.",
                         "Units": "s"},
            "trial_type": {"Description": "Stimulus type as recorded by the "
                                          "acquisition system. n/a marks a "
                                          "trial excluded from analysis, "
                                          "retained here so that the file "
                                          "accounts for every event."},
            "condition": {"Description": "Condition the trial was assigned to "
                                         "by the analyst: an experimental "
                                         "factor the recording itself does not "
                                         "distinguish, such as a timepoint or "
                                         "a block. n/a where none was "
                                         "assigned."},
        }, fh, indent=2)
    return path
