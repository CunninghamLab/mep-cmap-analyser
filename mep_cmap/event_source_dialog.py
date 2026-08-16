"""
mep_cmap.event_source_dialog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configure where stimulus events come from, and see what each choice finds.

Why a preview rather than a number
----------------------------------
A threshold level is not checkable by reading it. Two volts is right or wrong
depending on the trigger's amplitude, its baseline, and whether the pulse rings
-- none of which the analyst can know from the box they typed it into. The only
useful feedback is the trace with the level drawn across it and the detected
crossings marked, and a count that changes as the level does.

On a real recording this is the difference between "Channel 6 finds 79 events
where the comments say 162" being a mystery and being visible: either the
channel carries 79 pulses, or the level sits above 83 of them, and the picture
says which.

The trace is reduced by minimum and maximum per column, not by subsampling. A
stimulus trigger is a one-sample spike; on a two-thousand-second recording,
plain subsampling drew a flat line while the detector found two hundred events.
"""

import os
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from .event_sources import (DEFAULT_REFRACTORY_MS, EDGES, EventSource,
                            decimate_for_preview, detect_threshold_crossings)


class EventSourceDialog:
    """Modal editor for a file's event sources.

    ``result`` is the new list of EventSource, or None if cancelled.
    """

    def __init__(self, master, file_path, sources, available,
                 read_channel, log=None):
        """
        master       : parent window
        file_path    : the recording, for the title only
        sources      : list[EventSource] to start from
        available    : {"embedded": [names], "analogue": [names]}
        read_channel : callable(name) -> (signal, fs) for the preview
        log          : optional callable(str)
        """
        self.result = None
        self._read_channel = read_channel
        self._log = log or (lambda _m: None)
        self._available = available or {"embedded": [], "analogue": []}
        self._sources = [EventSource.from_dict(s.to_dict()) for s in
                         (sources or [])]
        # Cached per channel: reading a long recording for every keystroke in
        # the level box would make the dialogue unusable.
        self._cache = {}

        self.top = tk.Toplevel(master)
        self.top.title("Event sources")
        self.top.transient(master)
        self.top.grab_set()

        tk.Label(self.top, justify="left", fg="grey", text=(
            f"File: {os.path.basename(file_path)}\n\n"
            "Each source contributes stimulus events. A file's own markers or\n"
            "comments are one source; a trigger channel crossed by a voltage is\n"
            "another; fixed timing is a third, for triggers the file does not\n"
            "record."
        )).pack(anchor="w", padx=14, pady=(12, 8))

        body = tk.Frame(self.top)
        body.pack(fill="both", expand=True, padx=14)

        # ── Left: the list of sources ────────────────────────────────────────
        left = tk.Frame(body)
        left.pack(side="left", fill="y")
        tk.Label(left, text="Sources", anchor="w").pack(fill="x")
        self.listbox = tk.Listbox(left, width=46, height=7,
                                  exportselection=False)
        self.listbox.pack(fill="y", expand=True)
        self.listbox.bind("<<ListboxSelect>>",
                          lambda e: self._load_selected(force=True))

        btns = tk.Frame(left)
        btns.pack(fill="x", pady=6)
        tk.Button(btns, text="Add", width=8,
                  command=self._add).pack(side="left")
        tk.Button(btns, text="Remove", width=8,
                  command=self._remove).pack(side="left", padx=4)
        # The preview refreshes as fields change, but an explicit control means
        # the analyst never has to wonder whether what is drawn is current.
        tk.Button(btns, text="Preview", width=9,
                  command=self._preview_now).pack(side="left")

        # ── Right: the editor for the selected source ────────────────────────
        right = tk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(14, 0))
        self.editor = right

        self.kind_var = tk.StringVar(value="embedded")
        krow = tk.Frame(right)
        krow.pack(fill="x", pady=(0, 6))
        tk.Label(krow, text="Kind:", width=12, anchor="w").pack(side="left")
        for label, value in (("File's own events", "embedded"),
                             ("Trigger channel", "threshold"),
                             ("Fixed interval", "interval")):
            tk.Radiobutton(krow, text=label, value=value,
                           variable=self.kind_var,
                           command=self._kind_changed).pack(side="left")

        self.fields = tk.Frame(right)
        self.fields.pack(fill="x")

        self.v_channel = tk.StringVar()
        self.v_label = tk.StringVar(value="A")
        self.v_level = tk.StringVar(value="")
        self.v_edge = tk.StringVar(value="rising")
        self.v_refrac = tk.StringVar(value=str(DEFAULT_REFRACTORY_MS))
        self.v_start = tk.StringVar(value="0.0")
        self.v_period = tk.StringVar(value="1.0")
        self.v_count = tk.StringVar(value="0")

        # ── The preview ──────────────────────────────────────────────────────
        self.count_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.count_var, fg="#1F3864",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w",
                                                          pady=(8, 2))

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(7.2, 2.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.18)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(2, 0))

        foot = tk.Frame(self.top)
        foot.pack(fill="x", pady=10)
        tk.Button(foot, text="OK", width=10,
                  command=self._ok).pack(side="right", padx=(4, 14))
        tk.Button(foot, text="Cancel", width=10,
                  command=self._cancel).pack(side="right")

        # Never open with nothing selected.
        #
        # An empty list meant an empty editor and an empty preview, so the
        # dialogue looked broken until the analyst guessed that Add came first.
        # A file with no configured sources uses its own markers, so that is
        # what the first row describes -- the current behaviour, made visible
        # and editable rather than implied by an empty list.
        if not self._sources:
            embedded = self._available.get("embedded") or []
            self._sources.append(EventSource(
                kind="embedded", channel=embedded[0] if embedded else ""))
        self._refresh_list()
        self.listbox.selection_set(0)
        self._load_selected()
        self.top.protocol("WM_DELETE_WINDOW", self._cancel)

    # ── list handling ────────────────────────────────────────────────────────

    def _refresh_list(self):
        self.listbox.delete(0, "end")
        for src in self._sources:
            self.listbox.insert("end", src.describe())

    def _selected_index(self):
        sel = self.listbox.curselection()
        return int(sel[0]) if sel else None

    def _add(self):
        embedded = self._available.get("embedded") or []
        self._sources.append(EventSource(
            kind="embedded", channel=embedded[0] if embedded else ""))
        self._refresh_list()
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set("end")
        # Force the draw: a scheduled one may not run before the analyst looks,
        # which is what made a newly added source appear to have no preview
        # until the row was clicked again.
        self._load_selected(force=True)

    def _remove(self):
        i = self._selected_index()
        if i is None:
            return
        del self._sources[i]
        self._refresh_list()
        if self._sources:
            self.listbox.selection_set(min(i, len(self._sources) - 1))
        self._load_selected(force=True)

    # ── editor ───────────────────────────────────────────────────────────────

    def _kind_changed(self):
        self._build_fields()
        self._apply_edits()
        self._update_preview()

    def _build_fields(self):
        for w in self.fields.winfo_children():
            w.destroy()

        kind = self.kind_var.get()

        def row(label, widget):
            r = tk.Frame(self.fields)
            r.pack(fill="x", pady=2)
            tk.Label(r, text=label, width=12, anchor="w").pack(side="left")
            widget(r)

        if kind == "embedded":
            opts = self._available.get("embedded") or [""]
            row("Events from:", lambda r: ttk.Combobox(
                r, textvariable=self.v_channel, values=opts,
                state="readonly", width=26).pack(side="left"))
            tk.Label(self.fields, fg="grey", justify="left", anchor="w",
                     text=("The markers, comments or annotations the file "
                           "already carries.")).pack(fill="x", pady=(2, 0))

        elif kind == "threshold":
            opts = self._available.get("analogue") or [""]
            row("Channel:", lambda r: ttk.Combobox(
                r, textvariable=self.v_channel, values=opts,
                state="readonly", width=26).pack(side="left"))
            row("Level:", lambda r: tk.Entry(
                r, textvariable=self.v_level, width=12).pack(side="left"))
            row("Edge:", lambda r: ttk.Combobox(
                r, textvariable=self.v_edge, values=list(EDGES),
                state="readonly", width=10).pack(side="left"))
            row("Refractory (ms):", lambda r: tk.Entry(
                r, textvariable=self.v_refrac, width=12).pack(side="left"))
            row("Stimulus type:", lambda r: tk.Entry(
                r, textvariable=self.v_label, width=12).pack(side="left"))
            tk.Label(self.fields, fg="grey", justify="left", anchor="w",
                     text=("The level is shown on the trace below with every "
                           "crossing marked.\nRefractory ignores further "
                           "crossings for that long, so a pulse that\nrings is "
                           "counted once.")).pack(fill="x", pady=(2, 0))
            for v in (self.v_level, self.v_refrac, self.v_edge, self.v_channel):
                v.trace_add("write", lambda *a: self._on_edit())

        else:
            row("Start (s):", lambda r: tk.Entry(
                r, textvariable=self.v_start, width=12).pack(side="left"))
            row("Every (s):", lambda r: tk.Entry(
                r, textvariable=self.v_period, width=12).pack(side="left"))
            row("Count:", lambda r: tk.Entry(
                r, textvariable=self.v_count, width=12).pack(side="left"))
            row("Stimulus type:", lambda r: tk.Entry(
                r, textvariable=self.v_label, width=12).pack(side="left"))
            tk.Label(self.fields, fg="grey", justify="left", anchor="w",
                     text=("Nothing is detected: these times are asserted, and "
                           "no part of the\nrecording can confirm them. Leave "
                           "the count at 0 to fill the recording.")
                     ).pack(fill="x", pady=(2, 0))
            for v in (self.v_start, self.v_period, self.v_count):
                v.trace_add("write", lambda *a: self._on_edit())

    def _load_selected(self, force=False):
        i = self._selected_index()
        if i is None:
            self.count_var.set("")
            self.ax.clear()
            self.canvas.draw() if force else self.canvas.draw_idle()
            for w in self.fields.winfo_children():
                w.destroy()
            return
        src = self._sources[i]
        self.kind_var.set(src.kind)
        self.v_channel.set(src.channel)
        self.v_label.set(src.label)
        self.v_level.set("" if src.level is None else f"{src.level:g}")
        self.v_edge.set(src.edge)
        self.v_refrac.set(f"{src.refractory_ms:g}")
        self.v_start.set(f"{src.start_s:g}")
        self.v_period.set(f"{src.period_s:g}")
        self.v_count.set(str(int(src.count)))
        self._build_fields()
        self._update_preview(force=force)

    def _on_edit(self):
        self._apply_edits()
        self._update_preview()

    def _apply_edits(self):
        """Write the fields back into the selected source.

        A half-typed number is not an error: the analyst is mid-keystroke, and
        rejecting it would fight the typing. The field is left at its previous
        value and the preview simply does not update until it parses.
        """
        i = self._selected_index()
        if i is None:
            return
        src = self._sources[i]
        src.kind = self.kind_var.get()
        src.channel = self.v_channel.get()
        src.label = self.v_label.get().strip() or "A"
        src.edge = self.v_edge.get() or "rising"
        for attr, var, cast in (("level", self.v_level, float),
                                ("refractory_ms", self.v_refrac, float),
                                ("start_s", self.v_start, float),
                                ("period_s", self.v_period, float),
                                ("count", self.v_count, int)):
            try:
                setattr(src, attr, cast(var.get()))
            except (TypeError, ValueError):
                pass
        self.listbox.delete(i)
        self.listbox.insert(i, src.describe())
        self.listbox.selection_set(i)

    # ── preview ──────────────────────────────────────────────────────────────

    def _channel_data(self, name):
        if name not in self._cache:
            try:
                sig, fs = self._read_channel(name)
                self._cache[name] = (np.asarray(sig, dtype=float), float(fs))
            except Exception as exc:
                self._cache[name] = (None, str(exc))
        return self._cache[name]

    def _preview_now(self):
        """Redraw on request, forcing the canvas rather than scheduling it."""
        self._apply_edits()
        self._update_preview(force=True)

    def _update_preview(self, force=False):
        i = self._selected_index()
        self.ax.clear()
        if i is None:
            self.canvas.draw() if force else self.canvas.draw_idle()
            return
        src = self._sources[i]

        if src.kind != "threshold":
            self.ax.text(0.5, 0.5,
                         "No preview for this kind of source." if
                         src.kind == "interval" else
                         "The file's own events need no level.",
                         ha="center", va="center", color="0.5",
                         transform=self.ax.transAxes)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.count_var.set("")
            self.canvas.draw() if force else self.canvas.draw_idle()
            return

        sig, fs = self._channel_data(src.channel)
        if sig is None:
            self.ax.text(0.5, 0.5, f"Could not read {src.channel}",
                         ha="center", va="center", color="#B03A2E",
                         transform=self.ax.transAxes)
            self.count_var.set("")
            self.canvas.draw() if force else self.canvas.draw_idle()
            return

        t, lo, hi = decimate_for_preview(sig, fs)
        self.ax.fill_between(t, lo, hi, color="0.35", linewidth=0)
        self.ax.set_xlabel("Time (s)", fontsize=8)
        self.ax.tick_params(labelsize=8)

        times = detect_threshold_crossings(
            sig, fs, src.level, src.edge, src.refractory_ms)
        self.ax.axhline(src.level, color="#B03A2E", lw=1.2)
        if times:
            self.ax.plot(times, [src.level] * len(times), "|",
                         color="#1F3864", markersize=10, markeredgewidth=1.4)
        self.count_var.set(
            f"{len(times)} event(s) detected on {src.channel}"
            + (f"  ·  first at {times[0]:.3f} s, last at {times[-1]:.3f} s"
               if times else "  ·  try a different level or edge"))
        self.canvas.draw() if force else self.canvas.draw_idle()

    # ── finish ───────────────────────────────────────────────────────────────

    def _ok(self):
        self._apply_edits()
        for src in self._sources:
            if src.kind == "threshold" and not src.channel:
                messagebox.showwarning(
                    "Incomplete source",
                    "A trigger source needs a channel.", parent=self.top)
                return
            if src.kind == "interval" and src.period_s <= 0:
                messagebox.showwarning(
                    "Incomplete source",
                    "A fixed-interval source needs a positive interval.",
                    parent=self.top)
                return
        self.result = list(self._sources)
        self.top.destroy()

    def _cancel(self):
        self.result = None
        self.top.destroy()
