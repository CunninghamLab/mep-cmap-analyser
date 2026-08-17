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

#: Shown in the embedded dropdown for "take every label the file carries",
#: which is what an unconfigured file does. Defaulting to the first label
#: instead silently narrowed the analysis to whichever one sorted first --
#: 'Start Task' on a recording whose stimuli are called 'Trigger'.
ALL_LABELS = "(every marker in the file)"

from .event_sources import (DEFAULT_REFRACTORY_MS, EDGES, EventSource,
                            decimate_for_preview, detect_threshold_crossings)


class EventSourceDialog:
    """Modal editor for a file's event sources.

    ``result`` is the new list of EventSource, or None if cancelled.
    """

    def __init__(self, master, file_path, sources, available,
                 read_channel, log=None, channel_name=None, window_ms=None):
        """
        master       : parent window
        file_path    : the recording, for the title only
        sources      : list[EventSource] to start from
        available    : {"embedded": [names], "analogue": [names]}
        read_channel : callable(name) -> (signal, fs) for the preview
        log          : optional callable(str)
        channel_name : the channel being configured, shown in the header
        window_ms    : width of the detail view, defaulting to the analysis
                       window so what is inspected is roughly the epoch that
                       will be cut

        Sources are configured for ONE channel. ``copy_to_all`` reports
        whether the analyst asked for them to be applied to the rest -- the
        same gesture as tab 1a's "Copy this setup to all channels", so the
        shared case costs one click while a channel whose trigger sits nearer
        the noise floor can still be given its own level.
        """
        self.result = None
        self.copy_to_all = False
        self._channel_name = channel_name
        # Kept because the preview resolves events through io.extract_events --
        # the same call the analysis makes. A preview that computed events its
        # own way could agree with the run today and drift from it later.
        self._file_path = file_path
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

        _where = (f"File: {os.path.basename(file_path)}"
                  + (f"      Channel: {channel_name}" if channel_name else ""))
        tk.Label(self.top, justify="left", fg="grey", text=(
            f"{_where}\n\n"
            "Each source contributes stimulus events. A file's own markers or\n"
            "comments are one source; a trigger channel crossed by a voltage is\n"
            "another; fixed timing is a third, for triggers the file does not\n"
            "record.\n\n"
            "One source is usually enough: edit the row on the left and press\n"
            "OK. Add another only when the stimuli genuinely come from more\n"
            "than one place — every source adds its own row to tab 1a.\n\n"
            "These apply to this channel. A trigger that is clean on one\n"
            "electrode can sit near the noise floor on another, so each channel\n"
            "keeps its own -- use the tick below when they should share."
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
        # "Add" reads as "confirm this selection", which is not what it does:
        # the selected row is already live and OK accepts it. Pressing Add
        # after choosing a label appended a SECOND source, which on a file
        # with two comment types brought the excluded one back as a row on
        # tab 1a. Naming the action stops the misreading; the summary line
        # under the preview shows the consequence either way.
        tk.Button(btns, text="Add another", width=11,
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
        # Which trace the events are drawn against. Only meaningful for the
        # kinds that name no channel of their own; a threshold source is always
        # shown on the channel it thresholds.
        _an = list((available or {}).get("analogue") or [])
        # Start on the channel being configured. Defaulting to the first
        # analogue channel meant that configuring Channel 3 drew the events
        # over Channel 1, which is a different muscle and answers nothing.
        _pref = channel_name if channel_name in _an else (_an[0] if _an else "")
        self.v_display = tk.StringVar(value=_pref)

        # ── The preview ──────────────────────────────────────────────────────
        _show = tk.Frame(right)
        _show.pack(fill="x", pady=(10, 0))
        tk.Label(_show, text="Show against:", width=12,
                 anchor="w").pack(side="left")
        self._display_dd = ttk.Combobox(
            _show, textvariable=self.v_display, values=_an,
            state="readonly", width=26)
        self._display_dd.pack(side="left")
        tk.Label(_show, fg="grey",
                 text=("  the trace the single-event view is drawn on")).pack(
                     side="left")
        self.v_display.trace_add("write", lambda *a: self._update_preview())

        self.count_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.count_var, fg="#1F3864",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w",
                                                          pady=(8, 2))

        # What every source together will produce. The count line above
        # describes the selected row only, so a second source quietly widening
        # the analysis was invisible until tab 1a was rebuilt.
        self.total_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.total_var, fg="#555555").pack(
            anchor="w", pady=(0, 2))

        _nav = tk.Frame(right)
        _nav.pack(fill="x", pady=(2, 0))
        # First and last are on screen as well as on Home/End: a recording with
        # a hundred and sixty events makes "go to the end" a real request, and
        # a keyboard shortcut nothing announces is not an answer to it.
        tk.Button(_nav, text="◀◀", width=3,
                  command=lambda: self._event_jump("first")).pack(side="left")
        tk.Button(_nav, text="◀", width=3,
                  command=lambda: self._event_step(-1)).pack(side="left",
                                                             padx=(2, 0))
        # Type a number and press Enter. Jumping on every keystroke would send
        # you to event 1 on the way to 16, so the jump is on Enter only.
        self.v_goto = tk.StringVar(value="")
        self._goto_entry = tk.Entry(_nav, textvariable=self.v_goto, width=5,
                                    justify="center")
        self._goto_entry.pack(side="left", padx=(4, 0))
        self._goto_entry.bind("<Return>", lambda e: self._event_goto())
        tk.Button(_nav, text="▶", width=3,
                  command=lambda: self._event_step(1)).pack(side="left",
                                                            padx=(4, 0))
        tk.Button(_nav, text="▶▶", width=3,
                  command=lambda: self._event_jump("last")).pack(
                      side="left", padx=(2, 8))
        self.where_var = tk.StringVar(value="")
        tk.Label(_nav, textvariable=self.where_var).pack(side="left")
        tk.Label(_nav, text="Window (ms):").pack(side="left", padx=(16, 4))
        self.v_window = tk.StringVar(value=f"{float(window_ms or 200.0):g}")
        tk.Entry(_nav, textvariable=self.v_window, width=8).pack(side="left")
        self.v_window.trace_add("write", lambda *a: self._draw_detail())

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        # Overview above, detail below. A whole recording of marks answers
        # "did I catch them all" and nothing else -- at a hundred and seventy
        # events over seventy seconds the trace is a picket fence, and whether
        # any single mark sits on a response is invisible. Both questions are
        # live while a level is being chosen, so both are on screen rather
        # than behind a toggle.
        self.fig = Figure(figsize=(7.2, 4.4), dpi=100)
        self.ax = self.fig.add_subplot(211)
        self.ax_detail = self.fig.add_subplot(212)
        self.fig.subplots_adjust(left=0.08, right=0.99, top=0.95,
                                 bottom=0.12, hspace=0.45)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(2, 0))

        # Events currently drawn, in time order, and where we are among them.
        self._event_times = []
        self._cur_event = 0

        self._bind_edit_traces()

        # Same keys the Inspector uses for the same gesture, so it is learned
        # once rather than twice.
        self.top.bind("<Left>",  lambda e: self._on_nav_key(e, delta=-1))
        self.top.bind("<Right>", lambda e: self._on_nav_key(e, delta=1))
        self.top.bind("<Home>",  lambda e: self._on_nav_key(e, where="first"))
        self.top.bind("<End>",   lambda e: self._on_nav_key(e, where="last"))

        foot = tk.Frame(self.top)
        foot.pack(fill="x", pady=10)
        self._copy_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            foot, variable=self._copy_var,
            text="Apply these sources to every selected channel").pack(
                side="left", padx=(14, 0))
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
            # channel="" means every label, which is precisely what the file
            # did before anyone opened this dialogue. Seeding it with the first
            # label would be a change of behaviour disguised as a default.
            self._sources.append(EventSource(kind="embedded", channel=""))
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
        # Default to a label no existing source already names.
        #
        # Defaulting to embedded[0] meant that on a file whose labels sort as
        # ["Start Task", "Trigger"], setting the first source to Trigger and
        # then pressing Add silently appended a second source pointing at
        # Start Task -- so the type just excluded came straight back, and the
        # only sign was an extra row on tab 1a. Nothing warns about it either:
        # the merge only objects when two sources claim the SAME type.
        embedded = list(self._available.get("embedded") or [])
        taken = {src.channel for src in self._sources
                 if src.kind == "embedded"}
        free = [c for c in embedded if c not in taken]
        self._sources.append(EventSource(
            kind="embedded",
            channel=(free[0] if free else (embedded[0] if embedded else ""))))
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
        if not self._sources:
            # An empty list is the same instruction as one source reading every
            # marker, and leaving the list empty gave a dialogue with no
            # selection, no fields and stale panes. Say it explicitly instead.
            self._sources.append(EventSource(kind="embedded", channel=""))
        self._refresh_list()
        self.listbox.selection_set(min(i, len(self._sources) - 1))
        self._load_selected(force=True)

    # ── editor ───────────────────────────────────────────────────────────────

    def _kind_changed(self):
        # Embedded and threshold share self.v_channel but not their option
        # lists, and a readonly Combobox displays whatever its variable holds
        # even when that is not among its values. Switching kind therefore
        # showed the previous kind's channel over the new kind's dropdown, and
        # _apply_edits wrote that impossible name back to the source -- so the
        # preview stayed empty until the analyst happened to reselect.
        opts = self._options_for(self.kind_var.get())
        if self.v_channel.get() not in opts:
            self.v_channel.set(opts[0] if opts else "")
        self._build_fields()
        self._apply_edits()
        self._update_preview()

    def _options_for(self, kind):
        """The channel names a kind can legitimately name."""
        if kind == "embedded":
            return [ALL_LABELS] + list(self._available.get("embedded") or [])
        if kind == "threshold":
            return list(self._available.get("analogue") or [])
        return []

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
            opts = [ALL_LABELS] + list(self._available.get("embedded") or [])
            row("Events from:", lambda r: ttk.Combobox(
                r, textvariable=self.v_channel, values=opts,
                state="readonly", width=30).pack(side="left"))
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

    def _bind_edit_traces(self):
        """Watch the editable fields once.

        These were added inside _build_fields, which runs on every kind change
        and every selection change -- so the callbacks accumulated, and after a
        few switches one keystroke re-read the channel several times over.
        """
        if getattr(self, "_traces_bound", False):
            return
        for v in (self.v_level, self.v_refrac, self.v_edge, self.v_channel,
                  self.v_start, self.v_period, self.v_count, self.v_label):
            v.trace_add("write", lambda *a: self._on_edit())
        self._traces_bound = True

    def _load_selected(self, force=False):
        i = self._selected_index()
        if i is None:
            for w in self.fields.winfo_children():
                w.destroy()
            self._clear_panes(force=force)
            return
        src = self._sources[i]
        # A different source has a different event list, so the position in the
        # old one means nothing.
        self._cur_event = 0
        self.kind_var.set(src.kind)
        self.v_channel.set(src.channel if src.channel else
                           (ALL_LABELS if src.kind == "embedded" else ""))
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
        _ch = self.v_channel.get()
        src.channel = "" if _ch == ALL_LABELS else _ch
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

    def _clear_panes(self, force=False):
        """Empty every pane and label together.

        The early returns used to clear only the overview, so the summary line,
        the position label and the single-event view kept describing the
        previous selection -- which read as the dialogue not updating until
        something was clicked.
        """
        self.ax.clear()
        self.ax_detail.clear()
        for _ax in (self.ax, self.ax_detail):
            _ax.set_xticks([]); _ax.set_yticks([])
        self._event_times = []
        self._cur_event = 0
        self._detail_source = (None, None, None)
        self.count_var.set("")
        self.where_var.set("")
        self._sync_goto()
        self._refresh_total()
        self.canvas.draw() if force else self.canvas.draw_idle()

    def _refresh_total(self):
        """Summarise every source at once, in the words tab 1a will use."""
        try:
            from .io import extract_events
            events, warnings = extract_events(self._file_path, self._sources)
        except Exception:
            self.total_var.set("")
            return
        if not events:
            self.total_var.set("These sources produce no events.")
            return
        parts = ", ".join(f"{k} ({len(v)})" for k, v in sorted(events.items()))
        note = f"All sources together: {parts}"
        if len(events) > 1:
            note += "   \u2014 one row each on tab 1a"
        for w in warnings:
            note += f"   \u26a0 {w}"
        self.total_var.set(note)

    # ── detail view ──────────────────────────────────────────────────────────

    def _detail_window_ms(self):
        """Width of the detail view, ignoring a half-typed number."""
        try:
            w = float(self.v_window.get())
        except (TypeError, ValueError):
            return None
        return w if w > 0 else None

    def _draw_detail(self, redraw=True):
        """Draw the event currently stepped to, from raw samples.

        Raw, not a slice of the decimated overview arrays: decimate_for_preview
        collapses the whole recording to a few thousand min/max columns, so a
        two-hundred-millisecond window covers a handful of them and would draw
        a blocky trace that still looks like data.
        """
        ax = self.ax_detail
        ax.clear()
        shown, sig, fs = getattr(self, "_detail_source", (None, None, None))
        n = len(self._event_times)
        if not n or sig is None:
            self.where_var.set("")
            self._sync_goto()
            ax.text(0.5, 0.5, "No events to step through.",
                    ha="center", va="center", color="0.5",
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            if redraw:
                self.canvas.draw_idle()
            return

        i = max(0, min(self._cur_event, n - 1))
        self._cur_event = i
        t0 = float(self._event_times[i])
        half = (self._detail_window_ms() or 200.0) / 2000.0
        a = max(0, min(len(sig), int((t0 - half) * fs)))
        b = max(0, min(len(sig), int((t0 + half) * fs)))
        if b <= a:
            # An event can sit outside the waveform -- some formats carry
            # markers past the end of the data they are stored with. Saying so
            # is better than blank axes under a position label that still
            # claims to be showing something.
            self._sync_goto()
            self.where_var.set(f"of {n}   ·   {t0:.3f} s")
            ax.text(0.5, 0.5,
                    f"Event {i + 1} at {t0:.3f} s lies outside the recording "
                    f"({len(sig) / float(fs):.3f} s).",
                    ha="center", va="center", color="#B03A2E",
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            if redraw:
                self.canvas.draw_idle()
            return

        seg = sig[a:b]
        t = (np.arange(a, b) / float(fs) - t0) * 1000.0     # ms about the event
        ax.plot(t, seg, color="0.25", lw=0.8)
        ax.axvline(0.0, color="#1F3864", lw=1.4)
        ax.set_xlabel("Time about the event (ms)", fontsize=8)
        ax.set_title(f"This event — {shown}", fontsize=8, loc="left",
                     color="0.35")
        ax.tick_params(labelsize=8)

        # Show on the overview where this window sits, so stepping through a
        # hundred events also says how far along the recording you are.
        try:
            self.ax.axvspan(t0 - half, t0 + half, color="#1F3864", alpha=0.18)
        except Exception:
            pass

        self._sync_goto()
        self.where_var.set(f"of {n}   ·   {t0:.3f} s"
                           + (f"   ·   {shown}" if shown else ""))
        if redraw:
            self.canvas.draw_idle()

    def _event_step(self, delta):
        if not self._event_times:
            return
        self._cur_event = max(
            0, min(self._cur_event + int(delta), len(self._event_times) - 1))
        self._update_preview()

    def _event_goto(self):
        """Jump to a typed event number, counted as the label shows them.

        Out of range is clamped rather than refused: on a recording of 162
        events, someone typing 200 means the last one, and an error dialogue
        would be a worse answer than going there.
        """
        if not self._event_times:
            return
        try:
            n = int(float(self.v_goto.get()))
        except (TypeError, ValueError):
            self._sync_goto()
            return
        self._cur_event = max(0, min(n - 1, len(self._event_times) - 1))
        self._update_preview()

    def _sync_goto(self):
        """Show the current position, unless it is being typed into.

        Overwriting the box mid-number would fight the typing, and the preview
        refreshes on a level-box trace, so this can be called at any moment.
        """
        try:
            if self.top.focus_get() is self._goto_entry:
                return
        except Exception:
            pass
        self.v_goto.set(str(self._cur_event + 1) if self._event_times else "")

    def _event_jump(self, where):
        if not self._event_times:
            return
        self._cur_event = 0 if where == "first" else len(self._event_times) - 1
        self._update_preview()

    def _on_nav_key(self, event, delta=None, where=None):
        """Arrow keys step events -- unless something is being typed into.

        The channel and edge dropdowns consume Left and Right when focused, so
        an unconditional binding would change the source instead of the event.
        """
        w = self.top.focus_get()
        if isinstance(w, (tk.Entry, tk.Text, ttk.Combobox, tk.Listbox,
                          ttk.Entry)):
            return
        if where is not None:
            self._event_jump(where)
        else:
            self._event_step(delta)
        return "break"

    def _resolve_events(self, src):
        """The events this source produces, via the same call the run makes.

        io.extract_events is what pipeline_load_file uses, so what is drawn
        here is what will be analysed. Computing them separately would let the
        preview and the run agree today and diverge on the next change.
        """
        try:
            from .io import extract_events
            events, warnings = extract_events(self._file_path, [src])
            return events, warnings, None
        except Exception as exc:                # noqa: BLE001 — shown to user
            return {}, [], f"{type(exc).__name__}: {exc}"

    def _overview_channel(self, src):
        """The trace the whole file is drawn on.

        For a threshold source this is the channel being thresholded: the
        overview is where the level is set, and a level means nothing drawn
        over a different signal. Every other kind has no channel of its own,
        so it uses the chosen one.
        """
        if src.kind == "threshold" and src.channel:
            return src.channel
        return self._display_channel()

    def _display_channel(self):
        """The trace the detail view is drawn on -- always the chosen one.

        Returning src.channel here for threshold sources made the dropdown
        inert in the case it exists for. Setting a level is one question ("am
        I detecting the pulses?") and it is answered on the trigger channel;
        whether an event sits on a response is another, and it can only be
        answered somewhere else. The two axes answer one each.
        """
        want = self.v_display.get()
        opts = list(self._available.get("analogue") or [])
        return want if want in opts else (opts[0] if opts else "")

    def _update_preview(self, force=False):
        i = self._selected_index()
        self.ax.clear()
        if i is None:
            self._clear_panes(force=force)
            return
        src = self._sources[i]

        # Every kind draws a signal with its events over it. The old
        # placeholder ("the file's own events need no level") answered a
        # question nobody asked: the level is not the point, whether the events
        # land on responses is, and that is exactly what a preview can show.
        shown = self._overview_channel(src)
        if not shown:
            self._clear_panes()
            self.ax.text(0.5, 0.5, "This file has no channel to draw against.",
                         ha="center", va="center", color="0.5",
                         transform=self.ax.transAxes)
            self.canvas.draw() if force else self.canvas.draw_idle()
            return

        sig, fs = self._channel_data(shown)
        if sig is None:
            self._clear_panes()
            self.ax.text(0.5, 0.5, f"Could not read {shown}",
                         ha="center", va="center", color="#B03A2E",
                         transform=self.ax.transAxes)
            self.canvas.draw() if force else self.canvas.draw_idle()
            return

        t, lo, hi = decimate_for_preview(sig, fs)
        self.ax.fill_between(t, lo, hi, color="0.35", linewidth=0)
        self.ax.set_xlabel("Time (s)", fontsize=8)
        self.ax.set_title(f"Whole recording — {shown}", fontsize=8,
                          loc="left", color="0.35")
        self.ax.tick_params(labelsize=8)

        if src.kind == "threshold":
            # Detected here rather than through extract_events so the level box
            # stays responsive while it is being typed into: this reads the
            # already-cached array instead of reopening the recording.
            times = detect_threshold_crossings(
                sig, fs, src.level, src.edge, src.refractory_ms)
            self.ax.axhline(src.level, color="#B03A2E", lw=1.2)
            if times:
                self.ax.plot(times, [src.level] * len(times), "|",
                             color="#1F3864", markersize=10,
                             markeredgewidth=1.4)
            # "detected" here and "against" below, because the difference is
            # real: a crossing is found by this dialogue, whereas a comment was
            # already in the file and is only being drawn somewhere it can be
            # judged.
            summary = f"{len(times)} event(s) detected on {src.channel}"
            problem = None
        else:
            events, warnings, problem = self._resolve_events(src)
            times = sorted(t for v in events.values() for t in v)
            for _t in times:
                self.ax.axvline(_t, color="#1F3864", lw=0.8, alpha=0.8)
            summary = (f"{len(times)} event(s) against {shown}"
                       + (f"  ·  {', '.join(sorted(events))}" if events else ""))
            for _w in warnings:
                summary += f"  ·  {_w}"

        # Kept so the detail view and the arrow keys have a list to walk,
        # and clamped rather than reset: adjusting a level should not throw
        # away the place you were looking at.
        self._refresh_total()
        self._event_times = list(times)
        self._cur_event = min(self._cur_event, max(0, len(times) - 1))
        # The detail view reads the chosen channel, which for a threshold
        # source is deliberately not the one being thresholded: seeing the
        # trigger again at higher zoom says nothing the overview did not.
        _det = self._display_channel()
        if _det and _det != shown:
            _dsig, _dfs = self._channel_data(_det)
            self._detail_source = ((_det, _dsig, _dfs) if _dsig is not None
                                   else (shown, sig, fs))
        else:
            self._detail_source = (shown, sig, fs)
        self._draw_detail(redraw=False)

        if problem:
            self.count_var.set(problem)
        else:
            self.count_var.set(
                summary
                + (f"  ·  first at {times[0]:.3f} s, last at {times[-1]:.3f} s"
                   if times else "  ·  nothing found with these settings"))
        self.canvas.draw() if force else self.canvas.draw_idle()

    # ── finish ───────────────────────────────────────────────────────────────

    def _ok(self):
        self._apply_edits()
        self.copy_to_all = bool(self._copy_var.get())
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
