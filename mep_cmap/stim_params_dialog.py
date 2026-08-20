"""
mep_cmap.stim_params_dialog
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Session-level editor for stimulation parameter sets.

A recording says a pulse fired on code ``G``. It does not say that ``G`` was a
TMS pulse at 120% of a resting motor threshold while ``A`` was a peripheral
M-wave at 45 mA. This is where that is stated, once per session, and referenced
by every file that used it.

Why session level rather than a panel on each file: a threshold is not a
property of a recording. Fifteen files sharing one MEP protocol would otherwise
state it fifteen times, and correcting a mis-recorded rMT would mean fifteen
edits with nothing linking them.

The rules live in :mod:`mep_cmap.stim_params`, which holds no Tk. This module
is only the window.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from . import stim_params as sp

_BG     = "#f5f5f5"
_FG     = "#212121"
_ACCENT = "#2196F3"


class StimParamSetsDialog:
    """Modal. ``result`` is a list of StimParamSet, or None if cancelled."""

    def __init__(self, master, schema, sets=(), on_complete=None):
        self.schema = schema
        self.on_complete = on_complete
        self.result = None
        self._sets = [sp.StimParamSet(s.name, s.nibs_type, dict(s.values),
                                      s.position) for s in (sets or [])]
        self._current = None            # index being edited, or None
        self._field_vars = {}

        self.top = tk.Toplevel(master)
        self.top.title("BIDS-ify — Stimulation parameter sets")
        self.top.configure(bg=_BG)
        self.top.transient(master)
        self.top.grab_set()

        tk.Label(
            self.top, bg=_BG, fg=_FG, justify="left", wraplength=760,
            text=("One set per protocol you delivered, not one per file. Each "
                  "becomes a row of *_nibs.tsv, and each stim code in a "
                  "recording points at one of them.")
        ).pack(anchor="w", padx=14, pady=(12, 8))

        body = tk.Frame(self.top, bg=_BG)
        body.pack(fill="both", expand=True, padx=14)

        self._build_list(body)
        self._build_editor(body)
        self._build_footer()

        self._refresh_list()
        if self._sets:
            self._select(0)

        self.top.bind("<Escape>", lambda _e: self._cancel())
        self.top.protocol("WM_DELETE_WINDOW", self._cancel)

    # ── left: the set list ───────────────────────────────────────────────────

    def _build_list(self, parent):
        left = tk.Frame(parent, bg=_BG)
        left.pack(side="left", fill="y", padx=(0, 12))

        tk.Label(left, text="Parameter sets", bg=_BG, fg=_FG,
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        holder = tk.Frame(left, bg=_BG)
        holder.pack(fill="both", expand=True)
        bar = ttk.Scrollbar(holder, orient="vertical")
        self._list = tk.Listbox(holder, height=14, width=26,
                                exportselection=False, yscrollcommand=bar.set)
        bar.config(command=self._list.yview)
        bar.pack(side="right", fill="y")
        self._list.pack(side="left", fill="both", expand=True)
        self._list.bind("<<ListboxSelect>>", self._on_select)

        btns = tk.Frame(left, bg=_BG)
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="Add", width=8,
                   command=self._add).pack(side="left")
        ttk.Button(btns, text="Duplicate", width=10,
                   command=self._duplicate).pack(side="left", padx=(4, 0))
        ttk.Button(btns, text="Remove", width=8,
                   command=self._remove).pack(side="left", padx=(4, 0))

    # ── right: the selected set ──────────────────────────────────────────────

    def _build_editor(self, parent):
        right = tk.Frame(parent, bg=_BG)
        right.pack(side="left", fill="both", expand=True)

        head = tk.Frame(right, bg=_BG)
        head.pack(fill="x")

        tk.Label(head, text="Name", bg=_BG, fg=_FG).grid(row=0, column=0, sticky="w")
        self._name_var = tk.StringVar()
        e = ttk.Entry(head, textvariable=self._name_var, width=24)
        e.grid(row=0, column=1, sticky="w", padx=(8, 16))
        # Committed on focus-out as well as on selection change: a name typed
        # and then Saved without leaving the box would otherwise be lost.
        e.bind("<FocusOut>", lambda _e: self._commit())

        tk.Label(head, text="Type", bg=_BG, fg=_FG).grid(row=0, column=2, sticky="w")
        self._type_var = tk.StringVar(value="TMS")
        t = ttk.Combobox(head, textvariable=self._type_var, width=8,
                         state="readonly", values=list(sp.NIBS_TYPES))
        t.grid(row=0, column=3, sticky="w", padx=(8, 16))
        t.bind("<<ComboboxSelected>>", lambda _e: self._on_type_changed())

        tk.Label(head, text="Position", bg=_BG, fg=_FG).grid(row=0, column=4, sticky="w")
        self._pos_var = tk.StringVar()
        p = ttk.Entry(head, textvariable=self._pos_var, width=20)
        p.grid(row=0, column=5, sticky="w", padx=(8, 0))
        p.bind("<FocusOut>", lambda _e: self._commit())

        self._fields_holder = tk.Frame(right, bg=_BG)
        self._fields_holder.pack(fill="both", expand=True, pady=(10, 0))

    def _rebuild_fields(self):
        for w in self._fields_holder.winfo_children():
            w.destroy()
        self._field_vars = {}
        if self._current is None:
            tk.Label(self._fields_holder, bg=_BG, fg="#666",
                     text="Add a parameter set to begin.").pack(anchor="w")
            return

        s = self._sets[self._current]
        canvas = tk.Canvas(self._fields_holder, bg=_BG, highlightthickness=0,
                           height=300)
        bar = ttk.Scrollbar(self._fields_holder, orient="vertical",
                            command=canvas.yview)
        inner = tk.Frame(canvas, bg=_BG)
        inner.bind("<Configure>",
                   lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Only parameter_set scope. Session fields (the stimulator, the coil,
        # the measured threshold) belong to shared defaults, and repeating them
        # per set is what this design exists to avoid.
        flds = self.schema.fields_for(s.nibs_type, scope="parameter_set")
        # ...except where they genuinely differ per protocol, which is why they
        # are offered below as overrides rather than not at all. M-waves on a
        # Digitimer and MEPs on a Magstim in one recording is a device that
        # changes with the protocol, and v6.3 expects exactly that: stimulator_id
        # and nibs_element_id are columns of *_nibs.tsv pointing into the
        # StimulatorSet and ElementSet arrays, so the device is referenced per
        # row rather than fixed per file. Blank means "use the shared default".
        overrides = [f for f in self.schema.fields_for(s.nibs_type,
                                                       scope="session")
                     if not f.legacy]
        if not flds and not overrides:
            tk.Label(inner, bg=_BG, fg="#666",
                     text=f"No parameter fields apply to {s.nibs_type}.").pack(anchor="w")
            return

        r = 0
        for fld in flds:
            self._add_field(inner, fld, s, r)
            r += 1

        if overrides:
            tk.Label(inner, bg=_BG, fg=_ACCENT, anchor="w",
                     font=("TkDefaultFont", 9, "bold"),
                     text="Device and dosing for this protocol").grid(
                         row=r, column=0, columnspan=2, sticky="w", pady=(12, 0))
            r += 1
            tk.Label(inner, bg=_BG, fg="#666", anchor="w", justify="left",
                     wraplength=420,
                     text=("Leave blank to use the shared default. Set only "
                           "where this protocol differs \u2014 a different "
                           "stimulator, a different coil, a threshold measured "
                           "for this target.")).grid(
                         row=r, column=0, columnspan=2, sticky="w", pady=(0, 4))
            r += 1
            for fld in overrides:
                self._add_field(inner, fld, s, r)
                r += 1
        inner.columnconfigure(1, weight=1)

    def _add_field(self, parent, fld, s, row):
        label = fld.key + (f"  [{fld.units}]" if fld.units else "")
        tk.Label(parent, text=label, bg=_BG, fg=_FG, anchor="w").grid(
            row=row, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=str(s.values.get(fld.key, "") or ""))
        self._field_vars[fld.key] = var
        if fld.enum:
            w = ttk.Combobox(parent, textvariable=var, width=34,
                             state="readonly", values=[""] + list(fld.enum))
        else:
            w = ttk.Entry(parent, textvariable=var, width=36)
        w.grid(row=row, column=1, sticky="w", padx=(10, 0), pady=2)

    # ── state movement ───────────────────────────────────────────────────────

    def _commit(self):
        """Write the editor back into the selected set.

        Called before every selection change, type change and save. Without it,
        values typed into the last set edited are lost when the analyst clicks
        another one -- the classic form-editor fault, and silent.
        """
        if self._current is None:
            return
        s = self._sets[self._current]
        name = sp.sanitise_name(self._name_var.get()) or s.name
        vals = dict(s.values)
        for key, var in self._field_vars.items():
            raw = var.get().strip()
            if raw:
                vals[key] = raw
            else:
                vals.pop(key, None)
        self._sets[self._current] = sp.StimParamSet(
            name=name,
            nibs_type=self._type_var.get() or s.nibs_type,
            values=vals,
            position=self._pos_var.get().strip())

    def _refresh_list(self):
        keep = self._current
        self._list.delete(0, "end")
        for s in self._sets:
            self._list.insert("end", f"{s.name}  ({s.nibs_type})")
        if keep is not None and 0 <= keep < len(self._sets):
            self._list.selection_clear(0, "end")
            self._list.selection_set(keep)

    def _select(self, idx):
        # None reaches here from _on_type_changed when nothing is selected yet:
        # the type combobox exists before any set does, so changing it on an
        # empty dialogue used to compare None against an int and raise.
        if idx is None or not (0 <= idx < len(self._sets)):
            self._current = None
            self._rebuild_fields()
            return
        self._current = idx
        s = self._sets[idx]
        self._name_var.set(s.name)
        self._type_var.set(s.nibs_type)
        self._pos_var.set(s.position)
        self._rebuild_fields()
        self._list.selection_clear(0, "end")
        self._list.selection_set(idx)

    def _on_select(self, _e=None):
        sel = self._list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if idx == self._current:
            return
        self._commit()
        self._refresh_list()
        self._select(idx)

    def _on_type_changed(self):
        """A new type shows a different field list, so commit first.

        Values already entered are KEPT rather than cleared: switching TMS to
        PNS by mistake and back must not silently discard an intensity. Fields
        that no longer apply are simply not shown, and are dropped at write
        time by the same modality filter.
        """
        self._commit()
        self._refresh_list()
        self._select(self._current)

    # ── list actions ─────────────────────────────────────────────────────────

    def _add(self):
        self._commit()
        names = {s.name for s in self._sets}
        base, n = "set", 1
        while f"{base}_{n}" in names:
            n += 1
        self._sets.append(sp.StimParamSet(f"{base}_{n}"))
        self._refresh_list()
        self._select(len(self._sets) - 1)

    def _duplicate(self):
        """Because a recruitment curve is one protocol at several intensities,
        and retyping the rest of it invites a discrepancy."""
        if self._current is None:
            return
        self._commit()
        s = self._sets[self._current]
        new_name = sp.deduplicate_names([x.name for x in self._sets] + [s.name])[-1]
        self._sets.append(sp.StimParamSet(new_name, s.nibs_type,
                                          dict(s.values), s.position))
        self._refresh_list()
        self._select(len(self._sets) - 1)

    def _remove(self):
        if self._current is None:
            return
        s = self._sets[self._current]
        if not messagebox.askyesno(
                "Remove parameter set",
                f"Remove {s.name!r}?\n\nAny stim code pointing at it will "
                f"need reassigning before that file can be converted.",
                parent=self.top):
            return
        del self._sets[self._current]
        self._current = None
        self._refresh_list()
        self._select(min(len(self._sets) - 1, 0) if self._sets else -1)

    # ── footer ───────────────────────────────────────────────────────────────

    def _build_footer(self):
        foot = tk.Frame(self.top, bg=_BG)
        foot.pack(fill="x", padx=14, pady=12)
        self._status = tk.Label(foot, bg=_BG, fg="#666", anchor="w",
                                justify="left", wraplength=520)
        self._status.pack(side="left", fill="x", expand=True)
        ttk.Button(foot, text="Cancel", command=self._cancel).pack(side="right")
        ttk.Button(foot, text="Save", command=self._save).pack(side="right", padx=(0, 6))
        ttk.Button(foot, text="Validate",
                   command=self._validate).pack(side="right", padx=(0, 6))

    def _problems(self):
        self._commit()
        return sp.validate(self._sets)

    def _validate(self):
        errs = self._problems()
        self._refresh_list()
        if errs:
            self._status.config(fg="#b00020", text="  ".join(errs[:3]))
        else:
            self._status.config(fg="#2e7d32",
                                text=f"{len(self._sets)} parameter set(s) — OK")

    def _save(self):
        errs = self._problems()
        if errs:
            self._status.config(fg="#b00020", text="  ".join(errs[:3]))
            messagebox.showwarning("Parameter sets", "\n".join(errs),
                                   parent=self.top)
            return
        self.result = list(self._sets)
        cb, res = self.on_complete, self.result
        self.top.destroy()
        if cb:
            cb(res)

    def _cancel(self):
        self.result = None
        cb = self.on_complete
        self.top.destroy()
        if cb:
            cb(None)
