"""
mep_cmap.bidsify_tab
~~~~~~~~~~~~~~~~~~~~
BidsifyTabMixin — the BIDS-ify worklist tab for TMSAnalysisApp.

Model (see bidsify_state.py): shared session defaults ⊕ per-file overrides,
persisted to bidsify_state.json so the worklist is resumable across sessions.

Tab layout, top to bottom:
  • Shared defaults panel — edit the session-constant metadata once; choose the
    output rawdata folder.
  • Status tree — one row per queued file, colour-coded not_started / incomplete
    / ready / converted, sharing the analysis queue's file list.
  • Per-file editor (on double-click) — file-scoped fields foregrounded,
    session defaults shown inherited-but-overridable.
  • Convert ready files — runs the (idempotent) writer on everything Ready.

This mixin assumes the host provides: self.root, self.msg_q, self.notebook,
self._dataset / self._get_or_create_dataset(), self.derivatives_path,
self._rawdata_path, self._parse_bids_from_filename(), self._show_bidsify_preview(),
self._bidsify_done_gui(), self.log(). app.py wires the tab in and adds the
msg_q "bidsify-convert-done" branch that calls self._bidsify_convert_done().
"""

import os
import datetime
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .bids import StudyMetadata
from .bidsify import DatasetLayout, BidsifyItem, plan_bidsify, execute_plan
from .bidsify_dialog import BidsifyDialog
from .bidsify_state import (BidsifyState, STATUS_LABELS, STATUS_COLOURS,
                            STATUS_NOT_STARTED, STATUS_INCOMPLETE,
                            STATUS_READY, STATUS_CONVERTED)

_ACCENT = "#2196F3"


# ── Pure helper (unit-testable) ───────────────────────────────────────────────
def compute_overrides(file_vals: dict, session_vals: dict, defaults: dict) -> dict:
    """
    Build the minimal per-file override set.

      • file-scoped fields   → stored if non-blank
      • session-scoped fields→ stored only if non-blank AND different from the
                               shared default (so files don't just duplicate defaults)
    """
    ov = {}
    for k, v in (file_vals or {}).items():
        if v is not None and str(v).strip() != "":
            ov[k] = v
    for k, v in (session_vals or {}).items():
        if v is None or str(v).strip() == "":
            continue
        if str(v) != str(defaults.get(k, "") or ""):
            ov[k] = v
    return ov


class BidsifyTabMixin:
    # ---- state access ---------------------------------------------------------
    def _bidsify_state_root(self) -> str:
        """
        Where bidsify_state.json lives: the study root (parent of derivatives/,
        sibling to rawdata/), so its relative-path keys line up with the rawdata
        files. Falls back to the derivatives/rawdata folder, then home.
        """
        ds = self._get_or_create_dataset()
        root = ""
        try:
            dd = ds._deriv_dir                      # derivatives/ folder
            if dd and os.path.isabs(dd):
                root = os.path.dirname(dd)          # study root
        except Exception:
            root = ""
        if not root or not os.path.isabs(root):
            root = (getattr(ds, "derivatives_root", "") or self.derivatives_path.get()
                    or self._rawdata_path.get() or os.path.expanduser("~"))
        return root

    def _get_bidsify_state(self) -> BidsifyState:
        root = self._bidsify_state_root()
        st = getattr(self, "_bidsify_state", None)
        if st is None or st.root != root:
            self._bidsify_state = BidsifyState.load_or_create(root)
        return self._bidsify_state

    def _bidsify_all_paths(self) -> list:
        """Files shared with the analysis queue (dataset order)."""
        ds = self._dataset
        if ds is None:
            return []
        return [fe.path for fe in ds.files if os.path.isfile(fe.path)]

    # ---- tab construction -----------------------------------------------------
    def _build_bidsify_tab(self, parent: tk.Frame):
        from .bids_schema import load_schema
        self._bidsify_schema = load_schema()

        # ── Shared defaults panel ─────────────────────────────────────────────
        top = tk.LabelFrame(parent, text="Shared stimulation metadata  (applies to every file)",
                            padx=8, pady=6)
        top.pack(fill="x", padx=10, pady=(10, 4))

        row1 = tk.Frame(top); row1.pack(fill="x", pady=(0, 4))
        tk.Button(row1, text="✎  Edit shared defaults…",
                  font=("TkDefaultFont", 9, "bold"),
                  command=self._bidsify_edit_defaults).pack(side="left")
        self._bidsify_defaults_lbl = tk.Label(row1, text="Not set yet", fg="#888")
        self._bidsify_defaults_lbl.pack(side="left", padx=(10, 0))

        row2 = tk.Frame(top); row2.pack(fill="x", pady=(2, 0))
        tk.Label(row2, text="Output rawdata folder:", width=20, anchor="w").pack(side="left")
        self._bidsify_rawroot_var = tk.StringVar()
        tk.Button(row2, text="Browse…",
                  command=self._bidsify_browse_rawroot).pack(side="right", padx=(4, 0))
        tk.Entry(row2, textvariable=self._bidsify_rawroot_var, state="readonly",
                 fg="#555").pack(side="left", fill="x", expand=True, padx=(4, 4))

        # ── File status tree ──────────────────────────────────────────────────
        mid = tk.LabelFrame(parent,
            text="Files  (double-click to review / edit metadata)", padx=6, pady=4)
        mid.pack(fill="both", expand=True, padx=10, pady=(0, 4))

        bar = tk.Frame(mid); bar.pack(fill="x", pady=(0, 4))
        tk.Button(bar, text="🔄 Refresh", command=self._bidsify_tab_refresh).pack(side="left", padx=(0, 4))
        tk.Button(bar, text="✎ Review selected…", command=self._bidsify_review_selected).pack(side="left", padx=(0, 4))
        tk.Button(bar, text="✓ Accept defaults for selected",
                  command=self._bidsify_accept_defaults_selected).pack(side="left", padx=(0, 4))
        tk.Button(bar, text="▶  Convert ready files", command=self._bidsify_convert_ready,
                  bg="#5cb85c", fg="white",
                  font=("TkDefaultFont", 9, "bold")).pack(side="right", padx=(0, 4))

        cols = ("status", "sub", "ses", "cond", "modality", "missing", "path")
        tree_wrap = tk.Frame(mid); tree_wrap.pack(fill="both", expand=True)
        self._bidsify_tree = ttk.Treeview(tree_wrap, columns=cols, show="headings",
                                          height=14, selectmode="extended")
        for c, txt, w in [("status", "Status", 110), ("sub", "Subject", 90),
                          ("ses", "Session", 70), ("cond", "Condition/Task", 200),
                          ("modality", "Modality", 80), ("missing", "Missing required", 180),
                          ("path", "Path", 460)]:
            self._bidsify_tree.heading(c, text=txt)
            self._bidsify_tree.column(c, width=w, stretch=(c == "path"), minwidth=40)
        vs = ttk.Scrollbar(tree_wrap, orient="vertical", command=self._bidsify_tree.yview)
        hs = ttk.Scrollbar(tree_wrap, orient="horizontal", command=self._bidsify_tree.xview)
        self._bidsify_tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self._bidsify_tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        tree_wrap.grid_rowconfigure(0, weight=1)
        tree_wrap.grid_columnconfigure(0, weight=1)
        for status, colour in STATUS_COLOURS.items():
            self._bidsify_tree.tag_configure(status, foreground=colour)
        self._bidsify_tree.bind("<Double-1>", lambda _e: self._bidsify_review_selected())

        self._bidsify_counts_var = tk.StringVar(value="No files")
        tk.Label(parent, textvariable=self._bidsify_counts_var, fg="grey",
                 anchor="w").pack(fill="x", padx=10, pady=(0, 4))

        self._bidsify_iid_to_path = {}

    # ---- refresh --------------------------------------------------------------
    def _bidsify_tab_refresh(self):
        if not hasattr(self, "_bidsify_tree"):
            return
        state = self._get_bidsify_state()
        schema = self._bidsify_schema

        # sync the small header widgets from state
        self._bidsify_rawroot_var.set(state.rawdata_root or self._rawdata_path.get() or "")
        if state.defaults or state.modality:
            n = len(state.defaults)
            self._bidsify_defaults_lbl.config(
                text=f"{state.modality} · {n} field(s) set · {state.container}+ · marker '{state.marker_name}'",
                fg="#333")
        else:
            self._bidsify_defaults_lbl.config(text="Not set yet", fg="#888")

        self._bidsify_tree.delete(*self._bidsify_tree.get_children())
        self._bidsify_iid_to_path = {}
        paths = self._bidsify_all_paths()
        for p in paths:
            parsed = self._parse_bids_from_filename(p)
            status = state.status(p, schema)
            missing = ", ".join(state.missing_required(p, schema)) if status == STATUS_INCOMPLETE else ""
            cond = parsed.get("acq") or parsed.get("task") or ""
            iid = self._bidsify_tree.insert(
                "", "end",
                values=(STATUS_LABELS[status], parsed.get("participant_id", ""),
                        parsed.get("session", ""), cond, state.modality, missing, p),
                tags=(status,))
            self._bidsify_iid_to_path[iid] = p

        counts = state.counts(paths, schema)
        self._bidsify_counts_var.set(
            f"{len(paths)} file(s):  "
            f"{counts[STATUS_NOT_STARTED]} not started · "
            f"{counts[STATUS_INCOMPLETE]} incomplete · "
            f"{counts[STATUS_READY]} ready · "
            f"{counts[STATUS_CONVERTED]} converted")

    def _bidsify_selected_paths(self) -> list:
        return [self._bidsify_iid_to_path[i]
                for i in self._bidsify_tree.selection()
                if i in self._bidsify_iid_to_path]

    # ---- shared defaults ------------------------------------------------------
    def _bidsify_edit_defaults(self):
        state = self._get_bidsify_state()
        defaults = {
            "modality": state.modality,
            "container": state.container,
            "powerline_hz": state.powerline_hz,
            "marker_name": state.marker_name,
            "sidecar_values": state.defaults,
        }
        BidsifyDialog(self.root, schema=self._bidsify_schema,
                      on_complete=self._bidsify_defaults_saved, defaults=defaults)

    def _bidsify_defaults_saved(self, result):
        if result is None:
            return
        state = self._get_bidsify_state()
        # keep only session-scoped values as shared defaults; file-scoped values a
        # user happened to type here would wrongly apply to all files.
        session_keys = {f.key for f in self._bidsify_schema.fields_for(
            result.modality, scope="session")}
        shared = {k: v for k, v in result.sidecar_values.items()
                  if k in session_keys or k == "StimulationModality"}
        state.set_defaults(result.modality, shared,
                           result.container, result.powerline_hz, result.marker_name)
        state.save()
        self._bidsify_tab_refresh()

    def _bidsify_browse_rawroot(self):
        state = self._get_bidsify_state()
        init = state.rawdata_root or self._rawdata_path.get() or os.path.expanduser("~")
        d = filedialog.askdirectory(title="Choose the BIDS rawdata output folder",
                                    initialdir=init)
        if d:
            state.rawdata_root = d
            state.save()
            self._bidsify_rawroot_var.set(d)

    # ---- per-file editor ------------------------------------------------------
    def _bidsify_review_selected(self):
        paths = self._bidsify_selected_paths()
        if not paths:
            messagebox.showinfo("BIDS-ify", "Select a file to review.", parent=self.root)
            return
        self._bidsify_open_file_editor(paths[0])

    def _bidsify_accept_defaults_selected(self):
        """Mark selected files reviewed with no overrides (inherit defaults as-is)."""
        paths = self._bidsify_selected_paths()
        if not paths:
            return
        state = self._get_bidsify_state()
        for p in paths:
            state.set_overrides(p, {}, reviewed=True)
        state.save()
        self._bidsify_tab_refresh()

    def _bidsify_open_file_editor(self, path):
        state = self._get_bidsify_state()
        schema = self._bidsify_schema
        modality = state.modality
        rec = state.record_for(path, create=False)
        existing = dict(rec.overrides) if rec else {}

        win = tk.Toplevel(self.root)
        win.title(f"Review — {os.path.basename(path)}")
        win.configure(bg="#f5f5f5")
        win.grab_set()

        tk.Label(win, text=os.path.basename(path), bg="#f5f5f5",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 0))
        tk.Label(win, text=f"Modality: {modality}   (set in shared defaults)",
                 bg="#f5f5f5", fg="#777").pack(anchor="w", padx=12, pady=(0, 6))

        file_vars, session_vars = {}, {}

        def _add_field(parent, fld, value, r):
            lbl = f"{fld.key}{' *' if fld.level == 'required' else ''}" \
                  + (f"  [{fld.units}]" if fld.units else "")
            tk.Label(parent, text=lbl, bg=parent["bg"], anchor="w").grid(
                row=r, column=0, sticky="w", padx=(4, 8), pady=1)
            var = tk.StringVar(value=value or "")
            if fld.enum:
                w = ttk.Combobox(parent, textvariable=var, values=[""] + list(fld.enum),
                                 state="readonly")
            else:
                w = tk.Entry(parent, textvariable=var)
            w.grid(row=r, column=1, sticky="ew", padx=(0, 4), pady=1)
            parent.columnconfigure(1, weight=1)
            return var

        # This recording (file-scoped) — foregrounded
        f1 = tk.LabelFrame(win, text="This recording", bg="#f5f5f5", padx=8, pady=6)
        f1.pack(fill="x", padx=12, pady=4)
        for r, fld in enumerate(schema.fields_for(modality, scope="file")):
            file_vars[fld.key] = _add_field(f1, fld, existing.get(fld.key, ""), r)

        # Inherited defaults (session-scoped) — collapsible override section
        show_over = tk.BooleanVar(value=False)
        f2 = tk.LabelFrame(win, text="Shared defaults (inherited — override only if this file differs)",
                           bg="#f5f5f5", padx=8, pady=6)
        holder = tk.Frame(f2, bg="#f5f5f5")

        def _toggle():
            (holder.pack(fill="x") if show_over.get() else holder.pack_forget())
        tk.Checkbutton(f2, text="Show / override inherited fields", variable=show_over,
                       bg="#f5f5f5", command=_toggle).pack(anchor="w")
        for r, fld in enumerate(schema.fields_for(modality, scope="session")):
            inherited = existing.get(fld.key, state.defaults.get(fld.key, ""))
            session_vars[fld.key] = _add_field(holder, fld, inherited, r)
        f2.pack(fill="x", padx=12, pady=4)

        status_lbl = tk.Label(win, text="", bg="#f5f5f5", fg="#d9534f", wraplength=520,
                              justify="left")
        status_lbl.pack(anchor="w", padx=12, pady=(2, 0))

        def _collect():
            fv = {k: v.get() for k, v in file_vars.items()}
            sv = {k: v.get() for k, v in session_vars.items()}
            return compute_overrides(fv, sv, state.defaults)

        def _validate():
            ov = _collect()
            merged = dict(state.defaults); merged.update(ov)
            merged["StimulationModality"] = modality
            vr = schema.validate(merged, modality=modality)
            status_lbl.config(
                text=("✓ Ready" if vr.ok else "Missing required: "
                      + ", ".join(e.split(" ", 1)[0] for e in vr.errors)),
                fg=("#5cb85c" if vr.ok else "#d9534f"))

        def _save():
            state.set_overrides(path, _collect(), reviewed=True)
            state.save()
            self._bidsify_tab_refresh()
            win.destroy()

        btns = tk.Frame(win, bg="#f5f5f5"); btns.pack(fill="x", padx=12, pady=10)
        tk.Button(btns, text="Cancel", command=win.destroy).pack(side="right")
        tk.Button(btns, text="Save", bg=_ACCENT, fg="white",
                  command=_save).pack(side="right", padx=(0, 8))
        tk.Button(btns, text="Validate", command=_validate).pack(side="right", padx=(0, 8))

        win.update_idletasks()
        win.minsize(560, 400)

    # ---- conversion -----------------------------------------------------------
    def _bidsify_convert_ready(self):
        state = self._get_bidsify_state()
        schema = self._bidsify_schema
        paths = self._bidsify_all_paths()
        ready = state.ready_paths(paths, schema)
        if not ready:
            messagebox.showinfo(
                "BIDS-ify",
                "No files are marked Ready. Review files (double-click) so they "
                "validate, then convert.", parent=self.root)
            return

        rawroot = state.rawdata_root or self._rawdata_path.get()
        if not rawroot:
            rawroot = filedialog.askdirectory(
                title="Choose the BIDS rawdata output folder")
            if not rawroot:
                return
            state.rawdata_root = rawroot
            state.save()

        from .app import _make_bids_prefix
        from pathlib import Path
        items = []
        for p in ready:
            parsed = self._parse_bids_from_filename(p)
            meta = StudyMetadata(
                participant_id=parsed['participant_id'] or "sub-unknown",
                session=parsed['session'] or "ses-01", task=parsed['task'],
                timepoint=parsed['timepoint'], limb=parsed['limb'],
                measure=parsed['measure'], acq=parsed['acq'])
            prefix = _make_bids_prefix(meta.bids_prefix(), Path(p).stem)
            items.append(BidsifyItem(
                source_path=p, metadata=meta, modality=state.modality,
                sidecar_values=state.effective_values(p),
                marker_names=[state.marker_name] if state.marker_name else None,
                prefix_override=prefix))

        ds_name = (os.path.basename(os.path.dirname(rawroot.rstrip("/\\")))
                   or "MEP-CMAP dataset")
        layout = DatasetLayout(rawdata_root=rawroot, dataset_name=ds_name)
        plan = plan_bidsify(items, layout, container=state.container,
                            powerline_hz=state.powerline_hz)
        if not self._show_bidsify_preview(plan):
            return

        self.log(f"BIDS-ify: converting {len(plan.files)} ready file(s) → {rawroot}")

        def _worker():
            try:
                results = execute_plan(plan, log=lambda m: self.msg_q.put(("log", m)))
            except Exception as exc:
                self.msg_q.put(("log", f"BIDS-ify failed: {exc}"))
                results = []
            self.msg_q.put(("bidsify-convert-done", results))
        threading.Thread(target=_worker, daemon=True).start()

    def _bidsify_convert_done(self, results):
        """Main-thread handler: mark converted files, refresh, summarise."""
        state = self._get_bidsify_state()
        when = datetime.datetime.now().isoformat(timespec="seconds")
        for r in results:
            if getattr(r, "ok", False):
                state.mark_converted(r.source_path, when=when)
        state.save()
        self._bidsify_tab_refresh()
        self._bidsify_done_gui(results)
