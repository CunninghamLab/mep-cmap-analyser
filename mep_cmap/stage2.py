"""
mep_cmap.stage2
~~~~~~~~~~~~~~~
Stage 2 — Group Analysis mixin.

Contains all _s2_* methods and _build_stage2 / _on_tab_changed that implement
the group-level analysis tab.  Mixed into TMSAnalysisApp via Stage2Mixin.
"""

import os
import json
import re
import pathlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd

from .bids import StudyMetadata
from .preferences import accent_button_kw


# ─────────────────────────────────────────────────────────────────────────────
# Add-on sidecar join
# ─────────────────────────────────────────────────────────────────────────────
# Core first-level outputs that live beside <prefix>_trials.csv and must never be
# treated as add-on sidecars.
_S2_CORE_SUFFIXES = (
    "_trials.csv", "_trials_with_outliers.csv",
    "_summary.csv", "_summary_with_outliers.csv",
    "_bootstrap.csv",
)

# A sidecar must carry these to be joinable per trial. Segment is 1-based and
# indexes segs_all, so it is stable even if a table is filtered or re-sorted.
# Condition joins alongside StimType: with conditions assigned, A·pre and
# A·post both decompose to StimType=A and both have a Segment 1, so joining on
# the pair alone would match an add-on's row for one condition against the
# core row for the other -- silently, and only for files where conditions were
# used. A file without conditions has an empty Condition throughout, which
# joins exactly as before.
_S2_JOIN_KEYS = ("StimType", "Segment")

#: Joined on as well when BOTH frames carry it. With conditions assigned, A·pre
#: and A·post both report StimType=A and both have a Segment 1, so joining on
#: the pair alone would match an add-on's row for one condition against the core
#: row for the other. Optional rather than required, because an add-on written
#: before conditions existed -- including any third-party one -- emits no such
#: column and must still join; and where no conditions were assigned there is
#: nothing to disambiguate. The case where its absence WOULD be wrong is
#: detected below rather than assumed away.
_S2_OPTIONAL_JOIN_KEYS = ("Condition",)


def _s2_join_addon_sidecars(df, trials_csv_path, note):
    """Left-join any per-trial add-on outputs sitting beside <prefix>_trials.csv.

    An add-on sidecar is any other CSV in the same results folder sharing the
    session prefix and carrying the (StimType, Segment) join keys — the columns
    an add-on emits so its per-trial rows can be matched back to core trials.

    Purely additive: only columns absent from `df` are brought in, so an add-on
    can never overwrite a core measurement. Any sidecar that fails to load or
    join is skipped with a note; the group merge always proceeds.
    """
    import glob

    base = os.path.basename(trials_csv_path)
    if not base.endswith("_trials.csv"):
        return df
    prefix   = base[: -len("_trials.csv")]
    res_dir  = os.path.dirname(trials_csv_path)

    keys = [k for k in _S2_JOIN_KEYS + _S2_OPTIONAL_JOIN_KEYS
            if k in df.columns]
    if len(keys) < len(_S2_JOIN_KEYS):
        return df
    if "File" in df.columns:
        keys = ["File"] + keys

    for path in sorted(glob.glob(os.path.join(res_dir, f"{prefix}_*.csv"))):
        fn = os.path.basename(path)
        if any(fn.endswith(suf) for suf in _S2_CORE_SUFFIXES):
            continue
        try:
            side = pd.read_csv(path)
        except Exception as e:
            note(f"{fn}: unreadable ({e})")
            continue
        if side.empty or not set(_S2_JOIN_KEYS).issubset(side.columns):
            continue

        use = [k for k in keys if k in side.columns]
        if not set(_S2_JOIN_KEYS).issubset(use):
            continue

        # Refuse rather than mis-join. If the core table distinguishes trials
        # only by Condition -- the same StimType and Segment appearing more than
        # once -- then a sidecar without that column cannot say which row it
        # describes, and a left join would silently attach one condition's
        # measurements to another's trial. Re-running the add-on produces the
        # column; guessing produces a plausible wrong number.
        if "Condition" in df.columns and "Condition" not in use:
            _amb = df.duplicated(subset=list(_S2_JOIN_KEYS), keep=False).any()
            if _amb:
                note(f"{fn}: this file has conditions assigned and the sidecar "
                     f"carries no Condition column, so its rows cannot be "
                     f"matched unambiguously — skipped.  Re-run the add-on.")
                continue

        # Already joined? Every non-key column present means this sidecar has
        # been merged into `df` on a previous pass. Skip it, otherwise its own
        # columns would collide with themselves and get namespaced twice.
        plain = [c for c in side.columns if c not in use]
        if plain and all(c in df.columns for c in plain):
            continue

        # Columns that clash with an existing name are NAMESPACED rather than
        # dropped. MEPFeatX, for example, emits its own 'Latency(ms)' computed
        # by a different algorithm than the core pipeline's; silently discarding
        # it would lose data, and silently overwriting would corrupt the core
        # measurement. It arrives as 'mepfeatx_Latency(ms)' instead.
        tag = fn[len(prefix) + 1:-4] if len(fn) > len(prefix) + 5 else "addon"
        rename, new_cols = {}, []
        for c in side.columns:
            if c in use:
                continue
            target = c if c not in df.columns else f"{tag}_{c}"
            while target in df.columns or target in new_cols:
                target = f"{tag}_{target}"
            rename[c] = target
            new_cols.append(target)
        if not new_cols:
            continue

        # Merge on TEMPORARY normalised key columns. Coercing the real key
        # columns in place would rewrite core dtypes (e.g. 'Latency(ms)' holds
        # the string 'Not Detected' alongside floats), so the originals are left
        # exactly as read and the temp keys are dropped afterwards.
        def _norm(series, key):
            if key == "Segment":
                return pd.to_numeric(series, errors="coerce").astype("Int64")
            return series.astype(str).str.strip()

        tmp = [f"__s2join_{k}" for k in use]
        try:
            left  = df.copy()
            right = side.copy()
            for k, tk in zip(use, tmp):
                left[tk]  = _norm(left[k],  k)
                right[tk] = _norm(right[k], k)
            right = (right.rename(columns=rename)[new_cols + tmp]
                          .drop_duplicates(subset=tmp, keep="first"))

            # A sidecar whose keys don't actually line up would otherwise append
            # a block of all-NaN columns and look like it worked. Detect that and
            # skip loudly instead — a wrong File value is the usual cause.
            n_match = left[tmp].merge(right[tmp], on=tmp, how="inner").shape[0]
            if n_match == 0:
                note(f"{fn}: join keys matched no trials — skipped")
                continue

            merged = left.merge(right, on=tmp, how="left", validate="one_to_one")
            merged = merged.drop(columns=tmp)
        except Exception as e:
            note(f"{fn}: could not join ({e})")
            continue

        df = merged
        renamed = sum(1 for c, t in rename.items() if c != t)
        note(f"{fn}: +{len(new_cols)} column(s)"
             + (f" ({renamed} namespaced '{tag}_*' to avoid clashes)" if renamed else ""))

    return df


class Stage2Mixin:
    """
    Mixin providing the Stage 2 (Group Analysis) tab functionality.
    All methods are intended to be used as part of TMSAnalysisApp.
    """

    def _on_tab_changed(self, event=None):
        """Refresh whatever tab just became visible.

        Nested-notebook aware: fires on tab changes in any notebook (top or a
        sub-notebook) and uses winfo_ismapped() to decide what is now on screen,
        rather than relying on flat tab indices.
        """
        # Stage 2 — lazy-build on first visit
        tab2 = getattr(self, "tab2_frame", None)
        if tab2 is not None and tab2.winfo_ismapped() and not self._stage2_built:
            self._build_stage2()
            self._stage2_built = True

        # BIDS-ify — refresh the worklist when its sub-tab is shown
        bids = getattr(self, "tab_bidsify", None)
        if bids is not None and bids.winfo_ismapped():
            try:
                self._bidsify_tab_refresh()
            except Exception:
                pass

        # Run Analysis becomes available once the detection settings have been
        # seen for this recording. The footer is visible from every First Level
        # tab, so without this a run could be started from the labels tab on
        # whatever those settings were left at.
        try:
            _dt = getattr(self, "tab_detect", None)
            if _dt is not None and _dt.winfo_ismapped():
                self._seen_detection_tab = True
                self._refresh_run_button()
        except Exception:
            pass

        # Save on the way out of a tab, so preparation is never held only in
        # memory. A recording set up and then left for the next file used to
        # keep its labels, conditions and windows nowhere but the session that
        # had not been written yet.
        try:
            if self.file_path.get() and getattr(self, "_session_dirty", True):
                self._autosave_session()
                self._session_dirty = False
        except Exception:
            pass

        # Conditions — populate from the loaded recording when shown. The
        # table is built from the file's events, which do not exist until a
        # file has been opened, so it cannot be filled when the tab is created.
        cond = getattr(self, "tab_conditions", None)
        if cond is not None and cond.winfo_ismapped():
            try:
                self._cond_tab_shown()
            except Exception:
                pass

        # Add-ons — rescan when its sub-tab is shown (fresh drop-ins)
        addons_tab = getattr(self, "tab_addons", None)
        if addons_tab is not None and addons_tab.winfo_ismapped():
            try:
                self._addons_refresh_status()
                self._addons_discover()
            except Exception:
                pass

        # Group (second-level) Add-ons — rescan when its sub-tab is shown
        gadd = getattr(self, "tab_group_addons", None)
        if gadd is not None and gadd.winfo_ismapped():
            try:
                self._group_addons_refresh_status()
                self._group_addons_discover()
            except Exception:
                pass

    def _build_stage2(self):
        """Construct the entire Stage 2 panel inside self.tab2_frame."""
        f = self.tab2_frame

        # ── Top toolbar ───────────────────────────────────────────────────────
        toolbar = tk.Frame(f)
        toolbar.pack(fill="x", padx=10, pady=(10, 4))

        tk.Label(toolbar, text="Derivatives folder:").pack(side="left")
        self._s2_deriv_var = tk.StringVar(value=self.derivatives_path.get())
        tk.Entry(toolbar, textvariable=self._s2_deriv_var, width=45)            .pack(side="left", padx=(4, 2))
        tk.Button(toolbar, text="Browse",
                  command=self._s2_browse_deriv).pack(side="left")
        tk.Button(toolbar, text="Scan folder",
                  command=self._s2_scan).pack(side="left", padx=(12, 0))
        tk.Button(toolbar, text="Save design",
                  command=self._s2_save_design).pack(side="left", padx=(6, 0))
        tk.Button(toolbar, text="Load design",
                  command=self._s2_load_design).pack(side="left", padx=(2, 0))
        tk.Button(toolbar, text="▶  Build group analysis file",
                  command=self._s2_run,
                  **accent_button_kw("green")).pack(side="right", padx=(0, 4))

        # ── Group column manager ──────────────────────────────────────────────
        col_bar = tk.Frame(f)
        col_bar.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(col_bar, text="Group columns:").pack(side="left")
        tk.Button(col_bar, text="+ Add column",
                  command=self._s2_add_column).pack(side="left", padx=(6, 0))
        self._s2_col_buttons_frame = tk.Frame(col_bar)
        self._s2_col_buttons_frame.pack(side="left", padx=(8, 0))

        # ── Assignment table (Treeview + scrollbars) ──────────────────────────
        tbl_frame = tk.Frame(f)
        tbl_frame.pack(fill="both", expand=True, padx=10, pady=(0, 4))

        hscroll = ttk.Scrollbar(tbl_frame, orient="horizontal")
        hscroll.pack(side="bottom", fill="x")
        vscroll2 = ttk.Scrollbar(tbl_frame, orient="vertical")
        vscroll2.pack(side="right", fill="y")

        self._s2_tree = ttk.Treeview(
            tbl_frame,
            show="headings",
            selectmode="browse",
            yscrollcommand=vscroll2.set,
            xscrollcommand=hscroll.set,
        )
        self._s2_tree.pack(fill="both", expand=True)
        vscroll2.config(command=self._s2_tree.yview)
        hscroll.config(command=self._s2_tree.xview)

        # Bind double-click to edit a cell
        self._s2_tree.bind("<Double-1>", self._s2_on_double_click)
        # Bind right-click on heading for column context menu
        self._s2_tree.bind("<Button-3>", self._s2_on_right_click)

        # Shift+wheel scrolls sideways, which is what anyone reaches for before
        # dragging a scrollbar. Bound to the tree rather than the window so it
        # does not hijack the wheel elsewhere. Same idiom as the Stage 1 file
        # queue, deliberately: two tables that scroll differently is worse than
        # either behaviour on its own.
        def _s2_hwheel(event):
            _d = event.delta
            if _d:
                self._s2_tree.xview_scroll(int(-_d / 120) or
                                           (-1 if _d > 0 else 1), "units")
            return "break"

        self._s2_tree.bind("<Shift-MouseWheel>", _s2_hwheel)
        # X11 reports the wheel as buttons 6/7 horizontally. Windows Tk does
        # not know those numbers and REFUSES THE BIND -- "bad button number 6"
        # at construction, which stops the tab being built at all. So it is
        # attempted rather than assumed: a binding that cannot exist on this
        # platform is not an error, it is simply not that platform.
        for _seq, _dir in (("<Button-6>", -1), ("<Button-7>", 1)):
            try:
                self._s2_tree.bind(
                    _seq, lambda _e, _d=_dir:
                    self._s2_tree.xview_scroll(_d, "units"))
            except tk.TclError:
                pass

        # ── Status bar + quick-select ─────────────────────────────────────────
        bot = tk.Frame(f)
        bot.pack(fill="x", padx=10, pady=(0, 6))
        tk.Button(bot, text="Select all",
                  command=lambda: self._s2_set_all_include(True))            .pack(side="left", padx=(0, 4))
        tk.Button(bot, text="Deselect all",
                  command=lambda: self._s2_set_all_include(False))            .pack(side="left")
        self._s2_status = tk.Label(bot, text="", anchor="e")
        self._s2_status.pack(side="right")

        # ── Internal state ────────────────────────────────────────────────────
        # group_columns: list of {"name": str, "type": "between"|"within"}
        self._s2_group_cols  = []
        # group_values: {col_name: [val1, val2, ...]}
        self._s2_group_vals  = {}
        # row data: list of dicts (one per session row)
        self._s2_rows        = []

        # Follow Setup's derivatives folder, rather than reading it once.
        #
        # This panel is built when the notebook is, which is BEFORE Setup has a
        # folder and before a restored session writes one. So a single read at
        # construction found an empty string and nothing ever revisited it: the
        # analyst opened Second Level on a fully processed study, saw an empty
        # table, and had no clue the blank folder box was the reason. It only
        # ever appeared to work when the path happened to be known before the
        # notebook was built.
        #
        # Mirrored on write rather than re-read when the tab is selected,
        # because the scan is what fills _s2_rows, and Build group analysis file
        # works from _s2_rows whether or not anyone looked at the tab first.
        #
        # Registered AFTER the row state above exists: the scan it triggers is
        # deferred, so order does not bite today, but a synchronous scan would
        # reach _s2_rows before there was one.
        self._s2_deriv_followed = self._s2_deriv_var.get().strip()

        def _s2_follow_deriv(*_a):
            _new = self.derivatives_path.get().strip()
            _cur = self._s2_deriv_var.get().strip()
            # A path typed into this tab's own box wins. Only a box that is
            # empty, or still holds whatever was mirrored last time, follows
            # Setup -- otherwise choosing a folder in Setup would silently
            # discard a deliberately different one set here.
            if _cur and _cur != self._s2_deriv_followed and _cur != _new:
                return
            if _new != _cur:
                self._s2_deriv_var.set(_new)
            self._s2_deriv_followed = _new
            self._s2_autoscan()

        self.derivatives_path.trace_add("write", _s2_follow_deriv)
        # And once now, for the case the old one-shot read was written for: a
        # path already known by the time this panel is built.
        _s2_follow_deriv()

        self._s2_rebuild_tree_columns()
        self._s2_update_status()

    # ── Browse / scan ─────────────────────────────────────────────────────────

    def _s2_browse_deriv(self):
        folder = filedialog.askdirectory(
            title="Select derivatives folder", mustexist=True)
        if folder:
            self._s2_deriv_var.set(folder)
            self.derivatives_path.set(folder)

    def _s2_autoscan(self, delay_ms=150):
        """Scan because Setup changed, quietly, and once per burst.

        Not simply _s2_scan. That reports a missing folder with a dialogue,
        which is right when the analyst pressed Scan folder and wrong when
        nobody asked: startup writes derivatives_path more than once, and a
        folder that is briefly unset or absent is not worth interrupting anyone
        about. An automatic scan is therefore silent about a bad path and does
        nothing at all rather than complaining.

        Coalesced too, because those repeated writes would otherwise walk the
        whole derivatives tree once each.
        """
        _tok = getattr(self, "_s2_autoscan_token", None)
        if _tok is not None:
            try:
                self.root.after_cancel(_tok)
            except Exception:                       # noqa: BLE001 — stale token
                pass
            self._s2_autoscan_token = None
        self._s2_autoscan_token = self.root.after(delay_ms, self._s2_autoscan_fire)

    def _s2_autoscan_fire(self):
        """The deferred half. Re-checks the path, because it may have moved on."""
        self._s2_autoscan_token = None
        _path = self._s2_deriv_var.get().strip()
        if not _path or not os.path.isdir(_path):
            return
        if not hasattr(self, "_s2_tree"):
            return
        try:
            self._s2_scan()
        except Exception as exc:                    # noqa: BLE001 — logged
            # An automatic scan must not take the window down. The manual
            # button still surfaces problems the ordinary way.
            _log = getattr(self, "log", None)
            if callable(_log):
                _log(f"   ⚠️  Automatic derivatives scan failed: "
                     f"{type(exc).__name__}: {exc}")

    def _s2_scan(self):
        """Scan the derivatives folder for sidecar JSONs and populate the table."""
        root_dir = self._s2_deriv_var.get().strip()
        if not root_dir or not os.path.isdir(root_dir):
            messagebox.showerror("No folder",
                "Please enter or browse to a valid derivatives folder.",
                parent=self.root)
            return

        deriv_dir = os.path.join(root_dir, "derivatives")
        if not os.path.isdir(deriv_dir):
            # Try the folder itself as the derivatives root
            deriv_dir = root_dir

        # Walk and find all *_All_stims_trial_summary.json sidecars (one per session)
        found = []
        for dirpath, dirnames, filenames in os.walk(deriv_dir):
            for fn in filenames:
                if fn.endswith("_trials.json"):
                    jpath = os.path.join(dirpath, fn)
                    try:
                        with open(jpath, encoding="utf-8") as jf:
                            meta = json.load(jf)
                        # Fall back to parsing BIDS folder structure if metadata
                        # fields are blank (e.g. files processed before this fix)
                        _parts = pathlib.Path(dirpath).parts
                        _sub = next((p for p in _parts if p.startswith("sub-")), "")
                        _ses = next((p for p in _parts if p.startswith("ses-")), "")
                        # Several files can belong to one session (e.g. a
                        # 600-pulse protocol saved as six runs of 100 trials).
                        # Without a run discriminator they all collapse onto a
                        # single (participant_id, session) key and every row but
                        # one is silently dropped on rescan, so fall back to the
                        # filename's run- entity, then to the file stem.
                        _run = meta.get("run") or ""
                        if not _run:
                            _m = re.search(r"run-([A-Za-z0-9]+)", fn)
                            _run = _m.group(1) if _m else ""

                        # Limb and channel, so rows that differ can be told
                        # apart.
                        #
                        # A multi-channel run writes one set of derivatives per
                        # channel, so a two-channel recording produced two rows
                        # with identical participant, session and task -- no
                        # column said which was which, and the two muscles read
                        # as a duplicate.
                        #
                        # Both come from the filename's BIDS entities, with the
                        # sidecar as a fallback: the entity is what the run
                        # itself wrote, so it is right even when the sidecar is
                        # from an older version that did not record it.
                        _limb = meta.get("limb") or ""
                        if not _limb:
                            _m = re.search(r"limb-([A-Za-z0-9]+)", fn)
                            _limb = _m.group(1) if _m else ""
                        _chan = meta.get("channel") or ""
                        if not _chan:
                            _m = re.search(r"channel-([A-Za-z0-9]+)", fn)
                            _chan = _m.group(1) if _m else ""
                        found.append({
                            "include":        True,
                            "participant_id": meta.get("participant_id") or _sub,
                            "session":        meta.get("session")        or _ses,
                            "run":            _run,
                            "task":           meta.get("task",           ""),
                            "limb":           _limb,
                            "channel":        _chan,
                            "timepoint":      meta.get("timepoint",      ""),
                            "_json_path":     jpath,
                            "_trials_csv":    jpath.replace("_trials.json", "_trials.csv"),
                        })
                    except Exception:
                        pass

        if not found:
            messagebox.showinfo("Nothing found",
                "No First Level outputs found in that folder. "
                "Make sure you have run First Level with a derivatives folder set.",
                parent=self.root)
            return

        # Merge with existing rows: preserve group assignments for known sessions
        existing = {(r["participant_id"], r["session"], r.get("run", "")): r
                    for r in self._s2_rows}
        merged = []
        for row in found:
            key = (row["participant_id"], row["session"], row.get("run", ""))
            if key in existing:
                # Keep existing group assignments, update metadata
                old = existing[key].copy()
                old.update({k: v for k, v in row.items()
                            if k not in self._s2_group_cols})
                merged.append(old)
            else:
                # New row — add empty group columns
                for gc in self._s2_group_cols:
                    row[gc["name"]] = ""
                merged.append(row)

        # Sort by participant then session
        merged.sort(key=lambda r: (r["participant_id"], r["session"],
                                   r.get("run", ""), r.get("limb", ""),
                                   r.get("channel", "")))
        self._s2_rows = merged
        self._s2_refresh_tree()
        self._s2_update_status()

    # ── Tree column management ────────────────────────────────────────────────

    def _s2_rebuild_tree_columns(self):
        """Rebuild Treeview columns from current state."""
        fixed = ["include", "participant_id", "session", "run", "task",
                 "limb", "channel", "timepoint", "configure"]
        group_names = [gc["name"] for gc in self._s2_group_cols]
        all_cols = fixed + group_names

        self._s2_tree["columns"] = all_cols
        col_widths = {
            "include":        55,
            "participant_id": 110,
            "session":        80,
            "run":            55,
            "task":           90,
            "limb":           70,
            "channel":        85,
            "timepoint":      80,
            "configure":      80,
        }
        col_labels = {
            "include":        "Include",
            "participant_id": "Participant",
            "session":        "Session",
            "run":            "Run",
            "task":           "Task",
            "limb":           "Limb",
            "channel":        "Channel",
            "timepoint":      "Timepoint",
            "configure":      "Setup",
        }
        for col in all_cols:
            w = col_widths.get(col, 100)
            lbl = col_labels.get(col, col)
            # Mark between-subjects columns with a tilde prefix in header
            gc_meta = next((gc for gc in self._s2_group_cols
                            if gc["name"] == col), None)
            if gc_meta and gc_meta["type"] == "between":
                lbl = f"~ {col}"
            self._s2_tree.heading(col, text=lbl,
                command=lambda c=col: self._s2_sort_by(c))
            # stretch=False, and stated rather than left to the default.
            #
            # Tk's default is stretch=True, which makes every column absorb a
            # share of the leftover space: the columns then total exactly the
            # widget width, never more, so the horizontal scrollbar below has
            # nothing to scroll to and does nothing at all. The same default
            # made the Stage 1 file queue's scrollbar inert.
            #
            # It matters more here than it looks, because the column count is
            # not fixed -- every group column added by the analyst widens the
            # table, and with stretch on, they are absorbed by squeezing the
            # fixed columns until Participant and Timepoint are unreadable
            # instead of becoming scrollable.
            self._s2_tree.column(col, width=w, minwidth=50, stretch=False,
                                 anchor="center")

        self._s2_rebuild_col_buttons()

    def _s2_rebuild_col_buttons(self):
        """Refresh the row of column-management buttons."""
        for w in self._s2_col_buttons_frame.winfo_children():
            w.destroy()
        for gc in self._s2_group_cols:
            name = gc["name"]
            btn = tk.Button(
                self._s2_col_buttons_frame,
                text=f"⚙ {name}",
                relief="groove",
                padx=4,
                command=lambda n=name: self._s2_manage_column(n),
            )
            btn.pack(side="left", padx=2)

    def _s2_refresh_tree(self):
        """Clear and repopulate the Treeview from self._s2_rows."""
        for item in self._s2_tree.get_children():
            self._s2_tree.delete(item)
        # Filled from the tree's OWN column list, not a second one written by
        # hand.
        #
        # There were two lists and they disagreed: the columns included "run"
        # and this loop did not, so every value after it was written one column
        # to the left and Run was permanently blank. A column added to one list
        # and forgotten in the other shifts the whole row silently.
        _cols = [c for c in (self._s2_tree["columns"] or [])
                 if c not in ("include", "configure")]
        for i, row in enumerate(self._s2_rows):
            vals = ["☑" if row.get("include", True) else "☐"]
            for col in _cols:
                vals.append(row.get(col, ""))
            # Configure column: show tick if already configured
            cfg = row.get("_config", {})
            vals.insert(self._s2_tree["columns"].index("configure"),
                        "⚙ configured" if cfg.get("_done") else "⚙ setup")
            tag = "even" if i % 2 == 0 else "odd"
            self._s2_tree.insert("", "end", iid=str(i),
                                 values=vals, tags=(tag,))
        self._s2_tree.tag_configure("even", background="#f8f8f8")
        self._s2_tree.tag_configure("odd",  background="#ffffff")

    # ── Cell editing ──────────────────────────────────────────────────────────

    def _s2_on_double_click(self, event):
        """Handle double-click: toggle Include or open cell editor."""
        region = self._s2_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        col_id = self._s2_tree.identify_column(event.x)
        row_id = self._s2_tree.identify_row(event.y)
        if not row_id:
            return

        col_idx  = int(col_id.lstrip("#")) - 1
        all_cols = list(self._s2_tree["columns"])
        col_name = all_cols[col_idx]
        row_idx  = int(row_id)

        if col_name == "include":
            self._s2_rows[row_idx]["include"] = \
                not self._s2_rows[row_idx].get("include", True)
            self._s2_refresh_tree()
            self._s2_update_status()
            return

        if col_name == "configure":
            self._s2_open_configure(row_idx)
            return

        # Only group columns are editable
        gc_meta = next((gc for gc in self._s2_group_cols
                        if gc["name"] == col_name), None)
        if gc_meta is None:
            return

        self._s2_edit_cell(row_idx, col_name, gc_meta, event.x_root, event.y_root)

    def _s2_open_configure(self, row_idx):
        """
        Per-session Configure dialog — Section 1 only: stim type role assignment.
        Normalisation is already handled by Stage 1; role labels are the only
        additional metadata needed at the group level.
        """
        row = self._s2_rows[row_idx]
        csv_path = row.get("_trials_csv", "")

        if not csv_path or not os.path.isfile(csv_path):
            json_path = row.get("_json_path", "")
            if json_path:
                csv_path = json_path.replace(".json", ".csv")
            if not csv_path or not os.path.isfile(csv_path):
                messagebox.showerror("File not found",
                    "Could not locate the trials.csv for this session. "
                    "Please re-scan the derivatives folder.",
                    parent=self.root)
                return

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            messagebox.showerror("CSV error", str(e), parent=self.root)
            return

        stim_types = sorted(df["StimType"].unique()) if "StimType" in df.columns else []
        if not stim_types:
            messagebox.showinfo("No stim types",
                "No stim types found in this session's trial CSV.",
                parent=self.root)
            return

        cfg = row.setdefault("_config", {})

        title = " – ".join(filter(None, [
            row.get("participant_id",""), row.get("session",""),
            row.get("task",""), row.get("timepoint","")]))
        win = tk.Toplevel(self.root)
        win.title(f"Configure – {title}")
        win.transient(self.root)
        win.resizable(False, False)

        ROLES = ["None", "Reference (single pulse)", "Conditioned", "M-wave", "Other"]

        sec1 = tk.LabelFrame(win, text="Stim type roles", padx=8, pady=6)
        sec1.pack(fill="x", padx=10, pady=(10, 4))

        tk.Label(sec1, text="Stim",     width=8,  anchor="w").grid(row=0, column=0, sticky="w")
        tk.Label(sec1, text="Label",    width=16, anchor="w").grid(row=0, column=1, sticky="w")
        tk.Label(sec1, text="Role",     width=28, anchor="w").grid(row=0, column=2, sticky="w")
        tk.Label(sec1, text="N trials", width=8,  anchor="w").grid(row=0, column=3, sticky="w")

        role_vars = {}
        for r, st in enumerate(stim_types, start=1):
            n_trials = int((df["StimType"] == st).sum())
            lbl = (df.loc[df["StimType"] == st, "Stim_Label"].iloc[0]
                   if "Stim_Label" in df.columns else st)
            tk.Label(sec1, text=st,       width=8,  anchor="w").grid(row=r, column=0, sticky="w")
            tk.Label(sec1, text=str(lbl), width=16, anchor="w").grid(row=r, column=1, sticky="w")
            v = tk.StringVar(value=cfg.get(f"role_{st}", "None"))
            role_vars[st] = v
            ttk.Combobox(sec1, textvariable=v, values=ROLES,
                         state="readonly", width=26).grid(row=r, column=2, sticky="w", padx=4)
            tk.Label(sec1, text=str(n_trials), width=8, anchor="w").grid(row=r, column=3, sticky="w")

        tk.Label(win,
            text="First Level already handles normalisation. Roles are appended as\n"
                 "metadata to help identify stim type function in the merged file.",
            fg="grey", justify="left").pack(padx=10, pady=(4, 0), anchor="w")

        btn_row = tk.Frame(win)
        btn_row.pack(pady=10)

        def _save_config():
            new_cfg = {"_done": True}
            for st, v in role_vars.items():
                new_cfg[f"role_{st}"] = v.get()
            self._s2_rows[row_idx]["_config"] = new_cfg
            self._s2_refresh_tree()
            win.destroy()

        tk.Button(btn_row, text="Save & close", width=14,
                  command=_save_config).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", width=10,
                  command=win.destroy).pack(side="left", padx=6)
        win.grab_set()

    def _s2_edit_cell(self, row_idx, col_name, gc_meta, x_root, y_root):
        """Pop a small Combobox editor for a group cell."""
        current = self._s2_rows[row_idx].get(col_name, "")
        vals    = self._s2_group_vals.get(col_name, [])

        popup = tk.Toplevel(self.root)
        popup.overrideredirect(True)
        popup.geometry(f"+{x_root}+{y_root}")

        var = tk.StringVar(value=current)
        cb  = ttk.Combobox(popup, textvariable=var,
                           values=vals, width=18)
        cb.pack(padx=2, pady=2)
        cb.focus_set()
        cb.event_generate("<Button-1>")

        def _commit(_e=None):
            new_val = var.get().strip()
            if new_val and new_val not in self._s2_group_vals.get(col_name, []):
                self._s2_group_vals.setdefault(col_name, []).append(new_val)
            self._s2_rows[row_idx][col_name] = new_val
            # Auto-fill between-subjects: propagate to same participant
            if gc_meta["type"] == "between" and new_val:
                pid = self._s2_rows[row_idx]["participant_id"]
                for r in self._s2_rows:
                    if r["participant_id"] == pid and r.get(col_name, "") == "":
                        r[col_name] = new_val
            popup.destroy()
            self._s2_refresh_tree()

        cb.bind("<Return>",    _commit)
        cb.bind("<FocusOut>",  _commit)
        cb.bind("<<ComboboxSelected>>", _commit)

    # ── Include toggles ───────────────────────────────────────────────────────

    def _s2_set_all_include(self, state: bool):
        for row in self._s2_rows:
            row["include"] = state
        self._s2_refresh_tree()
        self._s2_update_status()

    def _s2_sort_by(self, col):
        """Sort table rows by the clicked column header."""
        self._s2_rows.sort(key=lambda r: str(r.get(col, "")))
        self._s2_refresh_tree()

    def _s2_update_status(self):
        n_total   = len(self._s2_rows)
        n_include = sum(1 for r in self._s2_rows if r.get("include", True))
        self._s2_status.config(
            text=f"{n_include} / {n_total} sessions included")

    # ── Right-click column context menu ───────────────────────────────────────

    def _s2_on_right_click(self, event):
        """Show context menu when right-clicking a column heading."""
        region = self._s2_tree.identify_region(event.x, event.y)
        if region != "heading":
            return
        col_id   = self._s2_tree.identify_column(event.x)
        col_idx  = int(col_id.lstrip("#")) - 1
        all_cols = list(self._s2_tree["columns"])
        col_name = all_cols[col_idx]
        gc_meta  = next((gc for gc in self._s2_group_cols
                         if gc["name"] == col_name), None)
        if gc_meta is None:
            return   # fixed column — no context menu

        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Rename column…",
                         command=lambda: self._s2_rename_column(col_name))
        menu.add_command(label="Edit allowed values…",
                         command=lambda: self._s2_manage_column(col_name))
        menu.add_separator()
        if gc_meta["type"] == "between":
            menu.add_command(label="Change to Within-subjects",
                             command=lambda: self._s2_set_col_type(col_name, "within"))
        else:
            menu.add_command(label="Change to Between-subjects",
                             command=lambda: self._s2_set_col_type(col_name, "between"))
        menu.add_separator()
        menu.add_command(label="Delete column",
                         command=lambda: self._s2_delete_column(col_name))
        menu.tk_popup(event.x_root, event.y_root)

    # ── Add / rename / delete columns ─────────────────────────────────────────

    def _s2_add_column(self):
        """Dialog to add a new group column."""
        win = tk.Toplevel(self.root)
        win.title("Add group column")
        win.resizable(False, False)
        win.transient(self.root)

        tk.Label(win, text="Column name:").grid(
            row=0, column=0, sticky="e", padx=8, pady=6)
        v_name = tk.StringVar()
        tk.Entry(win, textvariable=v_name, width=20).grid(
            row=0, column=1, sticky="w", padx=4)

        tk.Label(win, text="Column type:").grid(
            row=1, column=0, sticky="e", padx=8, pady=4)
        v_type = tk.StringVar(value="between")
        type_frame = tk.Frame(win)
        type_frame.grid(row=1, column=1, sticky="w")
        tk.Radiobutton(type_frame, text="Between-subjects  (auto-fills per participant)",
                       variable=v_type, value="between").pack(anchor="w")
        tk.Radiobutton(type_frame, text="Within-subjects / crossover  (fill each session independently)",
                       variable=v_type, value="within").pack(anchor="w")

        err = tk.Label(win, text="", fg="red")
        err.grid(row=2, column=0, columnspan=2, padx=8)

        def _ok(_e=None):
            name = v_name.get().strip()
            if not name:
                err.config(text="Name required.")
                return
            if any(gc["name"] == name for gc in self._s2_group_cols):
                err.config(text="Column already exists.")
                return
            self._s2_group_cols.append({"name": name, "type": v_type.get()})
            self._s2_group_vals[name] = []
            for row in self._s2_rows:
                row[name] = ""
            self._s2_rebuild_tree_columns()
            self._s2_refresh_tree()
            win.destroy()

        btn_row = tk.Frame(win)
        btn_row.grid(row=3, column=0, columnspan=2, pady=8)
        tk.Button(btn_row, text="Add", width=9, command=_ok).pack(side="left", padx=4)
        tk.Button(btn_row, text="Cancel", width=9,
                  command=win.destroy).pack(side="left", padx=4)
        win.bind("<Return>", _ok)
        win.grab_set()

    def _s2_manage_column(self, col_name):
        """Dialog to view/edit/delete the allowed values for a group column."""
        win = tk.Toplevel(self.root)
        win.title(f"Manage values – {col_name}")
        win.resizable(False, False)
        win.transient(self.root)

        tk.Label(win, text=f"Allowed values for  '{col_name}'  "
                           f"(double-click to rename, Delete key to remove):").pack(
                           padx=10, pady=(8, 2), anchor="w")

        lb_frame = tk.Frame(win)
        lb_frame.pack(fill="both", expand=True, padx=10)
        lb = tk.Listbox(lb_frame, height=8, width=28, selectmode="single")
        lb.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lb_frame, command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)

        def _repopulate():
            lb.delete(0, "end")
            for v in self._s2_group_vals.get(col_name, []):
                lb.insert("end", v)

        _repopulate()

        def _delete_val(_e=None):
            sel = lb.curselection()
            if not sel:
                return
            val = lb.get(sel[0])
            if messagebox.askyesno("Delete value",
                    f"Remove '{val}' from allowed values? "
                    "Cells currently set to this value will be cleared.",
                    parent=win):
                self._s2_group_vals[col_name].remove(val)
                for row in self._s2_rows:
                    if row.get(col_name) == val:
                        row[col_name] = ""
                _repopulate()
                self._s2_refresh_tree()

        def _rename_val(_e=None):
            sel = lb.curselection()
            if not sel:
                return
            old_val = lb.get(sel[0])
            new_val = simpledialog.askstring(
                "Rename value", f"New name for '{old_val}':",
                initialvalue=old_val, parent=win)
            if new_val and new_val.strip() and new_val.strip() != old_val:
                new_val = new_val.strip()
                idx = self._s2_group_vals[col_name].index(old_val)
                self._s2_group_vals[col_name][idx] = new_val
                for row in self._s2_rows:
                    if row.get(col_name) == old_val:
                        row[col_name] = new_val
                _repopulate()
                self._s2_refresh_tree()

        lb.bind("<Double-1>", _rename_val)
        lb.bind("<Delete>",   _delete_val)

        btn_row = tk.Frame(win)
        btn_row.pack(pady=6)
        tk.Button(btn_row, text="Rename selected",
                  command=_rename_val).pack(side="left", padx=4)
        tk.Button(btn_row, text="Delete selected",
                  command=_delete_val).pack(side="left", padx=4)
        tk.Button(btn_row, text="Close",
                  command=win.destroy).pack(side="left", padx=4)
        win.grab_set()

    def _s2_rename_column(self, old_name):
        new_name = simpledialog.askstring(
            "Rename column", f"New name for '{old_name}':",
            initialvalue=old_name, parent=self.root)
        if not new_name or not new_name.strip() or new_name.strip() == old_name:
            return
        new_name = new_name.strip()
        if any(gc["name"] == new_name for gc in self._s2_group_cols):
            messagebox.showerror("Duplicate", f"A column named '{new_name}' already exists.")
            return
        for gc in self._s2_group_cols:
            if gc["name"] == old_name:
                gc["name"] = new_name
        if old_name in self._s2_group_vals:
            self._s2_group_vals[new_name] = self._s2_group_vals.pop(old_name)
        for row in self._s2_rows:
            if old_name in row:
                row[new_name] = row.pop(old_name)
        self._s2_rebuild_tree_columns()
        self._s2_refresh_tree()

    def _s2_delete_column(self, col_name):
        if not messagebox.askyesno("Delete column",
                f"Delete column '{col_name}' and all its assignments?",
                parent=self.root):
            return
        self._s2_group_cols = [gc for gc in self._s2_group_cols
                               if gc["name"] != col_name]
        self._s2_group_vals.pop(col_name, None)
        for row in self._s2_rows:
            row.pop(col_name, None)
        self._s2_rebuild_tree_columns()
        self._s2_refresh_tree()

    def _s2_set_col_type(self, col_name, new_type):
        for gc in self._s2_group_cols:
            if gc["name"] == col_name:
                gc["type"] = new_type
        self._s2_rebuild_tree_columns()
        self._s2_refresh_tree()

    def _s2_run(self):
        """
        Merge all included sessions' trial-level CSVs into a single
        group-level LME-ready file, appending study design columns.

        Output: derivatives/group_level_LME_ready.csv
        """
        # ── Validate ──────────────────────────────────────────────────────────
        included = [r for r in self._s2_rows if r.get("include", True)]
        if not included:
            messagebox.showwarning("Nothing included",
                "No sessions are included. Use the checkboxes to include sessions.",
                parent=self.root)
            return

        root_dir  = self._s2_deriv_var.get().strip()
        deriv_dir = os.path.join(root_dir, "derivatives")
        if not os.path.isdir(deriv_dir):
            deriv_dir = root_dir
        if not os.path.isdir(deriv_dir):
            messagebox.showerror("No folder",
                "Could not locate the derivatives folder.", parent=self.root)
            return

        # ── Identify design columns ───────────────────────────────────────────
        group_cols  = [gc["name"] for gc in self._s2_group_cols]

        # ── Load and annotate each session ────────────────────────────────────
        all_frames = []
        skipped    = []
        sidecar_notes = []

        for row in included:
            csv_path = row.get("_trials_csv", "")
            if not csv_path or not os.path.isfile(csv_path):
                skipped.append(row.get("participant_id", "?") + "/" +
                               row.get("session", "?"))
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                skipped.append(f"{row.get('participant_id','?')}: {e}")
                continue

            if df.empty:
                skipped.append(row.get("participant_id", "?") + " (empty CSV)")
                continue

            # ── Join per-trial add-on outputs (temporal_decomposition, ...) ────
            # Additive only: add-on columns are appended, core columns untouched.
            _who = f"{row.get('participant_id','?')}/{row.get('session','?')}"
            df = _s2_join_addon_sidecars(
                df, csv_path, lambda m, w=_who: sidecar_notes.append(f"{w} {m}"))

            # ── Append Stim_Role from Configure dialog ─────────────────────────
            cfg = row.get("_config", {})
            df["Stim_Role"] = df["StimType"].map(
                lambda st: cfg.get(f"role_{st}", "None") if cfg else "None")

            # ── Prepend design columns in correct order ────────────────────────
            # Target order: File, participant_id, [group cols], session,
            #               task, timepoint, StimType, Stim_Label, Segment ...
            # Insert right-to-left so index 0 ends up as File
            for gc_name in reversed(group_cols):
                df.insert(1, gc_name, row.get(gc_name, ""))
            df.insert(1, "participant_id", row.get("participant_id", ""))

            # Move session/task/timepoint to just after participant/group cols
            # They already exist in the CSV from Stage 1 if BIDS was set,
            # otherwise add them from the study design
            n_design = 2 + len(group_cols)  # File + participant_id + group cols
            for i, col in enumerate(["session", "task", "timepoint"]):
                if col in df.columns:
                    # Move existing column to correct position
                    s = df.pop(col)
                    df.insert(n_design + i, col, row.get(col, "") or s)
                else:
                    df.insert(n_design + i, col, row.get(col, ""))

            # ── Reorder columns ────────────────────────────────────────────────
            # Final order: File, participant_id, [group cols], session, task,
            # timepoint, Limb, StimType, Stim_Label, Segment, [metrics...]
            if "Limb" in df.columns:
                limb = df.pop("Limb")
                # Insert after timepoint (n_design + 3 cols: session/task/timepoint)
                df.insert(n_design + 3, "Limb", limb)

            all_frames.append(df)

        # ── Bail if nothing loaded ─────────────────────────────────────────────
        if not all_frames:
            messagebox.showerror("No data",
                "Could not load any session CSVs. Check that First Level has been "
                "run and the derivatives folder is correct.", parent=self.root)
            return

        # ── Stack all sessions ─────────────────────────────────────────────────
        # Use outer join so sessions with different columns don't crash —
        # missing columns are filled with NaN.
        merged = pd.concat(all_frames, axis=0, ignore_index=True, sort=False)

        # Sort: participant → session → stim type → segment
        sort_cols = [c for c in ["participant_id", "session", "StimType", "Segment"]
                     if c in merged.columns]
        if sort_cols:
            merged = merged.sort_values(sort_cols).reset_index(drop=True)

        # ── Write output ──────────────────────────────────────────────────────
        out_path = os.path.join(deriv_dir, "group_level_LME_ready.csv")
        try:
            merged.to_csv(out_path, index=False)
        except Exception as e:
            messagebox.showerror("Write error", str(e), parent=self.root)
            return

        # ── Report ────────────────────────────────────────────────────────────
        n_sessions = len(all_frames)
        n_trials   = len(merged)
        n_cols     = len(merged.columns)
        msg = (f"Group analysis complete.\n\n"
               f"Sessions merged:  {n_sessions}\n"
               f"Total trials:     {n_trials}\n"
               f"Columns:          {n_cols}\n\n"
               f"Saved to:\n{out_path}")
        if sidecar_notes:
            _shown = sidecar_notes[:8]
            msg += (f"\n\nAdd-on columns joined ({len(sidecar_notes)}):\n"
                    + "\n".join(_shown)
                    + ("\n…" if len(sidecar_notes) > len(_shown) else ""))
        if skipped:
            msg += f"\n\nSkipped ({len(skipped)}):\n" + "\n".join(skipped)
        messagebox.showinfo("Done", msg, parent=self.root)
        self._s2_status.config(
            text=f"✔  Exported {n_trials} trials from {n_sessions} sessions → "
                 f"{os.path.basename(out_path)}")

    # ── Save / load study design ──────────────────────────────────────────────

    def _s2_save_design(self):
        root_dir = self._s2_deriv_var.get().strip()
        if not root_dir:
            messagebox.showerror("No folder",
                "Please set the derivatives folder first.", parent=self.root)
            return
        deriv_dir = os.path.join(root_dir, "derivatives")
        os.makedirs(deriv_dir, exist_ok=True)
        design = {
            "group_columns": self._s2_group_cols,
            "group_values":  self._s2_group_vals,
            "assignments": [
                {k: v for k, v in row.items() if not k.startswith("_")}
                for row in self._s2_rows
            ],
        }
        path = os.path.join(deriv_dir, "study_design.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(design, f, indent=2)
        self._s2_status.config(text=f"Design saved → {path}")

    def _s2_load_design(self):
        root_dir = self._s2_deriv_var.get().strip()
        deriv_dir = os.path.join(root_dir, "derivatives")             if root_dir else ""
        init = deriv_dir if os.path.isdir(deriv_dir) else root_dir
        path = filedialog.askopenfilename(
            title="Load study design",
            initialdir=init,
            filetypes=[("Study design", "study_design.json"),
                       ("JSON files", "*.json")],
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                design = json.load(f)
            self._s2_group_cols = design.get("group_columns", [])
            self._s2_group_vals = design.get("group_values", {})
            rows = design.get("assignments", [])
            # Re-attach private keys (csv paths) by re-scanning if needed
            self._s2_rows = rows
            self._s2_rebuild_tree_columns()
            self._s2_refresh_tree()
            self._s2_update_status()
        except Exception as e:
            messagebox.showerror("Load error", str(e), parent=self.root)

    # ─── End Stage 2 ──────────────────────────────────────────────────────────

    # ─── Input File Selection ────────────────────────────────────────────
