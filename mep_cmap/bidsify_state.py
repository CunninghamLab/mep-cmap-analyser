"""
mep_cmap.bidsify_state
~~~~~~~~~~~~~~~~~~~~~~~
Persistent, resumable state for the BIDS-ify worklist.

Model: shared session defaults ⊕ per-file overrides. You enter the
session-constant values once (device, coil, targeting, power line, container,
marker); every file inherits them. Each file then stores only its overrides —
the handful of values that vary per recording — plus a reviewed/converted flag.
A file's *effective* metadata is defaults with its overrides layered on top.

Everything lives in ``bidsify_state.json`` in the dataset root, keyed by path
relative to that root, so you can close the app and resume exactly where you
left off (and the state travels with the dataset).

Status lifecycle (derived, not hand-set):
    not_started  — never reviewed (still only inheriting defaults)
    incomplete   — reviewed, but required fields still missing (won't convert)
    ready        — reviewed and validates (recommended-missing is allowed)
    converted    — EDF/BDF + sidecars written

This module is pure logic (no Tk, no pyedflib); it depends only on
bids_schema for validation and stim_params for the parameter set model.
Fully unit-testable.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from . import stim_params as _stim_params

STATE_FILENAME = "bidsify_state.json"

STATUS_NOT_STARTED = "not_started"
STATUS_INCOMPLETE  = "incomplete"
STATUS_READY       = "ready"
STATUS_CONVERTED   = "converted"

STATUS_LABELS = {
    STATUS_NOT_STARTED: "Not started",
    STATUS_INCOMPLETE:  "Incomplete",
    STATUS_READY:       "Ready",
    STATUS_CONVERTED:   "Converted",
}
STATUS_COLOURS = {
    STATUS_NOT_STARTED: "#888888",
    STATUS_INCOMPLETE:  "#d9534f",
    STATUS_READY:       "#f0a500",
    STATUS_CONVERTED:   "#5cb85c",
}

_SCHEMA_STATE_VERSION = 3


# ── Per-file record ───────────────────────────────────────────────────────────
@dataclass
class FileBidsRecord:
    rel_path:      str
    overrides:     dict = field(default_factory=dict)  # per-file field values
    reviewed:      bool = False        # user opened + saved this file at least once
    converted:     bool = False
    converted_at:  str = ""
    output_prefix: str = ""            # last written BIDS prefix (for reference)
    marker_names:  list = field(default_factory=list)  # per-file stim codes (scan & pick)
    stim_channel:  str = ""            # per-file stim event channel (scan & pick)
    #: stim code -> parameter set name. Which protocol each ticked code was.
    #:
    #: Per file rather than per session because the same code means different
    #: things in different recordings: 'A' is the M-wave in one file and the
    #: only stimulus in another. The SETS are session-level, because a
    #: threshold is not a property of a recording; the ASSIGNMENT is per file,
    #: because a stim code is.
    code_sets:     dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FileBidsRecord":
        # Conditions are deliberately NOT held here. They belong to the
        # recording, not to a conversion, and the session JSON already stores
        # them under "condition_rows" -- so a copy in this file would be a
        # second truth that nothing reconciles. It would also put the work of
        # analysts who never open BIDS-ify into a file named after a feature
        # they do not use. bidsify.conditions_for() reads the session instead.
        #
        # Version 3 records that removal. Keys written by the brief version
        # that held them are ignored: only the fields below are read.
        return cls(
            rel_path=d.get("rel_path", ""),
            overrides=dict(d.get("overrides", {})),
            reviewed=bool(d.get("reviewed", False)),
            converted=bool(d.get("converted", False)),
            converted_at=d.get("converted_at", ""),
            output_prefix=d.get("output_prefix", ""),
            marker_names=list(d.get("marker_names", []) or []),
            stim_channel=d.get("stim_channel", ""),
            # Absent in state written before v2. An empty assignment is the
            # honest answer for a file reviewed before parameter sets existed:
            # its codes are unassigned, and it reports as incomplete rather
            # than silently claiming a protocol nobody stated.
            code_sets=dict(d.get("code_sets", {}) or {}),
        )


# ── Whole-worklist state ──────────────────────────────────────────────────────
class BidsifyState:
    def __init__(self, root: str):
        self.root         = root
        self.modality     = "TMS"
        self.defaults     = {}          # shared session-constant sidecar values
        self.container    = "EDF"
        self.powerline_hz = 50
        self.marker_name  = "A"
        self.rawdata_root = ""          # chosen output rawdata root (optional)
        #: Session-level stimulation parameter sets (see mep_cmap.stim_params).
        #: Written once and referenced by every file, because a resting motor
        #: threshold is a property of the session, not of a recording.
        self.param_sets: list = []
        self._files: dict = {}          # rel_path -> FileBidsRecord

    # ---- path keying ----------------------------------------------------------
    def key_for(self, path: str) -> str:
        """Path relative to the dataset root when possible, else absolute (normalised)."""
        ap = os.path.normpath(os.path.abspath(path))
        try:
            rel = os.path.relpath(ap, self.root)
            # os.path.relpath can produce '..' escapes — only use it if truly inside
            if not rel.startswith(".."):
                return rel.replace(os.sep, "/")
        except ValueError:              # different drive on Windows
            pass
        return ap.replace(os.sep, "/")

    # ---- persistence ----------------------------------------------------------
    @classmethod
    def state_path(cls, root: str) -> str:
        return os.path.join(root, STATE_FILENAME)

    @classmethod
    def load_or_create(cls, root: str) -> "BidsifyState":
        st = cls(root)
        path = cls.state_path(root)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    d = json.load(fh)
                st.modality     = d.get("modality", "TMS")
                st.defaults     = dict(d.get("defaults", {}))
                st.container    = d.get("container", "EDF")
                st.powerline_hz = int(d.get("powerline_hz", 50))
                st.marker_name  = d.get("marker_name", "A")
                st.rawdata_root = d.get("rawdata_root", "")
                # Absent before state version 2. from_dicts tolerates anything
                # a hand edit could do and skips what it cannot read, because
                # this runs at startup and an unreadable set must not stop the
                # tool opening on a study the analyst still needs.
                st.param_sets   = _stim_params.from_dicts(d.get("param_sets"))
                for rec in d.get("files", []):
                    fr = FileBidsRecord.from_dict(rec)
                    st._files[fr.rel_path] = fr
            except (OSError, ValueError, json.JSONDecodeError):
                pass                    # corrupt/unreadable → start fresh
        return st

    def save(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        payload = {
            "schema_state_version": _SCHEMA_STATE_VERSION,
            "modality": self.modality,
            "defaults": self.defaults,
            "container": self.container,
            "powerline_hz": self.powerline_hz,
            "marker_name": self.marker_name,
            "rawdata_root": self.rawdata_root,
            "param_sets": _stim_params.to_dicts(self.param_sets),
            "files": [r.to_dict() for r in self._files.values()],
        }
        tmp = self.state_path(self.root) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.state_path(self.root))   # atomic-ish write

    # ---- shared defaults ------------------------------------------------------
    def set_defaults(self, modality: str, defaults: dict,
                     container: str, powerline_hz: int, marker_name: str) -> None:
        self.modality     = modality
        self.defaults     = dict(defaults or {})
        self.container    = container if container in ("EDF", "BDF") else "EDF"
        self.powerline_hz = int(powerline_hz) if str(powerline_hz).strip() else 50
        self.marker_name  = marker_name or "A"

    # ---- per-file records -----------------------------------------------------
    def record_for(self, path: str, create: bool = True) -> Optional[FileBidsRecord]:
        key = self.key_for(path)
        rec = self._files.get(key)
        if rec is None and create:
            rec = FileBidsRecord(rel_path=key)
            self._files[key] = rec
        return rec

    def set_overrides(self, path: str, overrides: dict, reviewed: bool = True) -> None:
        rec = self.record_for(path, create=True)
        rec.overrides = {k: v for k, v in (overrides or {}).items()
                         if v not in (None, "") and str(v).strip() != ""}
        if reviewed:
            rec.reviewed = True
        # editing invalidates a prior conversion mark only if the user re-reviews;
        # we keep 'converted' as-is so re-running convert is a conscious action.

    def mark_converted(self, path: str, output_prefix: str = "",
                       when: str = "") -> None:
        rec = self.record_for(path, create=True)
        rec.converted = True
        rec.output_prefix = output_prefix
        rec.converted_at = when

    def reset_file(self, path: str) -> None:
        key = self.key_for(path)
        self._files.pop(key, None)

    # ---- effective metadata + status -----------------------------------------
    def effective_values(self, path: str) -> dict:
        """Shared defaults with this file's overrides layered on top."""
        vals = dict(self.defaults)
        rec = self.record_for(path, create=False)
        if rec:
            vals.update(rec.overrides)
        vals["StimulationModality"] = self.modality
        return vals

    def unassigned_codes(self, path: str, splits=()) -> list:
        """Ticked stim codes with no parameter set, for this file.

        A code with no set cannot be described: *_nibs.tsv would have no row
        for its stimuli and *_events.tsv nothing to reference. Reported rather
        than guessed, because guessing here means publishing a protocol nobody
        stated.

        ``splits`` is the (code, condition) pairs the Conditions tab created.
        A split code needs a set for EVERY half -- half of 'A' at 100 mA and
        half at 150 mA is two protocols -- so it is checked per pair rather
        than per code. Passed in rather than read here, because that keeps this
        module free of file access and lets the caller reuse a lookup it has
        already done.
        """
        rec = self.record_for(path, create=False)
        if rec is None:
            return []
        by_code = {}
        for _c, _cond in (splits or []):
            by_code.setdefault(_c, []).append(_cond)
        assigned = rec.code_sets or {}
        out = []
        for code in (rec.marker_names or []):
            conds = by_code.get(code) or []
            if not conds:
                if not assigned.get(code):
                    out.append(code)
                continue
            for cond in conds:
                if not assigned.get(f"{code}\u00b7{cond}"):
                    out.append(f"{code} / {cond}")
        return out

    def status(self, path: str, schema, splits=()) -> str:
        """Derive the worklist status for a file."""
        rec = self.record_for(path, create=False)
        if rec and rec.converted:
            return STATUS_CONVERTED
        if rec is None or not rec.reviewed:
            return STATUS_NOT_STARTED
        vr = schema.validate(self.effective_values(path), modality=self.modality)
        if not vr.ok:
            return STATUS_INCOMPLETE
        # A file reviewed before parameter sets existed has ticked codes and no
        # assignment. It was Ready under the old rules and is not under the new
        # ones, which is the honest answer: its stimulation is undescribed.
        if self.unassigned_codes(path, splits):
            return STATUS_INCOMPLETE
        return STATUS_READY

    def missing_required(self, path: str, schema) -> list:
        """Required field keys still missing for this file (for the tree's hint column)."""
        vr = schema.validate(self.effective_values(path), modality=self.modality)
        # errors read like "Key is required but missing." — pull the key.
        keys = []
        for e in vr.errors:
            if " is required" in e:
                keys.append(e.split(" ", 1)[0])
        return keys

    # ---- worklist queries -----------------------------------------------------
    def counts(self, paths: list, schema) -> dict:
        out = {STATUS_NOT_STARTED: 0, STATUS_INCOMPLETE: 0,
               STATUS_READY: 0, STATUS_CONVERTED: 0}
        for p in paths:
            out[self.status(p, schema)] += 1
        return out

    def reopen_for_edit(self, path: str) -> bool:
        """Clear the converted flag so a file can be corrected and rewritten.

        Keeps everything else -- overrides, ticked codes, parameter set
        assignment -- because the point is to change one thing that was wrong,
        not to start again. reset_file() is still there for starting again.

        The written files are NOT deleted. Rewriting overwrites them in place,
        and deleting first would lose the old output if the rewrite then failed.
        Returns False when there was nothing converted to reopen.
        """
        rec = self.record_for(path, create=False)
        if rec is None or not rec.converted:
            return False
        rec.converted = False
        rec.converted_at = ""
        return True

    def ready_paths(self, paths: list, schema, splits_for=None) -> list:
        """Files that are Ready (reviewed + valid, not yet converted).

        ``splits_for`` is an optional callable taking a path and returning that
        recording's (code, condition) pairs. Without it a split code is judged
        on the bare code alone, so a file with one half of a split assigned and
        the other not would be offered for conversion and would write events
        referencing a parameter set that is not there.
        """
        return [p for p in paths
                if self.status(p, schema,
                               splits_for(p) if splits_for else ()) == STATUS_READY]
