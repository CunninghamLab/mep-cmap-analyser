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
bids_schema for validation. Fully unit-testable.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

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

_SCHEMA_STATE_VERSION = 1


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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FileBidsRecord":
        return cls(
            rel_path=d.get("rel_path", ""),
            overrides=dict(d.get("overrides", {})),
            reviewed=bool(d.get("reviewed", False)),
            converted=bool(d.get("converted", False)),
            converted_at=d.get("converted_at", ""),
            output_prefix=d.get("output_prefix", ""),
            marker_names=list(d.get("marker_names", []) or []),
            stim_channel=d.get("stim_channel", ""),
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

    def status(self, path: str, schema) -> str:
        """Derive the worklist status for a file."""
        rec = self.record_for(path, create=False)
        if rec and rec.converted:
            return STATUS_CONVERTED
        if rec is None or not rec.reviewed:
            return STATUS_NOT_STARTED
        vr = schema.validate(self.effective_values(path), modality=self.modality)
        return STATUS_READY if vr.ok else STATUS_INCOMPLETE

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

    def ready_paths(self, paths: list, schema) -> list:
        """Files that are Ready (reviewed + valid, not yet converted)."""
        return [p for p in paths if self.status(p, schema) == STATUS_READY]
