"""
mep_cmap.bids_schema
~~~~~~~~~~~~~~~~~~~~~
Schema-driven NIBS (BEP037) metadata support.

The schema lives in an editable, versioned JSON asset (``schema/nibs_bep037.json``)
rather than in code, so the BEP037 field set can be reconciled against the spec by
editing the asset and bumping ``schema_version`` — no code change required.

Both consumers read from this single source of truth:

  • the metadata dialog renders fields via :meth:`NibsSchema.fields_for`
  • the BIDS writer validates with :meth:`NibsSchema.validate` and orders the
    sidecar with :meth:`NibsSchema.ordered_sidecar`

Nothing here imports from ``pipeline.py`` or ``app.py``; the dependency only ever
points the other way, which keeps the ingestion stage decoupled from the
derivatives logic.

PyInstaller note: the JSON asset must be added to the build. Add a ``datas`` entry
mapping ``mep_cmap/schema/nibs_bep037.json`` to ``mep_cmap/schema`` in the .spec
file, and include it as package data in pyproject (``[tool.setuptools.package-data]``
or the wheel MANIFEST), or exe/pip users get a tool that cannot find its own schema.
"""

import os
import sys
import json
from dataclasses import dataclass, field as _dc_field
from typing import Any, Optional


# ── Asset location ──────────────────────────────────────────────────────────
SCHEMA_FILENAME = "nibs_bep037.json"
_ENV_OVERRIDE   = "MEP_CMAP_NIBS_SCHEMA"   # absolute path override for testing


def _default_schema_path() -> str:
    """
    Resolve the bundled schema JSON across source, wheel and frozen (PyInstaller)
    layouts. Order: env override → PyInstaller _MEIPASS → module-relative.
    """
    env = os.environ.get(_ENV_OVERRIDE)
    if env and os.path.isfile(env):
        return env

    # PyInstaller unpacks datas under sys._MEIPASS.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cand = os.path.join(meipass, "mep_cmap", "schema", SCHEMA_FILENAME)
        if os.path.isfile(cand):
            return cand
        cand = os.path.join(meipass, "schema", SCHEMA_FILENAME)
        if os.path.isfile(cand):
            return cand

    # Source / installed-wheel layout: <this dir>/schema/<file>.
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "schema", SCHEMA_FILENAME)


# ── Field model ───────────────────────────────────────────────────────────────
_VALID_LEVELS = ("required", "recommended", "optional")
_VALID_TYPES  = ("string", "number", "integer", "boolean", "array")
_COMMON       = "common"   # sentinel modality meaning "applies to all modalities"


@dataclass(frozen=True)
class SchemaField:
    """One BEP037 field definition."""
    key:         str
    modalities:  tuple
    group:       str
    level:       str
    type:        str
    units:       Optional[str] = None
    enum:        Optional[tuple] = None
    advanced:    bool = False
    scope:       str = "session"   # 'session' | 'file' | 'parameter_set'
    #: Where this field serialises in the NIBS-BIDS v6.3 output. One of the
    #: JSON blocks ('StimulatorSet', 'ElementSet', 'IntensitySet'), one of the
    #: tabular files ('nibs.tsv', 'markers.tsv', 'events.tsv'), or '' for a
    #: top-level key of *_nibs.json.
    #:
    #: Held as data rather than as a mapping in the writer because the writer
    #: would then be a second list of every field, maintained by hand, and the
    #: two would drift the first time a field was added to one of them. The
    #: schema already says what each field IS; this says where it GOES.
    block:       str = ""
    #: The NIBS-BIDS v6.3 name this value is written under, or "" for the key
    #: itself. Kept separate from `key` deliberately: BEP037 has renamed fields
    #: twice already, and renaming `key` to match would invalidate every saved
    #: bidsify_state.json. A spec rename is an `emits` change and nothing else.
    emits:       str = ""
    #: Read from older saved state, never written. Superseded by a v6.3 field.
    legacy:      bool = False
    description: str = ""

    def applies_to(self, modality: Optional[str]) -> bool:
        """True if this field is relevant for the given modality (None = any)."""
        if _COMMON in self.modalities:
            return True
        if modality is None:
            return True
        return modality in self.modalities


@dataclass
class ValidationResult:
    """Outcome of :meth:`NibsSchema.validate`."""
    errors:       list = _dc_field(default_factory=list)   # block a valid write
    warnings:     list = _dc_field(default_factory=list)   # soft, non-blocking
    unknown_keys: list = _dc_field(default_factory=list)   # custom user fields (kept)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = []
        if self.errors:
            lines.append("Errors (must fix):")
            lines += [f"  • {e}" for e in self.errors]
        if self.warnings:
            lines.append("Recommended but missing:")
            lines += [f"  • {w}" for w in self.warnings]
        if self.unknown_keys:
            lines.append("Custom fields (will be written as-is): "
                         + ", ".join(self.unknown_keys))
        return "\n".join(lines) if lines else "All required fields present."


# ── Schema ──────────────────────────────────────────────────────────────────
class NibsSchema:
    """In-memory view of the BEP037 field schema, with filtering/validation."""

    def __init__(self, data: dict):
        self.schema_name    = data.get("schema_name", "nibs_bep037")
        self.schema_version = data.get("schema_version", "0.0.0")
        self.based_on       = data.get("based_on", "")
        self.datatype       = data.get("datatype", "nibs")
        self.modalities     = list(data.get("modalities", []))
        self.groups         = list(data.get("groups", []))
        self.levels         = list(data.get("levels", _VALID_LEVELS))

        self._fields = []
        self._by_key = {}
        for raw in data.get("fields", []):
            fld = self._parse_field(raw)
            if fld.key in self._by_key:
                raise ValueError(f"Duplicate field key in schema: {fld.key!r}")
            self._fields.append(fld)
            self._by_key[fld.key] = fld

    # ---- construction helpers -------------------------------------------------
    @staticmethod
    def _parse_field(raw: dict) -> SchemaField:
        key = raw.get("key")
        if not key:
            raise ValueError(f"Schema field is missing a 'key': {raw!r}")

        level = raw.get("level", "optional")
        if level not in _VALID_LEVELS:
            raise ValueError(f"Field {key!r}: invalid level {level!r}")

        ftype = raw.get("type", "string")
        if ftype not in _VALID_TYPES:
            raise ValueError(f"Field {key!r}: invalid type {ftype!r}")

        modalities = tuple(raw.get("modalities") or [_COMMON])
        enum       = raw.get("enum")
        enum       = tuple(enum) if enum else None

        return SchemaField(
            key=key,
            modalities=modalities,
            group=raw.get("group", "parameters"),
            level=level,
            type=ftype,
            units=raw.get("units"),
            enum=enum,
            advanced=bool(raw.get("advanced", False)),
            scope=raw.get("scope", "session"),
            block=raw.get("block", ""),
            emits=raw.get("emits", ""),
            legacy=bool(raw.get("legacy", False)),
            description=raw.get("description", ""),
        )

    # ---- queries --------------------------------------------------------------
    @property
    def fields(self) -> list:
        """All fields, in schema order."""
        return list(self._fields)

    def field(self, key: str) -> Optional[SchemaField]:
        return self._by_key.get(key)

    def fields_for(self,
                   modality: Optional[str] = None,
                   group:    Optional[str] = None,
                   level:    Optional[str] = None,
                   scope:    Optional[str] = None,
                   block:    Optional[str] = None,
                   include_advanced: bool = True,
                   include_legacy: bool = False) -> list:
        """
        Return fields filtered by modality / group / level / scope / block, in
        schema order. ``modality=None`` returns every modality's fields;
        otherwise common fields plus those tagged for ``modality``. ``scope``
        filters to 'session', 'file' or 'parameter_set' fields when given, and
        ``block`` to the fields serialising into one JSON block or TSV file.

        Legacy fields are excluded by default, which is what makes `legacy`
        mean anything: they are read from state saved by an older version but
        must not be offered in the dialogue, demanded by validation, or written
        to a sidecar. They stay in ``_by_key``, so a saved value is still
        recognised rather than reappearing as an unknown custom field.
        """
        out = []
        for f in self._fields:
            if not f.applies_to(modality):
                continue
            if not include_legacy and f.legacy:
                continue
            if group is not None and f.group != group:
                continue
            if level is not None and f.level != level:
                continue
            if scope is not None and f.scope != scope:
                continue
            if block is not None and f.block != block:
                continue
            if not include_advanced and f.advanced:
                continue
            out.append(f)
        return out

    def required_keys(self, modality: Optional[str] = None) -> list:
        return [f.key for f in self.fields_for(modality, level="required")]

    # ---- value handling -------------------------------------------------------
    @staticmethod
    def _is_blank(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        if isinstance(v, (list, tuple)) and len(v) == 0:
            return True
        return False

    def coerce_value(self, fld: SchemaField, value: Any):
        """
        Convert a (usually string, from the dialog) value to the field's type.
        Returns ``(coerced_value, error_or_None)``. On failure the original value
        is returned unchanged alongside a message, so nothing is silently dropped.
        """
        if self._is_blank(value):
            return None, None

        try:
            if fld.type == "string":
                return str(value), None

            if fld.type == "boolean":
                if isinstance(value, bool):
                    return value, None
                s = str(value).strip().lower()
                if s in ("true", "1", "yes", "y"):
                    return True, None
                if s in ("false", "0", "no", "n"):
                    return False, None
                raise ValueError("expected true/false")

            if fld.type == "integer":
                if isinstance(value, bool):
                    raise ValueError("expected an integer, got boolean")
                return int(str(value).strip()), None

            if fld.type == "number":
                return float(str(value).strip()), None

            if fld.type == "array":
                if isinstance(value, (list, tuple)):
                    items = list(value)
                else:
                    # accept comma- or whitespace-separated input from the dialog
                    raw = str(value).replace(",", " ").split()
                    items = []
                    for tok in raw:
                        try:
                            items.append(float(tok))
                        except ValueError:
                            items.append(tok)
                return items, None
        except (TypeError, ValueError) as exc:
            return value, f"{fld.key}: could not convert {value!r} to {fld.type} ({exc})"

        return value, None

    def validate(self,
                 values: dict,
                 modality: Optional[str] = None) -> ValidationResult:
        """
        Check ``values`` against the schema for ``modality``.

          • required + blank      → error
          • recommended + blank   → warning
          • bad enum / bad type   → error
          • key not in schema     → recorded as a custom field (kept, not an error)

        If ``modality`` is None it is taken from ``values['StimulationModality']``
        when present.
        """
        res = ValidationResult()
        if modality is None:
            modality = (values.get("StimulationModality") or None)

        applicable = {f.key: f for f in self.fields_for(modality)}

        # presence / enum / type for applicable fields
        for key, fld in applicable.items():
            raw = values.get(key)
            if self._is_blank(raw):
                if fld.level == "required":
                    res.errors.append(f"{key} is required but missing.")
                elif fld.level == "recommended":
                    res.warnings.append(f"{key} ({fld.description or 'recommended'})")
                continue

            coerced, cerr = self.coerce_value(fld, raw)
            if cerr:
                res.errors.append(cerr)
                continue
            if fld.enum and coerced not in fld.enum:
                res.errors.append(
                    f"{key}: {coerced!r} is not one of {list(fld.enum)}.")

        # keys the user supplied that the schema does not know about
        for key in values:
            if self._is_blank(values[key]):
                continue
            if key in self._by_key:
                continue
            res.unknown_keys.append(key)

        return res

    def ordered_sidecar(self,
                        values: dict,
                        modality: Optional[str] = None,
                        keep_custom: bool = True,
                        add_provenance: bool = True) -> dict:
        """
        Build the ``nibs`` sidecar dict: applicable schema fields in schema order
        (blanks omitted, per BIDS convention, values coerced to type), then any
        custom user fields, then provenance stamping the schema version targeted.
        """
        if modality is None:
            modality = (values.get("StimulationModality") or None)

        out = {}
        applicable = self.fields_for(modality)
        applicable_keys = {f.key for f in applicable}

        for fld in applicable:
            raw = values.get(fld.key)
            if self._is_blank(raw):
                continue
            coerced, cerr = self.coerce_value(fld, raw)
            out[fld.key] = raw if cerr else coerced

        if keep_custom:
            for key, raw in values.items():
                if key in applicable_keys or key in self._by_key:
                    continue
                if not self._is_blank(raw):
                    out[key] = raw

        if add_provenance:
            out["NIBSSchema"]        = self.schema_name
            out["NIBSSchemaVersion"] = self.schema_version

        return out


# ── Module-level convenience ──────────────────────────────────────────────────
_CACHE = {}


def load_schema(path: Optional[str] = None, use_cache: bool = True) -> NibsSchema:
    """Load and parse the schema JSON (cached per path)."""
    path = path or _default_schema_path()
    if use_cache and path in _CACHE:
        return _CACHE[path]
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"NIBS schema not found at {path!r}. If running a frozen build, "
            f"check the PyInstaller datas entry for mep_cmap/schema.")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    schema = NibsSchema(data)
    if use_cache:
        _CACHE[path] = schema
    return schema
