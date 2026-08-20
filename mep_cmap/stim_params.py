"""
mep_cmap.stim_params
~~~~~~~~~~~~~~~~~~~~
Named stimulation parameter sets, and the assignment of a recording's stim
codes to them.

Why this exists
---------------
A recording says a pulse fired on code ``G``. It does not say that ``G`` was a
TMS pulse at 120% of a resting motor threshold of 50% MSO while ``A`` was a
peripheral M-wave at 45 mA. Until now there was one intensity box per file, so
a file containing both could describe neither.

NIBS-BIDS v6.3 answers this by separating the parameters from the deliveries.
``*_nibs.tsv`` holds one row per unique stimulation parameter set, identified
by ``nibs_event_id``; ``*_events.tsv`` holds one row per delivery, naming a
parameter set. A set is written once and referenced many times. This module is
the parameter set.

Why a session-level table rather than fields on each file
---------------------------------------------------------
Because a threshold is not a property of a recording. A study with fifteen
files and one MEP protocol would otherwise state 120% rMT fifteen times, and
correcting a mis-recorded rMT would mean fifteen edits with nothing linking
them. Named sets state it once.

It also removes a guess. If parameters lived on each file, deciding whether two
files share a ``nibs_event_id`` would mean comparing tuples of typed floats,
where ``58``, ``58.0`` and ``58.0001`` are the same number to the analyst and
three different sets to the code. Getting that wrong silently emits either two
parameter sets that should be one, or one that should be two, and nothing
downstream can tell. A name is unambiguous.

What a set is NOT
-----------------
Not a condition. A condition (``pre``, ``post``) says what a trial meant and
determines nothing about the stimulus; a parameter set says what was delivered.
The same set is referenced by trials in both conditions. Where a condition DOES
change the stimulus -- three intensities of a recruitment curve -- that is
three parameter sets, which is what the spec requires: a recovery curve at five
inter-stimulus intervals is five rows, one per ISI.

This module holds no Tk and touches no file: it is the rules, so that they can
be tested without a display and without a recording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

#: `nibs_type` vocabulary, closed in v6.3. PNS arrived in v6.3 and is the one
#: this tool most needs: M-waves, H-reflexes and CMAPs are peripheral, and
#: before v6.3 they had to be misdescribed as TMS, tES or TUS.
NIBS_TYPES = ("TMS", "tES", "TUS", "PNS")

#: Reserved by the spec to separate simultaneous elements within one cell. A
#: name containing it would be read back as two values, so names are rejected
#: rather than quietly rewritten.
DELIMITER = "|"

#: What a BIDS tabular file writes where a column has no value.
NA = "n/a"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


class ParamSetError(ValueError):
    """A parameter set table that cannot be applied, with the reason."""


# ── names ────────────────────────────────────────────────────────────────────

def sanitise_name(text) -> str:
    """A name safe as a ``nibs_event_id`` and as a TSV cell.

    Restricted to letters, digits, underscore and hyphen, which is what the
    worked examples use (``spTMS``, ``SICI``). Anything else collapses to an
    underscore: a tab would split the cell, a ``|`` would be read as two
    values, and a space survives a write but is a poor identifier to reference
    from another file by hand.
    """
    out = _SAFE_NAME.sub("_", str(text or "").strip()).strip("_")
    return out


def deduplicate_names(names) -> list:
    """Make names unique, preserving order and first occurrence.

    ``nibs_event_id`` MUST be unique within ``*_nibs.tsv``, and a duplicate is
    not a validation nicety: two rows with one id makes every reference to it
    ambiguous, and a reader cannot tell which parameters were delivered.
    """
    seen, out = {}, []
    for raw in names:
        base = sanitise_name(raw) or "set"
        if base not in seen:
            seen[base] = 1
            out.append(base)
            continue
        seen[base] += 1
        cand = f"{base}_{seen[base]}"
        while cand in seen:
            seen[base] += 1
            cand = f"{base}_{seen[base]}"
        seen[cand] = 1
        out.append(cand)
    return out


# ── the set ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StimParamSet:
    """One stimulation parameter set: what was delivered, named.

    Frozen, and the helpers return new sets rather than mutating, for the same
    reason ``ConditionRow`` is: the table is edited by a GUI, and an undo that
    reconstructs prior state is harder to get right than one that kept it.

    ``values`` is keyed by SCHEMA FIELD KEY, not by the v6.3 output name. The
    schema's ``emits`` does that translation at write time, so a spec rename
    never touches saved state.
    """

    name:      str                                  # -> nibs_event_id
    nibs_type: str = "TMS"
    values:    dict = field(default_factory=dict)   # schema key -> value
    position:  str = ""                             # -> nibs_position_id

    @property
    def nibs_event_id(self) -> str:
        return self.name

    def with_value(self, key, value) -> "StimParamSet":
        vals = dict(self.values)
        if value in (None, ""):
            vals.pop(key, None)
        else:
            vals[key] = value
        return replace(self, values=vals)

    def describe(self) -> str:
        bits = [self.nibs_type]
        amp = self.values.get("StimulationIntensity")
        if amp not in (None, ""):
            unit = self.values.get("StimulationIntensityUnits") or ""
            bits.append(f"{amp} {unit}".strip())
        ref = self.values.get("IntensityReference")
        if ref:
            scale = self.values.get("IntensityScaling")
            bits.append(f"{scale}x {ref}" if scale else str(ref))
        return f"{self.name}: " + ", ".join(bits)


# ── the table ────────────────────────────────────────────────────────────────

def default_set_for(code, nibs_type="TMS") -> StimParamSet:
    """A blank set named after a stim code.

    So that the ordinary case costs nothing: a file whose codes each mean one
    protocol gets one set per code, already named and already assigned, and the
    analyst fills in intensities rather than inventing names first.
    """
    return StimParamSet(name=sanitise_name(code) or "set", nibs_type=nibs_type)


def ensure_sets_for(codes, sets=(), nibs_type="TMS") -> list:
    """Existing sets, plus a default for any code with no set of that name.

    Additive by design. A code appearing in a newly scanned file must not
    disturb the sets the analyst has already filled in for the others.
    """
    have = {s.name for s in sets}
    out = list(sets)
    for code in codes:
        name = sanitise_name(code)
        if name and name not in have:
            have.add(name)
            out.append(default_set_for(name, nibs_type))
    return out


def validate(sets, assignment=None, codes=()) -> list:
    """Every reason this table could not be written, as plain sentences.

    Returned rather than raised, and all of them rather than the first: a
    dialogue that reports one problem per attempt is a dialogue the analyst
    fights.
    """
    errors = []
    names = [s.name for s in sets]

    for s in sets:
        if not s.name:
            errors.append("A parameter set has no name.")
        elif sanitise_name(s.name) != s.name:
            errors.append(
                f"{s.name!r} contains characters that cannot be written to a "
                f"TSV cell. Use letters, digits, underscore or hyphen.")
        if s.nibs_type not in NIBS_TYPES:
            errors.append(
                f"{s.name or '(unnamed)'}: {s.nibs_type!r} is not a stimulation "
                f"type. One of {', '.join(NIBS_TYPES)}.")

    for name in sorted({n for n in names if names.count(n) > 1}):
        errors.append(
            f"{name!r} is used by more than one parameter set. "
            f"nibs_event_id must be unique.")

    if assignment is not None:
        known = set(names)
        for code, name in sorted(assignment.items()):
            if name and name not in known:
                errors.append(
                    f"Stim code {code!r} is assigned to {name!r}, "
                    f"which is not a parameter set.")
        for code in codes:
            if not assignment.get(code):
                errors.append(
                    f"Stim code {code!r} has no parameter set, so its stimuli "
                    f"cannot be described.")
    return errors


def unassigned(codes, assignment) -> list:
    """Ticked codes with no parameter set. The BIDS-ify status reads this."""
    return [c for c in codes if not (assignment or {}).get(c)]


def sets_in_use(assignment, sets) -> list:
    """Only the sets some code actually references, in table order.

    ``*_nibs.tsv`` describes this recording. A session-level set that no code in
    this file uses belongs to a different file and writing it here would
    describe stimulation that was never delivered.
    """
    used = {n for n in (assignment or {}).values() if n}
    return [s for s in sets if s.name in used]


# ── projection to NIBS-BIDS ──────────────────────────────────────────────────

def nibs_rows(sets, schema) -> tuple:
    """``(columns, rows)`` for ``*_nibs.tsv``, one row per parameter set.

    Driven entirely by the schema: a field lands in this file because its
    ``block`` says ``nibs.tsv``, and lands under the name its ``emits`` says.
    Nothing here lists columns, so adding a v6.3 field is a schema edit and no
    code change -- and, more to the point, a field cannot be added to the
    dialogue and forgotten in the writer, which is the failure this codebase
    keeps repeating.

    Columns are the union of what the given sets actually populate, plus the
    two identifiers, so a simple study does not carry forty empty columns.
    """
    fields = [f for f in schema.fields if f.block == "nibs.tsv" and not f.legacy]
    by_key = {f.key: f for f in fields}

    populated = []
    for f in fields:
        if any(s.values.get(f.key) not in (None, "") for s in sets):
            populated.append(f)

    columns = ["nibs_event_id", "nibs_type"]
    for f in populated:
        name = f.emits or f.key
        if name not in columns:
            columns.append(name)

    rows = []
    for s in sets:
        row = {"nibs_event_id": s.name, "nibs_type": s.nibs_type}
        for f in populated:
            name = f.emits or f.key
            val = s.values.get(f.key)
            # First writer wins when two keys emit the same column. They are
            # alternatives for one v6.3 column (Current and
            # StimulationIntensity both emit stimulus_intensity), never two
            # values of it, so a later blank must not erase an earlier value.
            if row.get(name) in (None, "", NA) or val not in (None, ""):
                if val not in (None, ""):
                    row[name] = val
                else:
                    row.setdefault(name, NA)
        for name in columns:
            row.setdefault(name, NA)
        rows.append(row)
    return columns, rows


def units_sidecar(sets, schema) -> dict:
    """``{column: {"Units": ...}}`` for the columns written above.

    The spec is explicit that units are always stated in the sidecar and never
    assumed from the numbers in the table, so a numeric column without this is
    not merely undocumented, it is unreadable: 58 could be percent of maximum
    stimulator output or milliamps.
    """
    columns, _ = nibs_rows(sets, schema)
    out = {}
    for f in schema.fields:
        if f.block != "nibs.tsv" or f.legacy:
            continue
        name = f.emits or f.key
        if name in columns and f.units:
            out.setdefault(name, {"Units": f.units})
    return out


# ── persistence ──────────────────────────────────────────────────────────────

def to_dicts(sets) -> list:
    return [{"name": s.name, "nibs_type": s.nibs_type,
             "values": dict(s.values), "position": s.position} for s in sets]


def from_dicts(data) -> list:
    """Rebuild from saved state, tolerating anything a hand edit could do.

    A malformed entry is skipped rather than raising: this is loaded at
    startup, and a state file that cannot be read must not stop the tool
    opening on a study the analyst still needs.
    """
    out = []
    for d in (data or []):
        if not isinstance(d, dict):
            continue
        name = sanitise_name(d.get("name"))
        if not name:
            continue
        ntype = d.get("nibs_type") or "TMS"
        out.append(StimParamSet(
            name=name,
            nibs_type=ntype if ntype in NIBS_TYPES else "TMS",
            values=dict(d.get("values") or {}),
            position=str(d.get("position") or ""),
        ))
    return out
