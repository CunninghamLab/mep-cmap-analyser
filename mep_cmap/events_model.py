"""
mep_cmap.events_model
~~~~~~~~~~~~~~~~~~~~~
The single answer to "what happened, and when" for one recording.

Why this exists
---------------
Two writers used to produce an ``_events.tsv``: the Conditions tab wrote one
beside the source recording, and BIDS-ify wrote one into rawdata. They read
different state and ran at different times, so a recording could end up with two
files claiming to be its events and disagreeing about them. A reader has no way
to tell which is right.

Both now project from here, so there is one answer rather than two.

The raw arrangement is kept
---------------------------
``raw_events`` is what the recording itself said: an onset and a stim code per
delivery, captured once when the file is first scanned and never rewritten.
Conditions are a LAYER over it, not a replacement.

That separation is what makes a mistake survivable. Splitting code ``A`` into
100 mA and 150 mA halves is a judgement about trials, and a judgement can be
wrong; reverting is then deleting the layer rather than reconstructing what the
file originally said from a file that has since been overwritten. It also lets
the tool tell "no conditions set" from "conditions that happen to match the raw
arrangement", which a single merged copy cannot express.

This module holds no Tk and reads no files.
"""

from __future__ import annotations

NA = "n/a"


def stim_times_from_raw(raw_events) -> dict:
    """``{code: [onset, ...]}`` in file order.

    Order matters and is not incidental: a ConditionRow names trials by INDEX
    within its stim type, so re-sorting here would silently reassign which
    trials belong to which condition.
    """
    out = {}
    for ev in (raw_events or []):
        code = str(ev.get("code", "") or "")
        if not code:
            continue
        out.setdefault(code, []).append(float(ev.get("onset", 0.0)))
    return out


def raw_from_stim_times(stim_times) -> list:
    """The inverse, sorted by onset. For capturing a scan into storage."""
    out = []
    for code, times in (stim_times or {}).items():
        for t in (times or []):
            out.append({"onset": float(t), "code": str(code)})
    out.sort(key=lambda e: e["onset"])
    return out


def project(raw_events, condition_rows=None, code_sets=None,
            param_sets=None, duration=0.0) -> tuple:
    """``(columns, rows)`` for an ``_events.tsv``.

    ``condition_rows`` is optional. Without it the raw arrangement is written
    as-is, which is the honest description of a recording nobody has grouped
    yet -- not an empty file, and not invented conditions.

    ``code_sets`` maps a stim code, or a ``code`` + ``condition`` pair, to a
    stimulation parameter set. The pair form is what makes a split expressible:
    half of ``A`` at 100 mA and half at 150 mA is two parameter sets, and the
    condition is the only thing that distinguishes which trial got which.
    """
    from . import conditions as C

    code_sets = dict(code_sets or {})
    pos_for = {s.name: (s.position or f"{s.name}_position")
               for s in (param_sets or [])}

    if condition_rows:
        stim_times = stim_times_from_raw(raw_events)
        rows = [dict(r) for r in C.to_event_rows(condition_rows, stim_times,
                                                 duration=duration)]
    else:
        rows = [{"onset": float(ev.get("onset", 0.0)),
                 "duration": float(duration),
                 "trial_type": str(ev.get("code", "") or NA),
                 "condition": NA}
                for ev in sorted(raw_events or [],
                                 key=lambda e: float(e.get("onset", 0.0)))]

    linked = bool(code_sets)
    for r in rows:
        code = r.get("trial_type") or NA
        cond = r.get("condition") or NA
        if linked:
            # The pair wins over the bare code, so a split overrides the
            # protocol the code carried before it was split.
            name = (code_sets.get(_pair_key(code, cond))
                    or code_sets.get(code) or "")
            r["nibs_event_id"] = name or NA
            r["nibs_position_id"] = pos_for.get(name, NA)

    columns = ["onset", "duration", "trial_type", "condition"]
    if linked:
        columns += ["nibs_event_id", "nibs_position_id"]
    # Written only when it says something. A column of n/a on every row is
    # noise in a file meant to be read by other people's scripts.
    if all((r.get("condition") or NA) == NA for r in rows):
        columns.remove("condition")
        for r in rows:
            r.pop("condition", None)
    for r in rows:
        for c in columns:
            r.setdefault(c, NA)
    return columns, rows


def _pair_key(code, condition) -> str:
    """Key for a code split by condition. Kept in one place so the writer and
    the assignment UI cannot disagree about its shape."""
    return f"{code}\u00b7{condition}"


def pair_key(code, condition) -> str:
    return _pair_key(code, condition)


def split_codes(condition_rows) -> list:
    """``(code, condition)`` pairs a condition table actually creates.

    Only codes carrying more than one condition are returned: a code with a
    single condition is not split, and asking the analyst to assign a parameter
    set to each half of something that was never halved is busywork.

    An unnamed row COUNTS towards the split but is reported through
    :func:`unnamed_splits`, not here. A code with one named group and one blank
    one is split -- ten trials treated differently from the other hundred and
    fifty is exactly the case this exists for -- but the blank group cannot be
    described until it is called something, so it is raised as a problem to fix
    rather than silently dropped, which is what hid it before.
    """
    by_code = _conditions_by_code(condition_rows)
    out = []
    for code, conds in by_code.items():
        if len(conds) < 2:
            continue
        out.extend((code, c) for c in sorted(c for c in conds if c))
    return out


def unnamed_splits(condition_rows) -> list:
    """Codes split into groups where at least one group has no name.

    A condition that reaches a published events file should be something the
    analyst chose to call something. A blank one cannot be referenced from
    *_nibs.tsv, cannot be told apart from its siblings by a reader, and would
    otherwise vanish from the assignment table with no explanation.
    """
    out = []
    for code, conds in _conditions_by_code(condition_rows).items():
        if len(conds) > 1 and any(not c for c in conds):
            out.append(code)
    return sorted(out)


def _conditions_by_code(condition_rows) -> dict:
    """``{code: {condition, ...}}``, blanks included."""
    by_code = {}
    for row in (condition_rows or []):
        by_code.setdefault(row.stim_type, set()).add(row.condition or "")
    return by_code


def rows_from_events(event_rows):
    """Rebuild a condition table from a written ``_events.tsv``.

    The round trip that was missing. Converting a recording writes the
    conditions into the events file, but reopening the CONVERTED recording
    built a fresh table from stim types alone, because the reader took
    ``trial_type`` and discarded ``condition``. The grouping was sitting in the
    file and was thrown away on the way back in.

    Trials are numbered within their stim type, in onset order, because that is
    what a ConditionRow means by a trial index. Rows come back in first-seen
    order so a table reopened looks like the one that was saved.

    Returns [] when nothing carries a condition, which is the honest answer for
    a recording that was never grouped: the caller then builds its usual
    one-row-per-stim-type table.
    """
    from .conditions import ConditionRow

    ordered = sorted(event_rows or [],
                     key=lambda r: float(r.get("onset") or 0.0))
    seen_order, by_pair, counters = [], {}, {}
    for r in ordered:
        stim = str(r.get("trial_type") or "").strip()
        cond = str(r.get("condition") or "").strip()
        if stim in ("", NA):
            # An excluded trial still consumes an index within its stim type,
            # but there is no stim type recorded to consume it in. Skipping it
            # would shift every later trial by one.
            continue
        idx = counters.get(stim, 0)
        counters[stim] = idx + 1
        if cond in ("", NA):
            continue
        key = (stim, cond)
        if key not in by_pair:
            by_pair[key] = []
            seen_order.append(key)
        by_pair[key].append(idx)

    return [ConditionRow(stim_type=stim, condition=cond,
                         trials=tuple(by_pair[(stim, cond)]))
            for (stim, cond) in seen_order]


def describe_source(condition_rows) -> str:
    """What the events file is, for its own sidecar. Provenance, not decoration:
    a reader needs to know whether the grouping came from the recording or from
    someone's judgement about it."""
    if condition_rows:
        return ("Stimulus codes as recorded, grouped into conditions in the "
                "Conditions tab. The unconditioned arrangement is retained and "
                "can be restored.")
    return "Stimulus codes exactly as recorded. No conditions have been applied."
