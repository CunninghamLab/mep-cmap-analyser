"""
mep_cmap.selection_summary
~~~~~~~~~~~~~~~~~~~~~~~~~~
Describe what a set of time ranges actually contains.

The crop dialogue showed the ranges as times -- "[12.40 - 88.10]" -- which says
nothing about the events inside them. A recruitment curve, a set of 120% aMT
trials and an iSP block can sit in one recording with no visible boundary
between them, and a collaborator reported having to count tick marks by eye to
work out which events were the first 90.

Times are the wrong unit for that question. What the analyst is choosing is a
set of TRIALS, so the summary is written in trials: how many of each stimulus
type, and where they sit in the file's own numbering.

Index conventions
-----------------
Indices are 1-based and per stimulus type, matching the Data Inspector's
"C - segment 3/15" -- two different numbers for the same trial would be worse
than none.

A selection of several ranges can take a discontinuous set of one type, so the
spans are reported as they are (``#1-20, #55-70``) rather than collapsed to
their outer bounds. Collapsing would claim 70 trials where 35 were chosen.

  * summarise_selection  -- the counts and spans
  * format_selection     -- the one-line description
"""

from collections import namedtuple

TypeSelection = namedtuple(
    "TypeSelection", ["stim_type", "n_selected", "n_total", "spans"])


def _contiguous_spans(indices):
    """Group sorted 1-based indices into (first, last) runs."""
    spans = []
    for i in indices:
        if spans and i == spans[-1][1] + 1:
            spans[-1][1] = i
        else:
            spans.append([i, i])
    return [(a, b) for a, b in spans]


def summarise_selection(stim_times_by_type, ranges):
    """Which events fall inside ``ranges``.

    Parameters
    ----------
    stim_times_by_type : {stim_type: [t_seconds, ...]}
    ranges             : [(t0, t1), ...] in seconds, any order, may overlap

    Returns
    -------
    list[TypeSelection], ordered by stimulus type, containing only types with
    at least one event selected.
    """
    out = []
    for stim_type, times in sorted((stim_times_by_type or {}).items()):
        ordered = sorted(float(t) for t in times)
        picked = [i for i, t in enumerate(ordered, start=1)
                  if any(lo <= t <= hi for lo, hi in
                         ((min(r), max(r)) for r in (ranges or [])))]
        if picked:
            out.append(TypeSelection(stim_type, len(picked), len(ordered),
                                     _contiguous_spans(picked)))
    return out


def format_selection(stim_times_by_type, ranges, max_spans=3):
    """One line describing the selection, for display under the plot.

    ``max_spans`` caps how many index spans are listed per type before the rest
    are summarised as "(+n more)": a fragmented selection could otherwise wrap
    over several lines and be harder to read than no summary at all.
    """
    if not ranges:
        return "No ranges yet — drag on the plot."

    sel = summarise_selection(stim_times_by_type, ranges)
    n_ranges = len(ranges)
    head = f"Selection: {n_ranges} range{'s' if n_ranges != 1 else ''}"

    if not sel:
        return f"{head} — no stimulus events inside"

    parts = []
    for s in sel:
        shown = s.spans[:max_spans]
        span_txt = ", ".join(f"#{a}" if a == b else f"#{a}\u2013{b}"
                             for a, b in shown)
        if len(s.spans) > max_spans:
            span_txt += f" (+{len(s.spans) - max_spans} more)"
        parts.append(f"{s.stim_type}: {s.n_selected} event"
                     f"{'s' if s.n_selected != 1 else ''} "
                     f"({span_txt} of {s.n_total})")

    return head + " \u00b7 " + " \u00b7 ".join(parts)
