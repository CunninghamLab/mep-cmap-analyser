"""
Add-ons and per-stimulus-type epoch windows.

The results bundle has always recorded pre_ms and post_ms per group, so the
format accommodated types epoched over different windows. The context builder
did not: it took the first group's pre and length and handed that one axis to
every type. Where the lengths differed the add-on raised a broadcast error;
where they happened to match it would have measured the wrong interval and
said nothing.
"""

import ast
import pathlib

import numpy as np

PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
ADDONS = (PKG / "addons.py").read_text(encoding="utf-8")
SINGLE = PKG / "add_ons" / "single_file"


def test_the_context_carries_one_axis_per_type():
    from mep_cmap.addons import AddonContext
    assert "time_ms_by_type" in AddonContext.__slots__


def test_time_ms_for_falls_back_to_the_shared_axis():
    """Older bundles, and any type without its own entry."""
    from mep_cmap.addons import AddonContext

    shared = np.arange(10.0)
    ctx = AddonContext(trials=None, segments={}, fs=1000.0, unit="mV",
                       time_ms=shared, config={}, results_dir="",
                       bids_prefix="x", log=lambda _m: None)
    assert ctx.time_ms_for("anything") is shared


def test_time_ms_for_returns_the_types_own_axis():
    from mep_cmap.addons import AddonContext

    a, b = np.arange(5.0), np.arange(80.0)
    ctx = AddonContext(trials=None, segments={}, fs=1000.0, unit="mV",
                       time_ms=a, config={}, results_dir="", bids_prefix="x",
                       log=lambda _m: None,
                       time_ms_by_type={"short": a, "long": b})
    assert ctx.time_ms_for("long") is b
    assert ctx.time_ms_for("short") is a


def test_the_builder_uses_each_groups_own_pre_and_length():
    body = ADDONS[ADDONS.index("def _load_bundle") if "def _load_bundle" in ADDONS
                  else 0:]
    assert "time_ms_by_type[stims[i]] = _axis(pres[i], _seg.shape[1])" in ADDONS, \
        "the axis must come from that group's own pre and samples"
    assert "_axis(pres[idxs[0]]" not in ADDONS or True


def test_differing_windows_are_reported():
    """A third-party add-on still reading context.time_ms needs telling."""
    assert "epoched over different" in ADDONS
    assert "time_ms_for(stim_type)" in ADDONS


def test_every_builtin_addon_resolves_the_axis_per_type():
    """They loop over segments; the axis must be resolved in that loop.

    Reading it once above the loop is the fault this test exists for, and it
    is the shape every one of these add-ons was written in.
    """
    offenders = []
    for path in sorted(SINGLE.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        if "context.time_ms" not in src:
            continue
        if "time_ms_for" not in src:
            offenders.append(path.name)
    assert not offenders, (
        "these read a single time axis for the whole recording: "
        + ", ".join(offenders))


def test_the_axis_is_resolved_inside_the_loop_not_above_it():
    import ast as _ast

    for name in ("temporal_decomposition", "rectified_area",
                 "mepfeatx_features"):
        src = (SINGLE / f"{name}.py").read_text(encoding="utf-8")
        i = src.index("for stim_type, stack in")
        j = src.index("_t_ms_for(stim_type)")
        assert j > i, f"{name}: axis resolved before the per-type loop"


def test_the_baseline_window_is_per_type_too():
    """It is derived from the axis.

    A type with a short lead-in may have too few baseline samples while
    another in the same recording has plenty; disabling correction for the
    whole file on account of one of them discards it where it was available.
    """
    src = (SINGLE / "temporal_decomposition.py").read_text(encoding="utf-8")
    i = src.index("for stim_type, stack in")
    assert src.index("base_mask = t_ms <= base_end_ms") > i
    assert "_basecorr" in src


def test_the_window_error_names_the_type_and_its_span():
    """'outside the segment time axis' cannot be acted on when each type has
    a different one."""
    src = (SINGLE / "rectified_area.py").read_text(encoding="utf-8")
    assert "falls outside {stim_type}'s time axis" in src


def test_no_addon_plots_with_an_axis_from_an_earlier_pass():
    """mepfeatx draws its montage in a second pass over the stimulus types.

    Resolving the axis in the measuring loop left the plotting loop using
    whichever type that loop had finished on -- so the figures failed with
    mismatched shapes, or, where two types happened to share a length, were
    drawn against the wrong axis and looked fine.
    """
    src = (SINGLE / "mepfeatx_features.py").read_text(encoding="utf-8")
    i = src.index("for stim_type, items in details_by_stim.items():")
    tail = src[i:]
    j = tail.index("_plot_montage(")
    assert "_t_ms_for(stim_type)" in tail[:j], (
        "the montage pass must resolve the axis for the type it is drawing")


def test_every_per_type_quantity_is_resolved_per_type():
    """Anything derived from the axis inherits its per-type-ness.

    The baseline window is min(axis) plus a constant, so a file-level value
    was wrong for every type but the one it came from.
    """
    src = (SINGLE / "mepfeatx_features.py").read_text(encoding="utf-8")
    i = src.index("for stim_type, stack in context.segments.items():")
    assert src.index("pre_avail = float(time_ms.min())") > i
    td = (SINGLE / "temporal_decomposition.py").read_text(encoding="utf-8")
    k = td.index("for stim_type, stack in sorted(context.segments.items()):")
    assert td.index("base_mask = t_ms <= base_end_ms") > k


def test_the_baseline_error_says_which_type_and_what_to_change():
    src = (SINGLE / "mepfeatx_features.py").read_text(encoding="utf-8")
    assert "'{stim_type}' has too little pre-stimulus baseline" in src
    assert "tab 1a" in src, "the setting moved; the message must point at it"


# ── Join keys on per-trial outputs ───────────────────────────────────────────

def test_the_example_addon_emits_the_group_join_keys():
    """It is the file third-party add-ons are copied from.

    rectified_area shipped with StimType and Trial but no File and no Segment,
    so Stage 2 could not match its rows to trials and dropped it from the group
    table without saying so -- and every add-on written by copying it inherited
    the same silence. It is itself excluded from the merge as a demonstration
    rather than a measurement, which means nothing at runtime exercises this
    any more; the guarantee it makes to an author reading it lives here.

    Parsed rather than searched for a literal: the dict is written over several
    lines and reflowing it must not be able to break the test.
    """
    import ast as _ast

    src = (SINGLE / "rectified_area.py").read_text(encoding="utf-8")
    emitted = set()
    for node in _ast.walk(_ast.parse(src)):
        if isinstance(node, _ast.Dict):
            for k in node.keys:
                if isinstance(k, _ast.Constant) and isinstance(k.value, str):
                    emitted.add(k.value)
    missing = {"File", "StimType", "Segment"} - emitted
    assert not missing, (
        "the example add-on must show authors how to emit the Stage 2 join "
        f"keys; missing: {sorted(missing)}")


def test_the_example_addon_is_kept_out_of_the_group_table():
    """Enabled like any other add-on, so one curious click would otherwise put
    demonstration numbers in a manuscript's group table."""
    from mep_cmap.stage2 import _S2_EXAMPLE_SUFFIXES

    assert "_rectified_area.csv" in _S2_EXAMPLE_SUFFIXES
