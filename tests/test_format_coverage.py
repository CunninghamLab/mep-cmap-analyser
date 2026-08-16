"""
Regression tests for format handling.

Two classes of bug motivated these, and neither was visible to reader-level
tests, because in both cases the format readers themselves were correct:

1. **A format could be readable but not wired into the workflow.**
   ``_browse_file_path()`` dispatches on the result of ``detect_format()`` to
   populate ``stim_events``.  A format with no branch fell through to the
   Spike2 text scanner, which found nothing, so ``stim_types_found`` was empty,
   ``_build_labels_tab()`` was never called, and the GUI stalled after the crop
   step with no error.  BrainVision, AcqKnowledge (.acq and .mat), Brainsight
   and LabChart MATLAB were all affected.  ``test_every_format_has_load_branch``
   fails if a new reader is added without a branch.

2. **The marker argument silently dropped stimulus types.**
   ``brainvision.extract_stim_times`` treated ``marker_name`` as a filter, so
   once the marker dropdown selected one label the others vanished — which on a
   paired-pulse recording means losing the conditioned or reference condition
   with no warning.  ``test_marker_name_never_drops_stim_types`` locks the
   EDF convention: event labels define the types, the argument is ignored.

Also locked here: 1-based .vmrk marker positions, and normalisation to mV.
"""

import ast
import pathlib

import numpy as np
import pytest

from conftest import (BV_CH0_PEAK_DIGITAL, BV_CHANNELS, BV_FS, BV_MARKERS,
                      BV_N_SAMPLES, BV_RESOLUTION)

REPO = pathlib.Path(__file__).resolve().parents[1]
PKG = REPO / "mep_cmap"


# ── Static analysis helpers ───────────────────────────────────────────────────

def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _detectable_formats():
    """Every string literal detect_format() can return."""
    fn = _find_function(_parse(PKG / "io.py"), "detect_format")
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            out.add(node.value.value)
    return out


def _fmt_constants(test_node, var="_fmt"):
    """Format literals compared against `var` in a single if/elif test."""
    out = set()
    for node in ast.walk(test_node):
        if not isinstance(node, ast.Compare):
            continue
        if not (isinstance(node.left, ast.Name) and node.left.id == var):
            continue
        for comp in node.comparators:
            if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                out.add(comp.value)
            elif isinstance(comp, (ast.Tuple, ast.List, ast.Set)):
                for elt in comp.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        out.add(elt.value)
    return out


def _stored_names(stmts):
    """Names assigned to across a list of statements."""
    out = set()
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                out.add(node.id)
    return out


def _fmt_dispatch_chains(func_name, module_path):
    """
    Return [(stored_names, formats)] for each top-level if/elif chain in the
    function that dispatches on _fmt.

    _browse_file_path contains two such chains — the load chain (which
    populates stim_events) and the stim_types_found chain.  They must be
    inspected separately: scanning the whole function unions them, so deleting
    a format from the load chain goes undetected because the name still
    appears in the other one.
    """
    fn = _find_function(_parse(module_path), func_name)
    ifs = [n for n in ast.walk(fn) if isinstance(n, ast.If)]
    continuation = {id(n.orelse[0]) for n in ifs
                    if len(n.orelse) == 1 and isinstance(n.orelse[0], ast.If)}

    chains = []
    for head in ifs:
        if id(head) in continuation:
            continue
        formats, stored, node = set(), set(), head
        while True:
            formats |= _fmt_constants(node.test)
            stored |= _stored_names(node.body)
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                node = node.orelse[0]
                continue
            stored |= _stored_names(node.orelse)
            break
        if formats:
            chains.append((stored, formats))
    return chains


def _load_chain_formats():
    """
    Formats handled by the *load* chain — the one that populates stim_events.
    Identified as the _fmt chain that does not assign stim_types_found.
    """
    candidates = [(stored, fmts)
                  for stored, fmts in _fmt_dispatch_chains(
                      "_browse_file_path", PKG / "app.py")
                  if "stim_types_found" not in stored and len(fmts) > 1]
    assert len(candidates) == 1, (
        f"expected exactly one load chain, found {len(candidates)}; "
        f"the chain-identification heuristic in this test needs updating")
    return candidates[0][1]


# ── 1. Workflow coverage ──────────────────────────────────────────────────────

# Not a format: a sentinel meaning "no reader recognises this file". It has no
# load branch by design -- the load stops and says so, rather than continuing
# into a reader that cannot help.
SENTINEL_FORMATS = {"unsupported_binary"}


def test_detect_format_returns_known_set():
    """Guard the guard: if this list changes, the tests below must be reviewed."""
    assert _detectable_formats() - SENTINEL_FORMATS == {
        "spike2", "spike2_smr", "labchart", "labchart_mat", "cfwb",
        "generic_tsv", "edf", "brainvision", "brainsight",
        "acqknowledge_acq", "acqknowledge_mat", "mne",
        "epoched_mat",
    }


def test_the_unsupported_sentinel_stops_the_load():
    """
    It must not simply lack a branch -- that is the silent stall this file
    exists to prevent. It needs an explicit branch that reports and returns.
    """
    import pathlib as _pl

    src = (_pl.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "app.py").read_text(encoding="utf-8")
    a = src.index('if _fmt == "unsupported_binary":')
    b = src.index("if _fmt == 'generic_tsv'", a)
    assert "return" in src[a:b]


def test_every_format_has_load_branch():
    """
    Every format detect_format() can return must be handled explicitly in
    _browse_file_path().  A missing branch stalls the GUI silently.
    """
    detectable = _detectable_formats() - SENTINEL_FORMATS
    handled = _load_chain_formats()
    missing = detectable - handled
    assert not missing, (
        f"Formats readable but not wired into the workflow: {sorted(missing)}. "
        f"Add a branch to _browse_file_path() that populates stim_events "
        f"(marker-based formats) or sets marker_choice (trigger-channel "
        f"formats), otherwise _build_labels_tab() is never called.")


def test_spike2_branch_is_explicit():
    """
    The Spike2 text scanner must be an explicit `elif _fmt == 'spike2'`, not the
    trailing `else`.  As the else, it silently swallowed every unhandled format.
    """
    assert "spike2" in _load_chain_formats()


def test_readers_have_the_three_public_functions():
    """Every format module must satisfy the io.py contract."""
    required = {"list_waveform_channels", "extract_emg_waveform_and_fs",
                "extract_stim_times"}
    for path in sorted((PKG / "formats").glob("*.py")):
        if path.name == "__init__.py":
            continue
        names = {n.name for n in ast.walk(_parse(path))
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert required <= names, f"{path.name} missing {sorted(required - names)}"


# ── 2. BrainVision reader contract ────────────────────────────────────────────

def test_channels_and_sampling_rate(brainvision_triplet):
    from mep_cmap import io
    assert io.detect_format(str(brainvision_triplet)) == "brainvision"
    assert io.list_waveform_channels(str(brainvision_triplet)) == BV_CHANNELS
    _, fs, _ = io.extract_emg_waveform_and_fs(str(brainvision_triplet), 0)
    assert fs == BV_FS


def test_vmrk_positions_are_one_based(brainvision_triplet):
    """
    .vmrk positions are 1-based: the leading 'New Segment' marker sits at
    position 1, denoting the first sample (t = 0).  Reading them as 0-based
    placed every event one sample late — 0.2 ms at 5 kHz, 1 ms at 1 kHz,
    biasing every onset latency.
    """
    from mep_cmap import io
    stim = io.extract_stim_times(str(brainvision_triplet), "")
    expected = {}
    for mtype, desc, pos in BV_MARKERS:
        if mtype == "New Segment":
            continue
        expected.setdefault(desc.strip(), []).append((pos - 1) / BV_FS)
    assert {k: sorted(v) for k, v in stim.items()} == \
           {k: sorted(v) for k, v in expected.items()}
    # Exact, not approximate: 5001 -> 1.0 s, not 1.0002 s
    assert stim["S  1"][0] == 1.0


def test_new_segment_marker_is_excluded(brainvision_triplet):
    from mep_cmap import io
    assert "New Segment" not in io.extract_stim_times(str(brainvision_triplet), "")


@pytest.mark.parametrize("marker_name",
                         ["", "A", "S  1", "S  2", "Keyboard", None])
def test_marker_name_never_drops_stim_types(brainvision_triplet, marker_name):
    """
    The file's event labels define the stimulus types.  marker_name is accepted
    for API parity and must never filter — otherwise selecting one label in the
    dropdown silently discards every other condition in a paired-pulse design.
    """
    from mep_cmap import io
    stim = io.extract_stim_times(str(brainvision_triplet), marker_name)
    assert set(stim) == {"S  1", "S  2"}
    assert all(len(v) == 2 for v in stim.values())


@pytest.mark.parametrize("ext", [".vhdr", ".vmrk", ".eeg"])
def test_any_triplet_member_resolves(brainvision_triplet, ext):
    """Selecting any member of the triplet must resolve to the same recording."""
    from mep_cmap import io
    path = str(brainvision_triplet.with_suffix(ext))
    assert io.detect_format(path) == "brainvision"
    assert io.list_waveform_channels(path) == BV_CHANNELS


# ── 3. Unit normalisation ─────────────────────────────────────────────────────

def test_microvolt_source_is_normalised_to_mV(brainvision_triplet):
    """
    LAT_COLS / SUM_HDR hardcode '(mV)'.  A microvolt source must be scaled, or
    amplitudes are reported 1000x too large in a column labelled mV.
    """
    from mep_cmap import io
    emg, _, unit = io.extract_emg_waveform_and_fs(str(brainvision_triplet), 0)
    assert unit == "mV"
    expected_mV = 2 * BV_CH0_PEAK_DIGITAL * BV_RESOLUTION / 1000.0   # 0.2 mV
    assert np.ptp(emg) == pytest.approx(expected_mV, rel=1e-12)
    assert emg.size == BV_N_SAMPLES


@pytest.mark.parametrize("unit,scale", [
    ("V", 1e3), ("volts", 1e3),
    ("mV", 1.0), ("*mV*", 1.0),
    ("uV", 1e-3), ("\u00b5V", 1e-3), ("\u03bcV", 1e-3),
    (" (\u00b5V) ", 1e-3), ("microvolts", 1e-3),
    ("nV", 1e-6),
])
def test_to_mV_recognised_units(unit, scale):
    from mep_cmap import io
    emg, out_unit = io._to_mV(np.array([1.0, -2.0]), unit)
    assert out_unit == "mV"
    assert emg[0] == pytest.approx(scale, rel=1e-12)


@pytest.mark.parametrize("unit", [None, "", "arbitrary", "counts"])
def test_to_mV_passes_unknown_units_through_untouched(unit):
    """Never guess: an unrecognised unit must not be scaled or relabelled."""
    from mep_cmap import io
    src = np.array([1.0, -2.0])
    emg, out_unit = io._to_mV(src.copy(), unit)
    assert out_unit == unit
    assert np.array_equal(emg, src)


# ── 4. Parity against MNE-Python ──────────────────────────────────────────────

def test_brainvision_matches_mne(brainvision_triplet):
    """
    Cross-check the native reader against the reference implementation.  This
    comparison is what originally exposed the 1-based marker-position bug.
    """
    mne = pytest.importorskip("mne", reason="MNE is an optional extra")
    mne.set_log_level("ERROR")
    from mep_cmap import io

    # importorskip only catches MNE being absent. MNE loads its submodules
    # lazily, so `import mne` can succeed while the reader machinery cannot
    # load at all — most often an MNE built against a SciPy that has since
    # removed a symbol it imports (scipy.special.sph_harm, gone in SciPy 1.17).
    # That is an environment mismatch, not a regression in this package, so it
    # skips rather than fails.
    try:
        raw = mne.io.read_raw_brainvision(str(brainvision_triplet), preload=True)
    except ImportError as exc:
        pytest.skip(f"MNE is installed but its readers cannot load ({exc}); "
                    f"usually SciPy 1.17+ against an older MNE")

    assert raw.ch_names == io.list_waveform_channels(str(brainvision_triplet))
    assert int(raw.info["sfreq"]) == BV_FS

    # MNE returns volts; io normalises to mV. Compare in mV.
    emg, _, _ = io.extract_emg_waveform_and_fs(str(brainvision_triplet), 0)
    np.testing.assert_allclose(emg, raw.get_data(picks=[0])[0] * 1e3,
                               rtol=1e-9, atol=1e-12)

    ours = io.extract_stim_times(str(brainvision_triplet), "")
    theirs = {}
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        full = str(desc)
        # Structural markers, not stimuli. Test the WHOLE description, not just
        # the part after the last '/': MNE versions differ on whether the
        # leading 'New Segment' marker becomes an annotation at all, and when
        # it does its description is 'New Segment/' — whose trailing segment is
        # the empty string, not 'New Segment'.
        if any(k in full.lower() for k in ("new segment", "boundary")):
            continue
        label = full.rsplit("/", 1)[-1].strip()
        if not label:
            continue
        theirs.setdefault(label, []).append(float(onset))

    assert set(ours) == set(theirs)
    for label in ours:
        np.testing.assert_allclose(sorted(ours[label]), sorted(theirs[label]),
                                   rtol=0, atol=1e-12)
