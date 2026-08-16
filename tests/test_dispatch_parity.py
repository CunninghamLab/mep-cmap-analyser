"""
Guard: the pipeline and the Data Inspector must detect onsets identically.

The inspector re-runs onset detection when a segment is reprocessed, and the
value it produces replaces one the pipeline computed. If the two paths disagree
about which algorithm runs, or about what parameters it runs with, a review
step silently changes the measurement -- and nothing errors, because both
numbers are plausible latencies.

Three ways they had already diverged before ``detection/dispatch.py`` existed:

1. The inspector knew four methods. Selecting ``rms_envelope``, ``cusum`` or
   ``consensus`` fell through its ``else`` branch to peak-fraction.
2. The inspector forwarded no ``min_peak_amplitude``, using the detector's
   built-in 0.05 while the pipeline used the configured value.
3. The inspector forwarded no ``peak_frac`` or ``slope_threshold``. The
   detector defaults ``slope_threshold`` to 0.05 and ``PipelineConfig``
   defaults it to 0.08, so the paths disagreed even at factory settings.

These tests compare the two entry points directly rather than checking that
each is individually reasonable, because "individually reasonable" is exactly
what the divergent versions looked like.
"""

import numpy as np
import pytest

from mep_cmap.detection import ONSET_METHOD_LABELS, detector_params, dispatch_onset
from mep_cmap.detection.defaults import DETECTION_DEFAULTS
from mep_cmap.pipeline import PipelineConfig, _detect_onset_dispatch

FS = 5000.0
PRE_MS = 100
POST_MS = 200
TRUE_ONSET = 22.0

ALL_METHODS = sorted(ONSET_METHOD_LABELS)


def make_trial(seed=0, noise=0.02, amp=1.0, onset_ms=TRUE_ONSET, dur_ms=18.0):
    rng = np.random.default_rng(seed)
    n = int((PRE_MS + POST_MS) * FS / 1000)
    stim = int(PRE_MS * FS / 1000)
    x = rng.normal(0.0, noise, n)
    i0 = stim + int(onset_ms * FS / 1000)
    i1 = stim + int((onset_ms + dur_ms) * FS / 1000)
    k = np.arange(i1 - i0) / float(i1 - i0)
    x[i0:i1] += amp * np.sin(2 * np.pi * k) * np.sin(np.pi * k) ** 0.5
    return x


def inspector_style_call(signal, cfg):
    """
    Reproduce how DataInspectorWindow invokes detection.

    The inspector builds one parameter dict and hands it to dispatch_onset with
    its own window arguments. It does not import PipelineConfig, so this
    mirrors the call rather than reusing the pipeline wrapper -- otherwise the
    test would compare the pipeline against itself and prove nothing.
    """
    params = dict(DETECTION_DEFAULTS)
    params.update(detector_params(cfg))
    return dispatch_onset(
        signal, FS, params,
        pre_ms=cfg.pre_ms,
        search_start_ms=cfg.ptp_start,
        search_end_ms=cfg.ptp_end,
        min_latency_ms=15.0,
        max_latency_ms=35.0,
    )


def _cfg(**kw):
    base = dict(pre_ms=PRE_MS, post_ms=POST_MS, ptp_start=5, ptp_end=60,
                latency_map={"TMS": (15.0, 35.0)})
    base.update(kw)
    return PipelineConfig(**base)


# ── Parity across every registered method ─────────────────────────────────────

@pytest.mark.parametrize("method", ALL_METHODS)
def test_pipeline_and_inspector_agree_for_every_method(method):
    cfg = _cfg(onset_method=method)
    for seed in range(8):
        sig = make_trial(seed)
        assert _detect_onset_dispatch(sig, FS, cfg, 15.0, 35.0) == \
            inspector_style_call(sig, cfg)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_every_registered_method_is_actually_reachable(method):
    """
    A method that silently falls through to peak-fraction would still return a
    number. Comparing each method against peak-fraction on a trace where they
    genuinely differ catches a dead branch; identical output on ALL traces
    would mean the branch is not wired.
    """
    if method == "peak_fraction":
        pytest.skip("comparison target")
    pf = _cfg(onset_method="peak_fraction")
    me = _cfg(onset_method=method)
    sigs = [make_trial(s, noise=0.03) for s in range(15)]
    pf_out = [_detect_onset_dispatch(s, FS, pf, 15.0, 35.0) for s in sigs]
    me_out = [_detect_onset_dispatch(s, FS, me, 15.0, 35.0) for s in sigs]
    assert pf_out != me_out, (
        f"'{method}' produced peak-fraction's exact output on 15 traces, "
        f"which means its dispatch branch is not being reached."
    )


# ── The specific settings the inspector used to ignore ────────────────────────

def test_amplitude_gate_is_honoured_by_both_paths():
    """The gate must be able to suppress detection, identically on both paths."""
    # A clearly detectable response, so that a None result can only be the
    # gate. At amp=0.2 the derivative detector declines for its own reasons and
    # the test would pass for the wrong reason.
    sig = make_trial(0, amp=1.0, noise=0.01)
    lenient = _cfg(onset_method="bigoni", min_peak_amplitude=0.05)
    strict = _cfg(onset_method="bigoni", min_peak_amplitude=5.0)
    assert _detect_onset_dispatch(sig, FS, lenient, 15.0, 35.0) is not None
    assert _detect_onset_dispatch(sig, FS, strict, 15.0, 35.0) is None
    assert inspector_style_call(sig, strict) is None
    assert inspector_style_call(sig, lenient) == \
        _detect_onset_dispatch(sig, FS, lenient, 15.0, 35.0)


@pytest.mark.parametrize("amp", [0.05, 0.5, 2.0, 5.0])
def test_amplitude_gate_parity_across_values(amp):
    cfg = _cfg(onset_method="rms_envelope", min_peak_amplitude=amp,
               onset_env_n_boot=150)
    for seed in range(5):
        sig = make_trial(seed, amp=0.6)
        assert _detect_onset_dispatch(sig, FS, cfg, 15.0, 35.0) == \
            inspector_style_call(sig, cfg)


def test_slope_threshold_reaches_the_peak_fraction_detector():
    """
    PipelineConfig defaults slope_threshold to 0.08; the detector defaults it
    to 0.05. If the parameter is dropped anywhere in the chain the two values
    produce identical results, which is what the inspector used to do.
    """
    sigs = [make_trial(s, noise=0.03) for s in range(15)]
    a = _cfg(onset_method="peak_fraction", slope_threshold=0.005)
    b = _cfg(onset_method="peak_fraction", slope_threshold=5.0)
    out_a = [_detect_onset_dispatch(s, FS, a, 15.0, 35.0) for s in sigs]
    out_b = [_detect_onset_dispatch(s, FS, b, 15.0, 35.0) for s in sigs]
    assert out_a != out_b, "slope_threshold is not reaching the detector"
    assert out_a == [inspector_style_call(s, a) for s in sigs]
    assert out_b == [inspector_style_call(s, b) for s in sigs]


def test_peak_fraction_parameter_reaches_the_detector():
    sigs = [make_trial(s, noise=0.02) for s in range(12)]
    a = _cfg(onset_method="peak_fraction", peak_fraction=0.05)
    b = _cfg(onset_method="peak_fraction", peak_fraction=0.80)
    out_a = [_detect_onset_dispatch(s, FS, a, 15.0, 35.0) for s in sigs]
    out_b = [_detect_onset_dispatch(s, FS, b, 15.0, 35.0) for s in sigs]
    assert out_a != out_b
    assert out_a == [inspector_style_call(s, a) for s in sigs]


@pytest.mark.parametrize("field,lo,hi,method", [
    ("onset_bigoni_smooth_ms", 0.4, 4.0, "bigoni"),
    ("onset_bigoni_walkback_sd", 0.5, 6.0, "bigoni_walkback"),
    ("onset_env_window_ms", 2.0, 20.0, "rms_envelope"),
    ("onset_cusum_h", 5.0, 500.0, "cusum"),
])
def test_method_parameters_reach_their_detectors(field, lo, hi, method):
    sigs = [make_trial(s, noise=0.04) for s in range(12)]
    a = _cfg(onset_method=method, **{field: lo})
    b = _cfg(onset_method=method, **{field: hi})
    out_a = [_detect_onset_dispatch(s, FS, a, 15.0, 35.0) for s in sigs]
    out_b = [_detect_onset_dispatch(s, FS, b, 15.0, 35.0) for s in sigs]
    assert out_a != out_b, f"{field} is not reaching the {method} detector"
    assert out_a == [inspector_style_call(s, a) for s in sigs]


# ── dispatch_onset behaviour ──────────────────────────────────────────────────

def test_unknown_method_falls_back_to_peak_fraction():
    sig = make_trial(0)
    kw = dict(pre_ms=PRE_MS, search_start_ms=5, search_end_ms=60,
              min_latency_ms=15.0, max_latency_ms=35.0)
    assert dispatch_onset(sig, FS, {"onset_method": "no_such_method"}, **kw) \
        == dispatch_onset(sig, FS, {"onset_method": "peak_fraction"}, **kw)


def test_dispatch_uses_defaults_when_params_is_none():
    sig = make_trial(0)
    kw = dict(pre_ms=PRE_MS, search_start_ms=5, search_end_ms=60,
              min_latency_ms=15.0, max_latency_ms=35.0)
    assert dispatch_onset(sig, FS, None, **kw) == \
        dispatch_onset(sig, FS, DETECTION_DEFAULTS, **kw)


def test_explicit_method_argument_overrides_params():
    sigs = [make_trial(s, noise=0.03) for s in range(12)]
    kw = dict(pre_ms=PRE_MS, search_start_ms=5, search_end_ms=60,
              min_latency_ms=15.0, max_latency_ms=35.0)
    p = {"onset_method": "peak_fraction"}
    forced = [dispatch_onset(s, FS, p, method="bigoni", **kw) for s in sigs]
    plain = [dispatch_onset(s, FS, p, **kw) for s in sigs]
    assert forced != plain


def test_dispatch_ignores_none_valued_params():
    """A caller passing None for an unset field must not blank a default."""
    sig = make_trial(0)
    kw = dict(pre_ms=PRE_MS, search_start_ms=5, search_end_ms=60,
              min_latency_ms=15.0, max_latency_ms=35.0)
    assert dispatch_onset(sig, FS, {"onset_bigoni_smooth_ms": None,
                                    "onset_method": "bigoni"}, **kw) == \
        dispatch_onset(sig, FS, {"onset_method": "bigoni"}, **kw)


def test_dispatch_is_deterministic():
    sig = make_trial(3)
    kw = dict(pre_ms=PRE_MS, search_start_ms=5, search_end_ms=60,
              min_latency_ms=15.0, max_latency_ms=35.0)
    for method in ALL_METHODS:
        p = {"onset_method": method, "onset_env_n_boot": 150}
        assert dispatch_onset(sig, FS, p, **kw) == dispatch_onset(sig, FS, p, **kw)


# ── The inspector no longer carries its own copy ──────────────────────────────

def _inspector_source():
    """
    Read inspector.py from disk rather than importing it.

    Importing pulls in matplotlib's TkAgg backend, which fails on CI runners
    without a working tkinter even though conftest stubs the tkinter modules
    themselves. Static analysis is also the stronger check here: the point is
    to assert what the file does NOT contain.
    """
    import pathlib as _pl
    root = _pl.Path(__file__).resolve().parent.parent
    return (root / "mep_cmap" / "inspector.py").read_text(encoding="utf-8")


def test_inspector_module_has_no_private_dispatch_chain():
    """
    Structural guard. If someone reintroduces a local branch chain in the
    inspector, the parity tests above would still pass for whichever methods
    that chain happened to implement, so the drift has to be caught by absence.
    """
    src = _inspector_source()
    assert 'self.onset_method == "bootstrap"' not in src
    assert 'self.onset_method == "bigoni"' not in src
    assert 'self.onset_method == "bigoni_walkback"' not in src
    assert "dispatch_onset(" in src


def test_inspector_accepts_a_detection_params_dict():
    import ast

    tree = ast.parse(_inspector_source())
    init = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DataInspectorWindow":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
                    init = sub
    assert init is not None, "DataInspectorWindow.__init__ not found"
    names = [a.arg for a in init.args.args + init.args.kwonlyargs]
    assert "detection_params" in names


# ── The condition template the derivative-ratio method needs ─────────────────

MEP_LO, MEP_HI = 15.0, 35.0


def _mep_trial(seed, onset_ms=22.0, dur_ms=18.0, noise=0.010, harmonic=0.5):
    """A trial the derivative-ratio detector is specified for.

    This module's own make_trial uses a smooth single-cycle burst in 0.02 mV
    noise, which suits threshold-based detectors but sits below the
    derivative-ratio method's working range: its gates are stated in absolute
    derivatives, so they need the response to be spectrally richer than the
    baseline, and on that fixture it detects nothing at all. Testing it there
    would measure the fixture, not the parity these tests are about.
    """
    rng = np.random.default_rng(seed)
    n = int((PRE_MS + POST_MS) * FS / 1000)
    stim = int(PRE_MS * FS / 1000)
    x = rng.normal(0.0, noise, n)
    i0 = stim + int(onset_ms * FS / 1000)
    i1 = stim + int((onset_ms + dur_ms) * FS / 1000)
    k = np.arange(i1 - i0) / float(i1 - i0)
    shape = np.sin(2 * np.pi * k) + harmonic * np.sin(8 * np.pi * k)
    x[i0:i1] += shape * np.sin(np.pi * k) ** 0.5
    return x


def _mep_trials(n=12, **kw):
    return [_mep_trial(s, **kw) for s in range(n)]


def _template(segs):
    return np.median(np.vstack(segs), axis=0)


def test_boyles_agrees_between_the_two_paths_when_both_have_a_template():
    """
    The derivative-ratio detector is the only one that consults a condition
    average. The analysis builds one; the Data Inspector must too, or the same
    trial is judged against different landmarks in analysis and in review.
    """
    cfg = _cfg(onset_method="boyles")
    segs = _mep_trials()
    tpl = _template(segs)
    detected = 0
    for sig in segs:
        pipeline = _detect_onset_dispatch(sig, FS, cfg, MEP_LO, MEP_HI,
                                          template=tpl)
        inspector = dispatch_onset(
            sig, FS, detector_params(cfg),
            pre_ms=cfg.pre_ms, search_start_ms=cfg.ptp_start,
            search_end_ms=cfg.ptp_end, min_latency_ms=MEP_LO,
            max_latency_ms=MEP_HI, template=tpl)
        assert pipeline == inspector
        detected += pipeline is not None
    assert detected >= 8, "the fixture is not producing detectable responses"


def test_a_missing_template_makes_the_detector_more_permissive_not_wrong():
    """
    Without a template the peak-jitter gate is skipped. That does not move an
    onset -- the template feeds no part of locating one -- but it removes a
    rejection criterion, so the Inspector could accept a trial the analysis
    rejected. That asymmetry is why the Inspector supplies one.
    """
    cfg = _cfg(onset_method="boyles")
    segs = _mep_trials()
    tpl = _template(segs)
    odd = _mep_trial(99, onset_ms=31.0)               # peak far from the condition's

    strict = dict(detector_params(cfg))
    strict["onset_boyles_peak_jitter_ms"] = 2.0
    kw = dict(pre_ms=cfg.pre_ms, search_start_ms=cfg.ptp_start,
              search_end_ms=cfg.ptp_end, min_latency_ms=MEP_LO,
              max_latency_ms=45.0)
    with_tpl = dispatch_onset(odd, FS, strict, template=tpl, **kw)
    without = dispatch_onset(odd, FS, strict, template=None, **kw)
    assert with_tpl is None
    assert without is not None

    # Where both accept, the latency is the same: the gate rejects, it does not
    # relocate.
    for sig in segs:
        a = dispatch_onset(sig, FS, detector_params(cfg), template=tpl, **kw)
        b = dispatch_onset(sig, FS, detector_params(cfg), template=None, **kw)
        if a is not None and b is not None:
            assert a == b


def test_the_template_landmark_is_stable_to_which_trials_are_retained():
    """
    The analysis screens trials by outlier detection; the Inspector knows
    manual exclusions. Those sets are not identical, so the two templates can
    differ. Measured on real recordings, dropping trials left the median's
    first-peak landmark unmoved to two decimal places, against a gate tolerance
    of 15 ms -- so the difference cannot change a decision. Asserted here so
    the claim is checked rather than assumed.
    """
    cfg = _cfg(onset_method="boyles")
    segs = _mep_trials(n=16)
    full = _template(segs)
    dropped = _template([s for i, s in enumerate(segs) if i not in (0, 1, 2)])

    stim = int(PRE_MS * FS / 1000)
    lo = stim + int(5 * FS / 1000)
    hi = stim + int(45 * FS / 1000)

    def anchor(t):
        w = t[lo:hi]
        return (lo + min(int(np.argmax(w)), int(np.argmin(w))) - stim) * 1000.0 / FS

    assert abs(anchor(full) - anchor(dropped)) < 1.0

    kw = dict(pre_ms=cfg.pre_ms, search_start_ms=cfg.ptp_start,
              search_end_ms=cfg.ptp_end, min_latency_ms=MEP_LO,
              max_latency_ms=MEP_HI)
    a = [dispatch_onset(s, FS, detector_params(cfg), template=full, **kw) for s in segs]
    b = [dispatch_onset(s, FS, detector_params(cfg), template=dropped, **kw) for s in segs]
    assert a == b


# ── The Inspector actually supplies one ──────────────────────────────────────

def _inspector_source():
    import pathlib as _pl
    root = _pl.Path(__file__).resolve().parent.parent
    return (root / "mep_cmap" / "inspector.py").read_text(encoding="utf-8")


def test_inspector_passes_a_template_to_the_dispatch():
    src = _inspector_source()
    assert "template=self._condition_template()" in src, (
        "the Inspector must supply a condition average, or the derivative-ratio "
        "method silently loses its peak-jitter gate during review"
    )


def test_inspector_template_excludes_excluded_trials():
    src = _inspector_source()
    a = src.index("def _condition_template")
    b = src.index("\n    def ", a + 10)
    body = src[a:b]
    assert '"exclude"' in body or "'exclude'" in body, (
        "the template must be built from retained trials only"
    )
    assert "np.median" in body


def test_inspector_invalidates_its_template_when_an_exclusion_changes():
    """A cached template built from a stale trial set is worse than none."""
    src = _inspector_source()
    a = src.index("def _set_exclude")
    b = src.index("\n    def ", a + 10)
    assert "_template_cache.pop" in src[a:b], (
        "changing an exclusion must invalidate the cached template"
    )
