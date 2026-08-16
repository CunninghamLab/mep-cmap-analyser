"""
Tests for measuring and applying the marker-to-stimulus delay.

The marker in a recording is not always the instant the stimulus fired. When it
is late the response begins before the epoch's t=0, part of it lands in the
pre-stimulus window, and every baseline-relative measure fails in a way that
looks like a separate detector fault. On one real recording whose markers were
2 ms late, the derivative-ratio detector returned 0 onsets from 15 trials,
peak-to-peak was read from a shoulder rather than the peak, and the offset
landed mid-excursion -- three symptoms, one cause.
"""

import numpy as np
import pytest

from mep_cmap.event_delay import (DEFAULT_NEGLIGIBLE_MS,
                                  DEFAULT_SPREAD_LIMIT_MS,
                                  measure_event_delay, scan_event_delays)

FS = 5000.0


def synth(n_trials=12, isi=1.0, delay_ms=0.0, artefact=2.0, response=6.0,
          jitter_ms=0.0, noise=0.01, seed=0, with_artefact=True):
    """A recording whose response is far LARGER than its artefact.

    That combination is the point: peak amplitude finds the response and gets
    the wrong answer, peak slope finds the artefact and gets the right one. On
    real data (sub-001 condition A) amplitude returned 6.90 ms -- the M-wave
    peak -- where the artefact was at 0.20 ms.
    """
    rng = np.random.default_rng(seed)
    n = int((n_trials + 1) * isi * FS)
    x = rng.normal(0, noise, n)
    times = []
    for k in range(n_trials):
        t = (k + 0.5) * isi
        times.append(t)
        i = int(t * FS)
        shift = int(round((delay_ms + (rng.normal(0, jitter_ms)
                                       if jitter_ms else 0.0)) * FS / 1000.0))
        if with_artefact:
            # One sample: abrupt, which is what makes it the steepest feature
            # even though the response dwarfs it.
            x[i + shift] += artefact
        j0 = i + shift + int(4 * FS / 1000)
        j1 = i + shift + int(20 * FS / 1000)
        k_ = np.arange(j1 - j0) / float(j1 - j0)
        x[j0:j1] += response * np.sin(2 * np.pi * k_) * np.sin(np.pi * k_) ** 0.5
    return x, times


# ── Detection ────────────────────────────────────────────────────────────────

def test_a_response_larger_than_the_artefact_does_not_fool_it():
    """Peak amplitude would return the response; peak slope must not."""
    x, t = synth(delay_ms=-2.0, artefact=2.0, response=8.0)
    r = measure_event_delay(x, FS, t, "C")
    assert r.proposed
    assert abs(r.delay_ms - (-2.0)) < 0.3
    # A peak-amplitude search on the same data lands on the response instead.
    half = int(20 * FS / 1000)
    i = int(t[0] * FS)
    by_amp = (int(np.argmax(np.abs(x[i - half:i + half]))) - half) * 1000.0 / FS
    assert abs(by_amp - (-2.0)) > 1.0, "fixture does not exercise the distinction"


@pytest.mark.parametrize("delay", [-10.0, -2.0, -1.6, 1.5, 4.0])
def test_delays_of_various_sizes_are_recovered(delay):
    x, t = synth(delay_ms=delay)
    r = measure_event_delay(x, FS, t, "A")
    assert r.proposed
    assert abs(r.delay_ms - delay) < 0.3


def test_no_delay_is_reported_as_none_needed():
    x, t = synth(delay_ms=0.0)
    r = measure_event_delay(x, FS, t, "A")
    assert not r.proposed
    assert r.delay_ms == 0.0
    assert "no delay needed" in r.reason


def test_a_negligible_offset_is_not_dignified_as_a_finding():
    x, t = synth(delay_ms=0.2)          # inside a sample interval
    r = measure_event_delay(x, FS, t, "A")
    assert not r.proposed
    assert abs(r.median_ms) < DEFAULT_NEGLIGIBLE_MS


def test_a_jittering_marker_is_declined_not_averaged():
    """
    The spread is the confidence signal. Where the artefact time varies across
    trials, a single delay is the wrong model -- averaging it would apply a
    correction that is wrong on every trial rather than right on average. Real
    data: one condition measured SD 3.93 ms while genuine fixed delays measured
    0.14-0.20 ms.
    """
    x, t = synth(delay_ms=-2.0, jitter_ms=4.0)
    r = measure_event_delay(x, FS, t, "G")
    assert not r.proposed
    assert r.sd_ms > DEFAULT_SPREAD_LIMIT_MS
    assert "vary" in r.reason or "inconsistent" in r.reason
    # The measurement is still reported, because a wide spread is itself worth
    # telling the analyst about.
    assert np.isfinite(r.median_ms)


def test_no_artefact_at_all_is_declined():
    """
    A shielded rig, or one where the artefact has been removed.

    This is the dangerous case rather than merely the unhelpful one. With no
    artefact the steepest feature is the response's own rising edge, and
    because that edge is highly consistent across trials the spread test passes
    -- the scan would confidently propose moving t=0 onto the response. The
    width check exists for this: measured on real recordings the artefact is
    0.4-0.6 ms wide at half its maximum, a response edge 4.6 ms.
    """
    x, t = synth(delay_ms=-2.0, with_artefact=False, noise=0.05)
    r = measure_event_delay(x, FS, t, "A")
    assert not r.proposed
    assert "too broad" in r.reason


def test_the_width_check_runs_before_the_spread_check():
    """
    Order matters. A response edge is consistent, so it passes the spread test;
    only the width check catches it. If the spread test came first this case
    would be proposed with confidence.
    """
    x, t = synth(delay_ms=-2.0, with_artefact=False, noise=0.02)
    r = measure_event_delay(x, FS, t, "A")
    assert not r.proposed
    assert r.sd_ms < DEFAULT_SPREAD_LIMIT_MS, (
        "fixture no longer exercises the ordering: the spread test would have "
        "caught this anyway"
    )
    assert "too broad" in r.reason


def test_too_few_trials_declines():
    x, t = synth(n_trials=2, delay_ms=-2.0)
    r = measure_event_delay(x, FS, t, "F")
    assert not r.proposed
    assert "trial" in r.reason


def test_the_proposed_value_is_a_whole_number_of_samples():
    """A delay finer than the sample interval cannot be applied."""
    x, t = synth(delay_ms=-2.0)
    r = measure_event_delay(x, FS, t, "C")
    assert abs(r.delay_ms * FS / 1000.0 - round(r.delay_ms * FS / 1000.0)) < 1e-9


def test_scan_covers_every_type_including_those_that_decline():
    x1, t1 = synth(delay_ms=-2.0, seed=1)
    x2, t2 = synth(delay_ms=0.0, seed=2)
    x = np.concatenate([x1, x2])
    t2 = [t + len(x1) / FS for t in t2]
    res = scan_event_delays(x, FS, {"C": t1, "A": t2})
    assert set(res) == {"A", "C"}
    assert res["C"].proposed
    assert not res["A"].proposed          # present, with a reason


# ── Application ──────────────────────────────────────────────────────────────

def test_applying_the_delay_moves_the_response_by_exactly_that_much():
    """
    Assert the SHIFT rather than two absolute latencies.

    The absolute value depends on the fixture's internal geometry, so an
    expectation written from it tests my arithmetic about the fixture. The
    shift is what the feature promises: correcting a marker that is 2 ms early
    must move everything measured from t=0 by 2 ms, no more and no less.
    """
    from mep_cmap.pipeline import PipelineConfig, pipeline_extract_segments

    x, t = synth(delay_ms=-2.0, n_trials=10)
    time = np.arange(len(x)) / FS
    stim = int(20 * FS / 1000)

    def response_ms(delay):
        cfg = PipelineConfig(pre_ms=20, post_ms=100, prestim_ms=100,
                             delay_ms_map={"C": delay})
        segs = pipeline_extract_segments(time, x, {"C": t}, ["C"], FS, cfg)
        seg = segs["C"][0][0]
        # First sample after t=0 that clearly belongs to the response rather
        # than the one-sample artefact.
        tail = np.abs(seg[stim + int(1 * FS / 1000):])
        return (int(np.argmax(tail > 1.0)) + int(1 * FS / 1000)) * 1000.0 / FS

    uncorrected = response_ms(0.0)
    corrected = response_ms(-2.0)
    assert abs((corrected - uncorrected) - 2.0) < 0.3, (
        f"a -2.0 ms correction moved the response by "
        f"{corrected - uncorrected:.2f} ms"
    )


def test_a_delay_that_pushes_a_trial_off_the_record_is_skipped():
    from mep_cmap.pipeline import PipelineConfig, pipeline_extract_segments

    x, t = synth(n_trials=6)
    time = np.arange(len(x)) / FS
    cfg = PipelineConfig(pre_ms=20, post_ms=100, prestim_ms=100,
                         delay_ms_map={"C": -1e6})
    segs = pipeline_extract_segments(time, x, {"C": t}, ["C"], FS, cfg)
    assert not segs.get("C")           # skipped, not crashed


def test_the_sidecar_records_the_delay_and_its_source():
    from mep_cmap.bids import StudyMetadata

    meta = StudyMetadata()
    d = meta.to_sidecar("x.smr", {},
                        event_delay_ms={"C": -2.0},
                        event_delay_source={"C": "detected"})
    assert d["event_delay_ms"] == {"C": -2.0}
    assert d["event_delay_source"] == {"C": "detected"}


def test_the_sidecar_fields_exist_even_when_no_delay_is_set():
    """Absence must mean 'unsupported version', not 'no delay applied'."""
    from mep_cmap.bids import StudyMetadata

    d = StudyMetadata().to_sidecar("x.smr", {})
    assert d["event_delay_ms"] == {}
    assert d["event_delay_source"] == {}


# ── The 1a table must stay aligned ───────────────────────────────────────────

def test_the_header_list_matches_the_grid_columns():
    """
    Inserting a column means shifting every column index after it AND adding
    the header. Doing only the first left every label after Gap pointing at the
    wrong field: the Delay entry appeared under "Detect CSP", Min lat under
    "Muscle group", and so on. The table read as if the widgets had moved.

    Nothing raised, nothing looked broken in the source, and the suite passed.
    """
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "app.py").read_text(encoding="utf-8")
    a = src.index('headers = ["Stim"')
    names = re.findall(r'"([^"]+)"', src[a:src.index("]", a) + 1])
    body = src[a:a + 17000]
    cols = sorted({int(m.group(1))
                   for m in re.finditer(r"\.grid\(row=r, column=(\d+)", body)})

    assert cols == list(range(len(names))), (
        f"{len(names)} headers but grid columns {cols}: the 1a table is "
        f"misaligned"
    )


def test_delay_sits_next_to_gap():
    """They are related and neither is interpretable without the other."""
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "app.py").read_text(encoding="utf-8")
    a = src.index('headers = ["Stim"')
    names = re.findall(r'"([^"]+)"', src[a:src.index("]", a) + 1])
    assert names.index("Delay (ms)") == names.index("Gap (ms)") + 1


# ── The scan must report visibly, including when it finds nothing ────────────

def _app_src():
    import pathlib
    return (pathlib.Path(__file__).resolve().parent.parent
            / "mep_cmap" / "app.py").read_text(encoding="utf-8")


def test_the_scan_reports_beside_the_button_not_only_in_the_log():
    """
    A scan that correctly finds nothing changes nothing on tab 1a, and is then
    indistinguishable from a button that does not work. The log carries the
    detail but lives on tab 1c, so the outcome has to appear here too.

    Observed on a real file whose 162 Trigger markers were accurate to 0.2 ms:
    the scan ran, correctly proposed nothing, and looked broken.
    """
    src = _app_src()
    assert "_delay_scan_status" in src
    a = src.index('text="🔎 Detect delays"')
    b = src.index("Confirm when you have finished", a)
    assert "textvariable=self._delay_scan_status" in src[a:b], (
        "the status must be displayed next to the button"
    )


def test_every_exit_path_sets_the_status():
    """
    A stale message from a previous scan would be read as the current result --
    worse than no message, because it is confidently wrong.
    """
    src = _app_src()
    a = src.index("def _detect_event_delays")
    b = src.index("\n    def ", a + 10)
    body = src[a:b]
    n_returns = body.count("return")
    n_sets = body.count("_delay_scan_status.set(")
    assert n_sets >= n_returns, (
        f"{n_returns} exit path(s) but only {n_sets} status update(s)"
    )


def test_the_no_delay_message_distinguishes_aligned_from_undecidable():
    """
    "Markers already line up" and "could not decide" are different outcomes and
    should not share a message: the first means the file needs nothing, the
    second means the analyst should look at the log.
    """
    src = _app_src()
    a = src.index("def _detect_event_delays")
    b = src.index("\n    def ", a + 10)
    body = src[a:b]
    assert "already line up" in body
    assert "see the log" in body


def test_the_scan_reports_its_outcome_on_the_setup_tab():
    """
    A scan that correctly proposes nothing changes nothing on tab 1a, which is
    indistinguishable from a button that does not work. The log carries the
    detail but lives on tab 1c.

    Every exit path must set the status, or a message left over from a previous
    scan would be read as the current result.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "app.py").read_text(encoding="utf-8")
    a = src.index("def _detect_event_delays")
    b = src.index("\n    def ", a + 10)
    body = src[a:b]

    returns = body.count("return")
    sets = body.count("_delay_scan_status.set(")
    assert sets >= returns, (
        f"{returns} exit path(s) but only {sets} set the status; one would "
        f"leave a stale message on screen"
    )
    assert "_delay_scan_status" in src[:a], "the label must exist before use"


# ── Every epoch extraction must apply the same delay ─────────────────────────

def test_every_stim_time_to_index_conversion_applies_the_delay():
    """
    The delay was applied when epoching for analysis but not where the
    Inspector's segments and the condition averages were re-extracted from the
    raw time axis. The Inspector then DISPLAYED a different epoch from the one
    being measured, and marker indices returned from it were offset by exactly
    the delay.

    On a real recording that made the corrected condition -- and only that one
    -- come back with its peak-to-peak read from the wrong samples: 3.21 mV
    where every window of the actual trace gives 9.92. Every uncorrected
    condition was fine, which is what made it look like a detector fault
    specific to that stimulus type.
    """
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "pipeline.py").read_text(encoding="utf-8")

    bad = []
    for m in re.finditer(r"np\.argmin\(np\.abs\(time - [^)]*\)\)\)", src):
        tail = src[m.end():m.end() + 40]
        if "delay" not in tail:
            bad.append(src[:m.start()].count("\n") + 1)

    assert not bad, (
        "these convert a stimulus time to a sample index without applying the "
        f"event delay, so they build a differently aligned epoch: lines {bad}"
    )


def test_out_of_range_indices_are_skipped_at_every_site():
    """A large correction can push a trial off the record; none may crash."""
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "pipeline.py").read_text(encoding="utf-8")
    for m in re.finditer(r"np\.argmin\(np\.abs\(time - [^)]*\)\)\) \+ \w*delay\w*",
                         src):
        window = src[m.end():m.end() + 160]
        assert "< 0" in window, (
            f"no bounds check after the shift at line "
            f"{src[:m.start()].count(chr(10)) + 1}"
        )
