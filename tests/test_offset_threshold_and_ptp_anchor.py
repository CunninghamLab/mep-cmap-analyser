"""
Tests for two faults that only real recordings exposed.

Both were invisible to synthetic testing because the synthetic trials were too
clean: a burst that returns instantly to Gaussian noise, in a file containing a
single stimulus type.

1. OFFSET. The return threshold was `baseline_mean + criterion*SD`, an
   absolute level tied to a resting-quiet baseline. Real EMG does not settle
   back to that level within tens of milliseconds after a large response --
   after-activity and filter ringing keep it above. On a real resting file the
   detector found an offset on 15 of 127 trials with an onset, and the ones it
   found had a MEDIAN SNR OF 12 while the ones it missed had 38. It failed
   worse the cleaner the trial, which is the wrong way round and the signature
   of an absolute threshold rather than a tuning problem.

2. PTP WINDOW. One window for the whole file, while the latency profile is per
   stimulus type. On a real file containing both M-waves and MEPs, with the
   default 10-50 ms window, the M-wave conditions had the first 6 ms of every
   response excluded -- and an M-wave's whole biphasic deflection lasts
   5-15 ms. Amplitude, the primary outcome, was affected rather than latency.

The synthetic fixtures here deliberately include a decaying tail and a mixed
stimulus file, so both failures reproduce and stay reproduced.
"""

import numpy as np
import pytest

from mep_cmap.detection.offset_detection import detect_mep_offset
from mep_cmap.detection.quantification import compute_ptp
from mep_cmap.pipeline import (ARTEFACT_FLOOR_MS, PipelineConfig,
                               pipeline_detect_onsets,
                               ptp_window_for_stim_type)

FS = 5000.0
PRE_MS = 20
POST_MS = 400
SB = int(PRE_MS * FS / 1000)


def burst(x, onset_ms, dur_ms, amp):
    i0 = SB + int(onset_ms * FS / 1000)
    i1 = SB + int((onset_ms + dur_ms) * FS / 1000)
    k = np.arange(i1 - i0) / float(i1 - i0)
    x[i0:i1] += amp * np.sin(2 * np.pi * k) * np.sin(np.pi * k) ** 0.5
    return i1


def trial_with_tail(seed=0, amp=1.0, onset_ms=22.0, dur_ms=18.0,
                    noise=0.006, tail_frac=0.06, tau_ms=60.0):
    """A response followed by exponentially decaying after-activity.

    The tail is the part synthetic testing was missing. Without it the signal
    returns to the noise floor the instant the burst ends and any return
    criterion succeeds.
    """
    rng = np.random.default_rng(seed)
    n = int((PRE_MS + POST_MS) * FS / 1000)
    x = rng.normal(0.0, noise, n)
    i1 = burst(x, onset_ms, dur_ms, amp)
    t = np.arange(n - i1) / FS * 1000.0
    x[i1:] += rng.normal(0, 1, n - i1) * amp * tail_frac * np.exp(-t / tau_ms)
    return x


def plain_trial(seed, onset_ms, dur_ms, amp, noise=0.006):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, noise, int((PRE_MS + POST_MS) * FS / 1000))
    burst(x, onset_ms, dur_ms, amp)
    return x


# ── 1. Offset: the return threshold must scale with the response ──────────────

OFFSET_KW = dict(onset_ms=22.0, pre_ms=PRE_MS, search_end_ms=300, n_boot=200)


def _offset_rate(amp, peak_frac, max_duration_ms, n=15):
    got = [detect_mep_offset(trial_with_tail(s, amp=amp), FS,
                             peak_frac=peak_frac,
                             max_duration_ms=max_duration_ms, **OFFSET_KW)
           for s in range(n)]
    return [v for v in got if v is not None], n


@pytest.mark.parametrize("amp", [0.25, 1.0, 2.5, 8.0])
def test_offset_is_found_on_large_responses_with_a_decay_tail(amp):
    """The exact case that failed on real data: clean, large responses."""
    found, n = _offset_rate(amp, peak_frac=0.12, max_duration_ms=100.0)
    assert len(found) >= n - 1, (
        f"offset found on only {len(found)}/{n} trials at {amp} mV"
    )
    assert abs(float(np.mean(found)) - 40.0) < 4.0


@pytest.mark.parametrize("amp", [0.25, 1.0, 2.5, 8.0])
def test_an_absolute_threshold_alone_fails_on_those_same_responses(amp):
    """
    Guards the diagnosis, not just the fix. With the peak-relative floor
    disabled the detector must still fail here -- if this starts passing, the
    fixture has drifted back to being unrealistically clean and the test above
    no longer proves anything.
    """
    found, n = _offset_rate(amp, peak_frac=0.0, max_duration_ms=60.0)
    assert len(found) <= n // 3


def test_offset_success_no_longer_falls_away_as_snr_rises():
    """
    The real-data signature: found at median SNR 12, missed at median SNR 38.
    Detection rate must not be inversely related to amplitude.
    """
    rates = []
    for amp in (0.10, 0.25, 1.0, 2.5, 8.0):
        found, n = _offset_rate(amp, peak_frac=0.12, max_duration_ms=100.0)
        rates.append(len(found) / float(n))
    assert min(rates) >= 0.8, f"detection rate collapses with amplitude: {rates}"


def test_peak_fraction_floor_can_be_disabled():
    a = detect_mep_offset(trial_with_tail(0, amp=0.1), FS, peak_frac=0.0,
                          max_duration_ms=100.0, **OFFSET_KW)
    b = detect_mep_offset(trial_with_tail(0, amp=0.1), FS, peak_frac=0.12,
                          max_duration_ms=100.0, **OFFSET_KW)
    assert a is None or b is None or b <= a  # relative floor cannot be later


def test_offset_still_respects_min_and_max_duration():
    x = trial_with_tail(0, amp=1.0)
    assert detect_mep_offset(x, FS, peak_frac=0.12, max_duration_ms=2.0,
                             **OFFSET_KW) is None
    got = detect_mep_offset(x, FS, peak_frac=0.12, max_duration_ms=100.0,
                            min_duration_ms=5.0, **OFFSET_KW)
    assert got is None or (got - 22.0) >= 5.0


# ── 2. PTP window anchoring ───────────────────────────────────────────────────

M_WAVE = dict(onset_ms=4.0, dur_ms=8.0, amp=2.0, profile=(2.0, 12.0))
TMS_MEP = dict(onset_ms=20.0, dur_ms=18.0, amp=0.25, profile=(15.0, 35.0))

FILE_WIDE_START = SB + int(10 * FS / 1000)      # the shipped 10-50 ms default
FILE_WIDE_END = SB + int(50 * FS / 1000)


def _condition(spec, n=12):
    return np.vstack([plain_trial(s, spec["onset_ms"], spec["dur_ms"],
                                  spec["amp"]) for s in range(n)])


def _measure(spec, anchor):
    segs = _condition(spec)
    cfg = PipelineConfig(pre_ms=PRE_MS, post_ms=POST_MS,
                         ptp_start=10, ptp_end=50,
                         onset_method="bigoni",
                         latency_map={"X": spec["profile"]},
                         ptp_anchor=anchor)
    onsets = pipeline_detect_onsets("X", segs, set(), FILE_WIDE_START,
                                    FILE_WIDE_END, FS, cfg,
                                    log_callback=lambda *a, **k: None)
    s_i, e_i, win = ptp_window_for_stim_type(
        "X", onsets, FS, cfg, FILE_WIDE_START, FILE_WIDE_END, SB,
        log_callback=lambda *a, **k: None)
    measured = float(np.mean([compute_ptp(sg, s_i, e_i) for sg in segs]))
    truth = float(np.mean([compute_ptp(sg, SB, SB + int(60 * FS / 1000))
                           for sg in segs]))
    return measured, truth, win


def test_fixed_window_underestimates_an_m_wave_badly():
    """Guards the diagnosis: without anchoring the error must still be large."""
    measured, truth, win = _measure(M_WAVE, anchor=False)
    assert win is None
    assert measured < 0.75 * truth, (
        "the fixed-window M-wave error has disappeared; the fixture no longer "
        "reproduces the condition this feature exists for"
    )


def test_anchoring_recovers_the_m_wave_amplitude():
    measured, truth, win = _measure(M_WAVE, anchor=True)
    assert win is not None
    assert abs(measured - truth) / truth < 0.05


def test_anchoring_does_not_disturb_a_condition_the_window_already_suited():
    before, truth, _ = _measure(TMS_MEP, anchor=False)
    after, _, win = _measure(TMS_MEP, anchor=True)
    assert abs(before - truth) / truth < 0.05
    assert abs(after - before) / before < 0.02


def test_anchored_window_respects_the_configured_end_as_a_ceiling():
    _, _, win = _measure(TMS_MEP, anchor=True)
    assert win is not None
    assert win[1] <= 50.0


def test_anchored_window_respects_the_artefact_floor():
    _, _, win = _measure(M_WAVE, anchor=True)
    assert win is not None
    assert win[0] >= ARTEFACT_FLOOR_MS


def test_anchoring_is_off_by_default():
    assert PipelineConfig().ptp_anchor is False


def test_too_few_onsets_falls_back_to_the_file_wide_window():
    cfg = PipelineConfig(pre_ms=PRE_MS, post_ms=POST_MS,
                         ptp_start=10, ptp_end=50,
                         ptp_anchor=True, ptp_anchor_min_trials=8)
    s_i, e_i, win = ptp_window_for_stim_type(
        "X", {0: 20.0, 1: 20.5}, FS, cfg,
        FILE_WIDE_START, FILE_WIDE_END, SB, log_callback=lambda *a, **k: None)
    assert win is None
    assert (s_i, e_i) == (FILE_WIDE_START, FILE_WIDE_END)


def test_no_detected_onsets_falls_back_to_the_file_wide_window():
    cfg = PipelineConfig(pre_ms=PRE_MS, post_ms=POST_MS,
                         ptp_start=10, ptp_end=50, ptp_anchor=True)
    s_i, e_i, win = ptp_window_for_stim_type(
        "X", {0: None, 1: None}, FS, cfg,
        FILE_WIDE_START, FILE_WIDE_END, SB, log_callback=lambda *a, **k: None)
    assert win is None
    assert (s_i, e_i) == (FILE_WIDE_START, FILE_WIDE_END)


def test_a_late_median_onset_cannot_invert_the_window():
    """Median onset beyond the configured end must not yield end <= start."""
    cfg = PipelineConfig(pre_ms=PRE_MS, post_ms=POST_MS,
                         ptp_start=10, ptp_end=50, ptp_anchor=True)
    msgs = []
    s_i, e_i, win = ptp_window_for_stim_type(
        "X", {i: 80.0 for i in range(10)}, FS, cfg,
        FILE_WIDE_START, FILE_WIDE_END, SB,
        log_callback=lambda *a: msgs.append(" ".join(str(x) for x in a)))
    assert win is None
    assert (s_i, e_i) == (FILE_WIDE_START, FILE_WIDE_END)
    assert any("PTP anchor" in m for m in msgs)


def test_anchoring_reports_the_window_it_chose():
    """A silently different measurement window would be worse than none."""
    segs = _condition(M_WAVE)
    cfg = PipelineConfig(pre_ms=PRE_MS, post_ms=POST_MS,
                         ptp_start=10, ptp_end=50, onset_method="bigoni",
                         latency_map={"X": M_WAVE["profile"]}, ptp_anchor=True)
    onsets = pipeline_detect_onsets("X", segs, set(), FILE_WIDE_START,
                                    FILE_WIDE_END, FS, cfg,
                                    log_callback=lambda *a, **k: None)
    msgs = []
    ptp_window_for_stim_type("X", onsets, FS, cfg, FILE_WIDE_START,
                             FILE_WIDE_END, SB,
                             log_callback=lambda *a: msgs.append(
                                 " ".join(str(x) for x in a)))
    assert any("PTP anchor" in m and "->" in m for m in msgs)


# ── 3. Superseded defaults must reach existing installations ─────────────────

def test_a_stale_saved_value_is_migrated_to_the_new_default():
    """
    Raising a default reaches nobody who has pressed Apply, because the dialog
    writes every field on the tab. `mep_offset_max_duration_ms` was raised from
    60 to 100 ms after real recordings showed ~54 ms responses; on the machine
    where it mattered the stored 60 won and offset detection succeeded on 1
    trial of 81 instead of 81 of 81, with no error and no warning.
    """
    from mep_cmap.detection.defaults import (DETECTION_DEFAULTS,
                                             migrate_detection_defaults)

    data = {"mep_offset_max_duration_ms": 60.0}
    changed = migrate_detection_defaults(data)
    assert changed
    assert data["mep_offset_max_duration_ms"] == \
        DETECTION_DEFAULTS["mep_offset_max_duration_ms"]


def test_a_value_the_analyst_chose_is_never_overwritten():
    """Migration applies only where the stored value is still the old default."""
    from mep_cmap.detection.defaults import migrate_detection_defaults

    data = {"mep_offset_max_duration_ms": 75.0}
    assert migrate_detection_defaults(data) == []
    assert data["mep_offset_max_duration_ms"] == 75.0


def test_migration_is_idempotent():
    from mep_cmap.detection.defaults import migrate_detection_defaults

    data = {"mep_offset_max_duration_ms": 60.0}
    migrate_detection_defaults(data)
    first = data["mep_offset_max_duration_ms"]
    assert migrate_detection_defaults(data) == []
    assert data["mep_offset_max_duration_ms"] == first


def test_already_current_version_is_left_untouched():
    from mep_cmap.detection.defaults import (DETECTION_DEFAULTS_VERSION,
                                             migrate_detection_defaults)

    data = {"mep_offset_max_duration_ms": 60.0,
            "detection_defaults_version": DETECTION_DEFAULTS_VERSION}
    assert migrate_detection_defaults(data) == []
    assert data["mep_offset_max_duration_ms"] == 60.0


def test_superseded_table_only_names_real_settings():
    from mep_cmap.detection.defaults import (DETECTION_DEFAULTS,
                                             SUPERSEDED_DEFAULTS)

    for version, table in SUPERSEDED_DEFAULTS.items():
        for key in table:
            assert key in DETECTION_DEFAULTS, (
                f"SUPERSEDED_DEFAULTS[{version}] names '{key}', which is not a "
                f"detection setting"
            )


def test_reset_clears_every_stored_detection_setting():
    from mep_cmap.detection.defaults import (DETECTION_DEFAULTS,
                                             reset_detection_defaults)

    from mep_cmap.detection.defaults import pref_key_for

    data = {pref_key_for(k): "junk" for k in DETECTION_DEFAULTS}
    data["font_scale"] = 1.25
    removed = reset_detection_defaults(data)
    assert data["font_scale"] == 1.25, "reset must not touch unrelated settings"
    assert removed, "nothing was removed"
    for k in DETECTION_DEFAULTS:
        assert pref_key_for(k) not in data, (
            f"'{k}' survived the reset and would keep shadowing its default"
        )


# ── 4. Migration must work through the REAL load path ────────────────────────

def _prefs_with_stored(tmp_path, stored):
    """Construct a Preferences object against a temporary preferences file.

    Exercises Preferences.load() rather than calling the migration helper with
    a hand-made dict. The first version of this migration was correct as a
    helper and broken as a caller: `_data` is seeded from DEFAULTS, which
    already carries the current version, so reading the version out of `_data`
    made every install look up to date. Helper-level tests passed and the
    upgrade silently did nothing.
    """
    import json

    import mep_cmap.preferences as P

    pf = tmp_path / "preferences.json"
    pf.write_text(json.dumps(stored), encoding="utf-8")
    old_file, old_dir = P.PREFS_FILE, P.PREFS_DIR
    P.PREFS_FILE, P.PREFS_DIR = pf, tmp_path
    try:
        return type(P.prefs)(), pf
    finally:
        P.PREFS_FILE, P.PREFS_DIR = old_file, old_dir


def test_stale_value_is_migrated_when_preferences_are_actually_loaded(tmp_path):
    from mep_cmap.detection.defaults import DETECTION_DEFAULTS

    p, _ = _prefs_with_stored(tmp_path, {"mep_offset_max_duration_ms": 60.0})
    assert p.mep_offset_max_duration_ms == \
        DETECTION_DEFAULTS["mep_offset_max_duration_ms"]
    assert p.migration_notes


def test_a_deliberate_value_survives_a_real_load(tmp_path):
    p, _ = _prefs_with_stored(tmp_path, {"mep_offset_max_duration_ms": 75.0})
    assert p.mep_offset_max_duration_ms == 75.0
    assert p.migration_notes == []


def test_migration_stamps_the_file_so_it_runs_once(tmp_path):
    import json

    from mep_cmap.detection.defaults import DETECTION_DEFAULTS_VERSION

    p, pf = _prefs_with_stored(tmp_path, {"mep_offset_max_duration_ms": 60.0})
    assert p.migration_notes
    written = json.loads(pf.read_text(encoding="utf-8"))
    assert written["detection_defaults_version"] == DETECTION_DEFAULTS_VERSION

    import mep_cmap.preferences as P
    old_file, old_dir = P.PREFS_FILE, P.PREFS_DIR
    P.PREFS_FILE, P.PREFS_DIR = pf, tmp_path
    try:
        second = type(P.prefs)()
    finally:
        P.PREFS_FILE, P.PREFS_DIR = old_file, old_dir
    assert second.migration_notes == []


def test_an_unstamped_file_is_treated_as_version_one(tmp_path):
    """A file with no version key predates the stamp and must be migrated."""
    p, _ = _prefs_with_stored(tmp_path, {"mep_offset_max_duration_ms": 60.0,
                                         "onset_method": "consensus"})
    assert p.migration_notes
    assert p.onset_method == "consensus"      # unrelated settings preserved


def test_a_missing_preferences_file_does_not_break_startup(tmp_path):
    p, _ = _prefs_with_stored(tmp_path, {})
    assert p.migration_notes == []
    assert p.mep_offset_max_duration_ms > 0
