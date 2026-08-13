"""
Guard: detection defaults must be defined in exactly one place.

``mep_cmap/detection/defaults.py`` is the single source of truth. The same
values are consumed by ``PipelineConfig``, by ``preferences.DEFAULTS``, and by
the Tk variable initialisers in ``app.py``. Before that module existed the
values were restated at each site and had already drifted: ``PipelineConfig``
defaulted ``onset_method`` to ``"peak_fraction"`` while ``preferences``
defaulted to ``"bigoni"``. The GUI hid it, because it always passes an explicit
value, so the divergence surfaced only for code constructing ``PipelineConfig``
directly -- which silently ran a different detector than the application would
have.

These tests fail if any consumer stops agreeing with the canonical dict, so a
default changed in one place cannot ship.
"""

import dataclasses

import pytest

from mep_cmap.detection import (
    ONSET_METHOD_HINTS,
    ONSET_METHOD_LABELS,
    _METHOD_REGISTRY,
)
from mep_cmap.detection.defaults import (
    DEFAULT_CONSENSUS_METHODS,
    DEFAULT_ONSET_METHOD,
    DETECTION_DEFAULTS,
    PREF_KEY_ALIASES,
    config_key_for,
    detector_params,
    pref_key_for,
)
from mep_cmap.pipeline import PipelineConfig
from mep_cmap.preferences import DEFAULTS as PREF_DEFAULTS


def _config_defaults():
    """Default value of every PipelineConfig field, factories resolved."""
    out = {}
    for f in dataclasses.fields(PipelineConfig):
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            out[f.name] = f.default_factory()              # type: ignore[misc]
    return out


# ── The canonical dict and its consumers agree ────────────────────────────────

@pytest.mark.parametrize("key", sorted(DETECTION_DEFAULTS))
def test_pipeline_config_matches_canonical_default(key):
    cfg_defaults = _config_defaults()
    assert key in cfg_defaults, (
        f"PipelineConfig has no field '{key}'. Every detection default must be "
        f"reachable from the config, or detector_params() will silently "
        f"substitute the canonical value for something the user set."
    )
    assert cfg_defaults[key] == DETECTION_DEFAULTS[key], (
        f"PipelineConfig.{key} = {cfg_defaults[key]!r} but "
        f"detection.defaults says {DETECTION_DEFAULTS[key]!r}. Change it in "
        f"detection/defaults.py only."
    )


@pytest.mark.parametrize("key", sorted(DETECTION_DEFAULTS))
def test_preferences_matches_canonical_default(key):
    pkey = pref_key_for(key)
    assert pkey in PREF_DEFAULTS, (
        f"preferences.DEFAULTS is missing '{pkey}'; a preference that is never "
        f"seeded reads back as None and the GUI will show a blank field."
    )
    assert PREF_DEFAULTS[pkey] == DETECTION_DEFAULTS[key]


def test_the_specific_drift_that_motivated_this_module_is_gone():
    """Regression guard for the peak_fraction/bigoni divergence."""
    assert PipelineConfig().onset_method == DEFAULT_ONSET_METHOD
    assert PREF_DEFAULTS["onset_method"] == DEFAULT_ONSET_METHOD
    assert DEFAULT_ONSET_METHOD == "bigoni"


# ── Key aliasing is bidirectional ─────────────────────────────────────────────

@pytest.mark.parametrize("cfg_key,pref_key", sorted(PREF_KEY_ALIASES.items()))
def test_alias_round_trips(cfg_key, pref_key):
    assert pref_key_for(cfg_key) == pref_key
    assert config_key_for(pref_key) == cfg_key


def test_unaliased_keys_pass_through_unchanged():
    assert pref_key_for("onset_env_window_ms") == "onset_env_window_ms"
    assert config_key_for("onset_env_window_ms") == "onset_env_window_ms"


def test_aliases_only_cover_legacy_keys():
    """
    Aliases exist solely because three preference keys predate the naming
    convention and cannot be renamed without invalidating every saved
    preferences.json. New parameters must not add to this list.
    """
    assert set(PREF_KEY_ALIASES) == {
        "peak_fraction", "min_peak_amplitude", "slope_threshold"}


# ── Method registry is internally consistent ──────────────────────────────────

def test_registry_labels_and_hints_cover_the_same_methods():
    assert set(_METHOD_REGISTRY) == set(ONSET_METHOD_LABELS) == \
        set(ONSET_METHOD_HINTS)


def test_default_method_is_registered():
    assert DEFAULT_ONSET_METHOD in _METHOD_REGISTRY


def test_every_consensus_member_is_registered():
    for m in DEFAULT_CONSENSUS_METHODS:
        assert m in _METHOD_REGISTRY


def test_consensus_member_count_is_odd():
    """An odd count makes the median a member of the set, not an average."""
    assert len(DEFAULT_CONSENSUS_METHODS) % 2 == 1


def test_new_methods_are_registered():
    for m in ("rms_envelope", "cusum", "consensus"):
        assert m in _METHOD_REGISTRY


def test_legacy_methods_survive_registration():
    """Frozen for reproducibility: they must remain selectable."""
    for m in ("peak_fraction", "bootstrap", "bigoni", "bigoni_walkback"):
        assert m in _METHOD_REGISTRY


# ── detector_params ───────────────────────────────────────────────────────────

def test_detector_params_returns_every_key_from_a_default_config():
    params = detector_params(PipelineConfig())
    assert set(params) == set(DETECTION_DEFAULTS)


def test_detector_params_reflects_user_overrides():
    cfg = PipelineConfig(onset_env_window_ms=12.5, onset_cusum_h=99.0)
    params = detector_params(cfg)
    assert params["onset_env_window_ms"] == 12.5
    assert params["onset_cusum_h"] == 99.0


def test_detector_params_accepts_a_plain_dict():
    """The GUI snapshots settings into a dict before starting the worker."""
    params = detector_params({"onset_cusum_h": 42.0})
    assert params["onset_cusum_h"] == 42.0
    assert params["onset_env_window_ms"] == \
        DETECTION_DEFAULTS["onset_env_window_ms"]


def test_detector_params_keys_match_consensus_adapter_expectations():
    """
    onset_consensus adapters look parameters up by config field name. If the
    two ever diverge the consensus members silently fall back to their own
    defaults instead of the user's settings -- a bug that would produce
    plausible numbers and no error.
    """
    from mep_cmap.detection import onset_consensus as oc
    params = detector_params(PipelineConfig(
        onset_env_window_ms=7.0, onset_cusum_h=33.0,
        onset_bigoni_smooth_ms=0.9))
    common = dict(pre_ms=100, search_start_ms=5, search_end_ms=60,
                  min_latency_ms=15.0, max_latency_ms=35.0,
                  min_peak_amplitude=0.05)
    # Exercised via the adapters' .get() defaults: a mismatched key would
    # return the fallback rather than the value set above.
    assert params.get("onset_env_window_ms") == 7.0
    assert params.get("onset_cusum_h") == 33.0
    assert params.get("onset_bigoni_smooth_ms") == 0.9
    assert set(oc._ADAPTERS) >= set(DEFAULT_CONSENSUS_METHODS)
    assert common["pre_ms"] == 100
