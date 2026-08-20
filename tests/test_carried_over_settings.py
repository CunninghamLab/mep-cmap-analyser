"""
Three fixes that each lost something the analyst had already supplied.

  * epochs set in the Conditions tab never reached tab 1a
  * a unit the file recorded was replaced with 'dimensionless'
  * one stimulator typed two ways became two stimulators, silently
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
COND = (ROOT / "mep_cmap" / "conditions_tab.py").read_text(encoding="utf-8")
SMR = (ROOT / "mep_cmap" / "formats" / "spike2_smr.py").read_text(encoding="utf-8")


def _method(name, src):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found")


def _code_only(name, src):
    """The function's body without its docstring.

    A rule stated in prose must not satisfy a test looking for its absence in
    the code: the docstring for _raw_channel_unit names 'Volt' precisely to say
    that it is NOT translated.
    """
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = list(node.body)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]
            return "\n".join(ast.unparse(n) for n in body)
    raise AssertionError(f"{name} not found")


# ── epochs reach tab 1a ──────────────────────────────────────────────────────

def test_epochs_are_written_to_every_channels_snapshot():
    """window_map is per-channel state held in _chan_settings, and
    _restore_chan_settings loads the live map FROM a snapshot. Writing the
    current channel's epochs only to the live map meant the restore a few lines
    later overwrote them with a snapshot that never received them -- so on the
    ordinary path, where the Conditions channel is also the first analysed one,
    the epochs vanished and 1a fell back to the file-wide window."""
    body = _method("_cond_apply", COND)
    # Quote style is whatever ast.unparse chooses.
    assert "window_map'] = merged" in body or 'window_map"] = merged' in body


def test_a_channel_without_a_snapshot_gets_one_first():
    """Writing epochs into a snapshot that does not exist would lose the rest
    of that channel's setup when the restore reads it back."""
    body = _method("_cond_apply", COND)
    assert "_snapshot_chan_settings" in body


def test_the_live_map_is_updated_too():
    """So the table is right even if nothing restores before it is drawn."""
    body = _method("_cond_apply", COND)
    assert "self.window_map = _live" in body


# ── the unit the file recorded ───────────────────────────────────────────────

def test_the_raw_header_is_preferred_over_the_parsed_unit():
    """quantities substitutes 'dimensionless' for anything it cannot parse, so
    a torque channel arrived already stripped of 'Nm'."""
    body = _method("extract_emg_waveform_and_fs", SMR)
    i_raw = body.index("_raw_channel_unit")
    i_fallback = body.index("units.dimensionality")
    assert i_raw < i_fallback, "the raw header must be tried first"


def test_the_parsed_unit_is_still_the_fallback():
    """A file the raw reader cannot open must not lose its unit entirely."""
    body = _method("extract_emg_waveform_and_fs", SMR)
    assert "if unit is None:" in body


def test_the_unit_is_trimmed_but_not_translated():
    """The header pads some entries (' Volt'). Mapping 'Volt' to 'V' is the
    first step back towards a units library, which is what lost 'Nm'.

    Checked against the CODE, not the docstring: the docstring names 'Volt'
    precisely in order to say that it is left alone."""
    code = _code_only("_raw_channel_unit", SMR)
    assert ".strip()" in code
    for invented in ("volt", "replace(", "rescale", "dimensionality"):
        assert invented not in code.lower()


def test_a_failure_returns_none_rather_than_raising():
    """Reading a unit label must not stop a recording being read."""
    from mep_cmap.formats.spike2_smr import _raw_channel_unit
    assert _raw_channel_unit("no-such-file.smr", 0) is None
    assert _raw_channel_unit(__file__, 0) is None


def test_an_out_of_range_channel_is_not_an_error():
    from mep_cmap.formats.spike2_smr import _raw_channel_unit
    assert _raw_channel_unit("no-such-file.smr", 999) is None


# ── one stimulator typed two ways ────────────────────────────────────────────

def test_two_spellings_of_one_device_are_reported():
    from mep_cmap.bidsify import _near_duplicate_devices
    dup = _near_duplicate_devices(
        [{"StimulatorID": "Magstim"}, {"StimulatorID": "magstim"}],
        "StimulatorID")
    assert dup and sorted(next(iter(dup.values()))) == ["Magstim", "magstim"]


def test_surrounding_space_counts_as_the_same_device():
    from mep_cmap.bidsify import _near_duplicate_devices
    assert _near_duplicate_devices(
        [{"ElementID": "D70 "}, {"ElementID": "D70"}], "ElementID")


def test_genuinely_different_devices_are_not_reported():
    """A Digitimer beside a Magstim is the case the whole feature exists for."""
    from mep_cmap.bidsify import _near_duplicate_devices
    assert _near_duplicate_devices(
        [{"StimulatorID": "Digitimer"}, {"StimulatorID": "Magstim"}],
        "StimulatorID") == {}


def test_one_device_is_not_reported():
    from mep_cmap.bidsify import _near_duplicate_devices
    assert _near_duplicate_devices([{"StimulatorID": "Magstim"}],
                                   "StimulatorID") == {}


def test_they_are_reported_not_merged():
    """The tool cannot know which spelling was meant, and quietly picking one
    would rewrite what the analyst entered."""
    from mep_cmap import bidsify
    body = _method("write_nibs_sidecar",
                   (ROOT / "mep_cmap" / "bidsify.py").read_text(encoding="utf-8"))
    assert "_near_duplicate_devices" in body
    assert "log_callback" in body


# ── units are declared, not tabulated ────────────────────────────────────────

def test_the_intensity_unit_is_not_a_column():
    """The spec states units in the sidecar so a number in the table is never
    ambiguous. Tagged for the table, it was written twice."""
    from mep_cmap.bids_schema import load_schema
    f = load_schema().field("StimulationIntensityUnits")
    assert f.block == ""
    assert f.scope == "parameter_set"      # still read per set, for the sidecar


def test_it_no_longer_appears_in_the_table():
    from mep_cmap.bids_schema import load_schema
    from mep_cmap import stim_params as sp
    schema = load_schema()
    sets = [sp.StimParamSet("A", nibs_type="TMS",
                            values={"StimulationIntensity": 60,
                                    "StimulationIntensityUnits": "%MSO"})]
    cols, _ = sp.nibs_rows(sets, schema)
    assert "StimulationIntensityUnits" not in cols
    assert "stimulus_intensity" in cols
