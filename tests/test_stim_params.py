"""
Stimulation parameter sets must describe what was delivered, per stim code.

A file containing an M-wave and an MEP had one intensity box, so it could
describe neither. Every test here guards one way the parameter sets and the
recording could disagree about what was stimulated.
"""

import pytest

from mep_cmap import stim_params as sp
from mep_cmap.bids_schema import load_schema


# ── names ────────────────────────────────────────────────────────────────────

def test_a_name_cannot_carry_the_element_delimiter():
    """The spec separates simultaneous elements with `|` inside one cell, so a
    name containing it would be read back as two values."""
    assert sp.DELIMITER not in sp.sanitise_name("MEP|120")
    assert sp.sanitise_name("MEP|120") == "MEP_120"


def test_a_name_cannot_carry_a_tab_or_newline():
    for bad in ("MEP\t120", "MEP\n120", "MEP 120"):
        out = sp.sanitise_name(bad)
        assert "\t" not in out and "\n" not in out and " " not in out


def test_duplicate_names_are_made_unique():
    """nibs_event_id MUST be unique within the file: two rows sharing one id
    makes every reference to it ambiguous."""
    out = sp.deduplicate_names(["MEP", "MEP", "MEP"])
    assert len(set(out)) == 3
    assert out[0] == "MEP"


# ── the table ────────────────────────────────────────────────────────────────

def test_a_default_set_is_created_for_each_code():
    sets = sp.ensure_sets_for(["A", "C", "G"])
    assert [s.name for s in sets] == ["A", "C", "G"]


def test_rescanning_does_not_disturb_filled_in_sets():
    """A code appearing in a newly scanned file must not reset the others."""
    sets = sp.ensure_sets_for(["A"])
    sets = [sets[0].with_value("StimulationIntensity", 45)]
    sets = sp.ensure_sets_for(["A", "G"], sets)
    assert sets[0].values["StimulationIntensity"] == 45
    assert [s.name for s in sets] == ["A", "G"]


def test_duplicate_set_names_are_an_error():
    sets = [sp.StimParamSet("MEP"), sp.StimParamSet("MEP")]
    errs = sp.validate(sets)
    assert any("unique" in e for e in errs)


def test_an_unknown_stimulation_type_is_an_error():
    errs = sp.validate([sp.StimParamSet("X", nibs_type="magnets")])
    assert any("not a stimulation type" in e for e in errs)


def test_pns_is_a_valid_type():
    """M-waves, H-reflexes and CMAPs are peripheral. Before v6.3 they had to be
    misdescribed as TMS."""
    assert "PNS" in sp.NIBS_TYPES
    assert sp.validate([sp.StimParamSet("Mmax", nibs_type="PNS")]) == []


def test_a_ticked_code_with_no_set_is_reported():
    errs = sp.validate([sp.StimParamSet("A")], {"A": "A", "G": ""}, ["A", "G"])
    assert any("'G'" in e for e in errs)
    assert sp.unassigned(["A", "G"], {"A": "A"}) == ["G"]


def test_only_sets_this_file_uses_are_written():
    """A session-level set no code here references belongs to another file, and
    writing it would describe stimulation that never happened."""
    sets = [sp.StimParamSet("A"), sp.StimParamSet("G"), sp.StimParamSet("Zz")]
    used = sp.sets_in_use({"A": "A", "G": "G"}, sets)
    assert [s.name for s in used] == ["A", "G"]


# ── projection ───────────────────────────────────────────────────────────────

def test_a_mixed_file_writes_one_row_per_parameter_set():
    """The case the whole feature exists for: peripheral M-waves and a TMS MEP
    in one recording, which a single per-file intensity could not describe."""
    schema = load_schema()
    sets = [
        sp.StimParamSet("Mmax", nibs_type="PNS",
                        values={"StimulationIntensity": 45}),
        sp.StimParamSet("MEP_120", nibs_type="TMS",
                        values={"StimulationIntensity": 60}),
    ]
    cols, rows = sp.nibs_rows(sets, schema)
    assert cols[:2] == ["nibs_event_id", "nibs_type"]
    assert [r["nibs_type"] for r in rows] == ["PNS", "TMS"]
    assert [r["nibs_event_id"] for r in rows] == ["Mmax", "MEP_120"]


def test_intensity_is_written_under_its_v63_name():
    """The schema's `emits` is what renames it, so a spec rename never touches
    saved state."""
    schema = load_schema()
    sets = [sp.StimParamSet("A", values={"StimulationIntensity": 45})]
    cols, rows = sp.nibs_rows(sets, schema)
    assert "stimulus_intensity" in cols
    assert "StimulationIntensity" not in cols
    assert rows[0]["stimulus_intensity"] == 45


def test_unpopulated_columns_are_not_written():
    """A simple study must not carry forty empty columns."""
    schema = load_schema()
    cols, _ = sp.nibs_rows([sp.StimParamSet("A")], schema)
    assert cols == ["nibs_event_id", "nibs_type"]


def test_every_row_fills_every_column():
    """A short row would misalign the TSV."""
    schema = load_schema()
    sets = [sp.StimParamSet("A", values={"StimulationIntensity": 45}),
            sp.StimParamSet("G", values={"StimulationDuration": 200})]
    cols, rows = sp.nibs_rows(sets, schema)
    for r in rows:
        assert set(r) == set(cols)
    assert sp.NA in rows[0].values()


def test_legacy_fields_are_never_written():
    """Read from older saved state, but superseded by a v6.3 field."""
    schema = load_schema()
    legacy = [f.key for f in schema.fields if f.legacy]
    assert legacy, "expected the retagged schema to mark legacy fields"
    sets = [sp.StimParamSet("A", values={k: "x" for k in legacy})]
    cols, _ = sp.nibs_rows(sets, schema)
    assert cols == ["nibs_event_id", "nibs_type"]


def test_units_are_declared_in_the_sidecar():
    """The spec is explicit that units are never assumed from the numbers in
    the table: 58 could be %MSO or mA."""
    schema = load_schema()
    sets = [sp.StimParamSet("A", values={"StimulationDuration": 200})]
    side = sp.units_sidecar(sets, schema)
    for col, meta in side.items():
        assert meta.get("Units")


# ── persistence ──────────────────────────────────────────────────────────────

def test_a_set_survives_a_save_and_load():
    sets = [sp.StimParamSet("Mmax", nibs_type="PNS",
                            values={"StimulationIntensity": 45},
                            position="tibial_nerve")]
    back = sp.from_dicts(sp.to_dicts(sets))
    assert back == sets


@pytest.mark.parametrize("junk", [None, [], [None], [{}], [{"name": ""}],
                                  ["not a dict"], [{"name": "A|B"}]])
def test_malformed_saved_state_does_not_stop_the_tool_opening(junk):
    out = sp.from_dicts(junk)
    assert all(isinstance(s, sp.StimParamSet) for s in out)
    assert all(sp.sanitise_name(s.name) == s.name for s in out)


def test_an_unknown_saved_type_falls_back_rather_than_raising():
    out = sp.from_dicts([{"name": "A", "nibs_type": "magnets"}])
    assert out[0].nibs_type in sp.NIBS_TYPES
