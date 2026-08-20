"""
The written stimulation description must match what was entered.

A file with a peripheral M-wave on one code and a TMS MEP on another is the
case the old flat sidecar could not describe. These tests write the v6.3
structure for exactly that file and check the four parts agree.
"""

import json
import os

import pytest

from mep_cmap import bidsify
from mep_cmap import stim_params as sp
from mep_cmap.bids_schema import load_schema


class _Meta:
    measure = "MEP"
    acq = ""


class _Item:
    def __init__(self, sets, code_sets, values=None, condition_rows=None):
        self.source_path = "sub-001_limb-right_000.smr"
        self.modality = "PNS"
        self.sidecar_values = values or {}
        self.param_sets = sets
        self.code_sets = code_sets
        self.condition_rows = condition_rows or []
        self.metadata = _Meta()


class _PF:
    def __init__(self, tmp, item):
        self.item = item
        self.rel_dir = "sub-001/ses-01/emg"
        self.nibs_tsv_path = os.path.join(tmp, "x_nibs.tsv")
        self.nibs_json_path = os.path.join(tmp, "x_nibs.json")
        self.markers_tsv_path = os.path.join(tmp, "x_markers.tsv")
        self.markers_json_path = os.path.join(tmp, "x_markers.json")
        self.events_tsv_path = os.path.join(tmp, "x_events.tsv")


class _Rec:
    def __init__(self, events):
        self._events = events

    def events_table(self):
        return self._events


def _sets():
    return [
        sp.StimParamSet("Mmax", nibs_type="PNS", position="tibial_nerve",
                        values={"StimulationIntensity": 45,
                                "StimulationIntensityUnits": "mA",
                                "PulseShape": "Rectangle"}),
        sp.StimParamSet("MEP_120", nibs_type="TMS", position="M1_left",
                        values={"StimulationIntensity": 60,
                                "StimulationIntensityUnits": "%MSO",
                                "IntensityReference": "rMT",
                                "IntensityScaling": "1.2"}),
    ]


def _read_tsv(path):
    with open(path, encoding="utf-8") as fh:
        lines = [l.rstrip("\n") for l in fh if l.strip()]
    head = lines[0].split("\t")
    return head, [dict(zip(head, l.split("\t"))) for l in lines[1:]]


# ── nibs.tsv ─────────────────────────────────────────────────────────────────

def test_a_mixed_file_writes_one_row_per_protocol(tmp_path):
    """The case the flat sidecar could not describe."""
    pf = _PF(str(tmp_path), _Item(_sets(), {"A": "Mmax", "G": "MEP_120"}))
    bidsify.write_nibs_sidecar(pf, load_schema())

    head, rows = _read_tsv(pf.nibs_tsv_path)
    assert head[:2] == ["nibs_event_id", "nibs_type"]
    assert [r["nibs_event_id"] for r in rows] == ["Mmax", "MEP_120"]
    assert [r["nibs_type"] for r in rows] == ["PNS", "TMS"]
    assert {r["stimulus_intensity"] for r in rows} == {"45", "60"}


def test_dosing_is_recorded_as_reference_and_scaling(tmp_path):
    """120% of rMT, not '%RMT' as a unit: the derivation has to survive."""
    pf = _PF(str(tmp_path), _Item(_sets(), {"G": "MEP_120"}))
    bidsify.write_nibs_sidecar(pf, load_schema())
    head, rows = _read_tsv(pf.nibs_tsv_path)
    assert rows[0]["intensity_reference"] == "rMT"
    assert rows[0]["intensity_scaling"] == "1.2"


def test_only_sets_this_file_uses_are_written(tmp_path):
    """A session set no code here references belongs to another recording."""
    pf = _PF(str(tmp_path), _Item(_sets(), {"A": "Mmax"}))
    bidsify.write_nibs_sidecar(pf, load_schema())
    _, rows = _read_tsv(pf.nibs_tsv_path)
    assert [r["nibs_event_id"] for r in rows] == ["Mmax"]


# ── nibs.json ────────────────────────────────────────────────────────────────

def test_units_are_declared_not_assumed(tmp_path):
    """45 could be mA or %MSO. The spec requires the sidecar say which."""
    pf = _PF(str(tmp_path), _Item(_sets(), {"A": "Mmax"}))
    bidsify.write_nibs_sidecar(pf, load_schema())
    d = json.load(open(pf.nibs_json_path, encoding="utf-8"))
    assert d["stimulus_intensity"]["Units"] == "mA"


def test_mixed_units_are_flagged_rather_than_guessed(tmp_path):
    pf = _PF(str(tmp_path), _Item(_sets(), {"A": "Mmax", "G": "MEP_120"}))
    bidsify.write_nibs_sidecar(pf, load_schema())
    d = json.load(open(pf.nibs_json_path, encoding="utf-8"))
    assert "mixed" in d["stimulus_intensity"]["Units"]


def test_the_timeline_is_pointed_at_not_duplicated(tmp_path):
    pf = _PF(str(tmp_path), _Item(_sets(), {"A": "Mmax"}))
    bidsify.write_nibs_sidecar(pf, load_schema())
    d = json.load(open(pf.nibs_json_path, encoding="utf-8"))
    assert d["IntendedFor"].endswith("x_events.tsv")
    assert d["ConcurrentModalities"] == ["emg"]


def test_the_measured_threshold_is_written_once(tmp_path):
    """The rMT lives in IntensitySet, not on every row."""
    item = _Item(_sets(), {"G": "MEP_120"},
                 values={"RestingMotorThreshold": 50,
                         "MotorThresholdMethod": "5/10 Rossini-Rothwell"})
    pf = _PF(str(tmp_path), item)
    bidsify.write_nibs_sidecar(pf, load_schema())
    d = json.load(open(pf.nibs_json_path, encoding="utf-8"))
    entry = d["IntensitySet"][0]
    assert entry["IntensityID"] == "rMT"
    assert entry["Value"] == 50
    assert entry["Type"] == "resting_motor"
    assert entry["Algorithm"].startswith("5/10")


def test_an_unreferenced_threshold_is_not_written(tmp_path):
    """A threshold nothing was dosed against describes nothing."""
    item = _Item(_sets(), {"A": "Mmax"},
                 values={"RestingMotorThreshold": 50})
    pf = _PF(str(tmp_path), item)
    bidsify.write_nibs_sidecar(pf, load_schema())
    d = json.load(open(pf.nibs_json_path, encoding="utf-8"))
    assert "IntensitySet" not in d


# ── markers.tsv ──────────────────────────────────────────────────────────────

def test_each_placement_gets_a_row(tmp_path):
    pf = _PF(str(tmp_path), _Item(_sets(), {"A": "Mmax", "G": "MEP_120"}))
    bidsify.write_nibs_sidecar(pf, load_schema())
    head, rows = _read_tsv(pf.markers_tsv_path)
    assert head[0] == "nibs_position_id"
    assert {r["nibs_position_id"] for r in rows} == {"tibial_nerve", "M1_left"}


def test_a_set_with_no_position_still_gets_one(tmp_path):
    """A delivery with nowhere to point is unreadable."""
    sets = [sp.StimParamSet("A", nibs_type="PNS")]
    pf = _PF(str(tmp_path), _Item(sets, {"A": "A"}))
    bidsify.write_nibs_sidecar(pf, load_schema())
    _, rows = _read_tsv(pf.markers_tsv_path)
    assert rows[0]["nibs_position_id"] == "A_position"


# ── events.tsv ───────────────────────────────────────────────────────────────

def test_events_reference_the_parameter_set_and_position(tmp_path):
    """The link that makes the whole structure readable."""
    pf = _PF(str(tmp_path), _Item(_sets(), {"A": "Mmax", "G": "MEP_120"}))
    rec = _Rec([{"onset": 1.0, "duration": 0, "trial_type": "A"},
                {"onset": 2.0, "duration": 0, "trial_type": "G"}])
    bidsify.write_events_tsv(pf, rec)
    head, rows = _read_tsv(pf.events_tsv_path)
    assert head == ["onset", "duration", "trial_type",
                    "nibs_event_id", "nibs_position_id"]
    assert rows[0]["nibs_event_id"] == "Mmax"
    assert rows[0]["nibs_position_id"] == "tibial_nerve"
    assert rows[1]["nibs_event_id"] == "MEP_120"


def test_every_event_id_exists_in_the_nibs_table(tmp_path):
    """A dangling reference is worse than no reference."""
    pf = _PF(str(tmp_path), _Item(_sets(), {"A": "Mmax", "G": "MEP_120"}))
    bidsify.write_nibs_sidecar(pf, load_schema())
    rec = _Rec([{"onset": 1.0, "duration": 0, "trial_type": "A"},
                {"onset": 2.0, "duration": 0, "trial_type": "G"}])
    bidsify.write_events_tsv(pf, rec)

    _, nibs_rows = _read_tsv(pf.nibs_tsv_path)
    _, ev_rows = _read_tsv(pf.events_tsv_path)
    _, mk_rows = _read_tsv(pf.markers_tsv_path)
    known = {r["nibs_event_id"] for r in nibs_rows}
    places = {r["nibs_position_id"] for r in mk_rows}
    for r in ev_rows:
        assert r["nibs_event_id"] in known
        assert r["nibs_position_id"] in places


def test_an_unassigned_code_is_marked_not_invented(tmp_path):
    """A code with no set must not silently borrow another's protocol."""
    pf = _PF(str(tmp_path), _Item(_sets(), {"A": "Mmax"}))
    rec = _Rec([{"onset": 1.0, "duration": 0, "trial_type": "A"},
                {"onset": 2.0, "duration": 0, "trial_type": "Z"}])
    bidsify.write_events_tsv(pf, rec)
    _, rows = _read_tsv(pf.events_tsv_path)
    assert rows[1]["nibs_event_id"] == "n/a"


# ── conditions reach the events file ─────────────────────────────────────────

def test_a_split_code_writes_a_set_per_half(tmp_path):
    """Half of A at 100 mA and half at 150 mA. A per-code assignment could not
    express this, and it is the case the Conditions tab exists to describe."""
    from mep_cmap.conditions import ConditionRow
    from mep_cmap import events_model as em

    sets = [sp.StimParamSet("A_100", nibs_type="PNS", position="tibial",
                            values={"StimulationIntensity": 100,
                                    "StimulationIntensityUnits": "mA"}),
            sp.StimParamSet("A_150", nibs_type="PNS", position="tibial",
                            values={"StimulationIntensity": 150,
                                    "StimulationIntensityUnits": "mA"})]
    conds = [ConditionRow(stim_type="A", condition="100mA", trials=(0, 1)),
             ConditionRow(stim_type="A", condition="150mA", trials=(2, 3))]
    code_sets = {em.pair_key("A", "100mA"): "A_100",
                 em.pair_key("A", "150mA"): "A_150"}

    pf = _PF(str(tmp_path), _Item(sets, code_sets, condition_rows=conds))
    bidsify.write_nibs_sidecar(pf, load_schema())
    rec = _Rec([{"onset": float(i), "duration": 0, "trial_type": "A"}
                for i in range(4)])
    bidsify.write_events_tsv(pf, rec)

    _, nibs = _read_tsv(pf.nibs_tsv_path)
    assert {r["nibs_event_id"] for r in nibs} == {"A_100", "A_150"}
    assert {r["stimulus_intensity"] for r in nibs} == {"100", "150"}

    head, ev = _read_tsv(pf.events_tsv_path)
    assert "condition" in head
    assert [r["condition"] for r in ev] == ["100mA", "100mA", "150mA", "150mA"]
    assert [r["nibs_event_id"] for r in ev] == ["A_100", "A_100",
                                                "A_150", "A_150"]


def test_an_ungrouped_recording_writes_its_codes_as_recorded(tmp_path):
    """The ordinary case: most analysts never open the Conditions tab."""
    pf = _PF(str(tmp_path), _Item(_sets(), {"A": "Mmax"}))
    rec = _Rec([{"onset": 1.0, "duration": 0, "trial_type": "A"}])
    bidsify.write_events_tsv(pf, rec)
    head, rows = _read_tsv(pf.events_tsv_path)
    assert "condition" not in head
    assert rows[0]["trial_type"] == "A"


def test_the_events_file_records_where_its_grouping_came_from(tmp_path):
    """A reader needs to know whether the grouping is the recording's or
    somebody's judgement about it."""
    from mep_cmap.conditions import ConditionRow
    pf = _PF(str(tmp_path), _Item(_sets(), {"A": "Mmax"}))
    bidsify.write_events_tsv(pf, _Rec([{"onset": 1.0, "duration": 0,
                                        "trial_type": "A"}]))
    side = json.load(open(os.path.splitext(pf.events_tsv_path)[0] + ".json",
                          encoding="utf-8"))
    assert "No conditions" in side["trial_type"]["Description"]

    pf2 = _PF(str(tmp_path), _Item(
        _sets(), {"A": "Mmax"},
        condition_rows=[ConditionRow(stim_type="A", condition="x",
                                     trials=(0,))]))
    pf2.events_tsv_path = os.path.join(str(tmp_path), "y_events.tsv")
    bidsify.write_events_tsv(pf2, _Rec([{"onset": 1.0, "duration": 0,
                                         "trial_type": "A"}]))
    side2 = json.load(open(os.path.join(str(tmp_path), "y_events.json"),
                           encoding="utf-8"))
    assert "Conditions tab" in side2["trial_type"]["Description"]


# ── device per protocol ──────────────────────────────────────────────────────

def test_two_stimulators_in_one_file_are_both_described(tmp_path):
    """M-waves on a Digitimer and MEPs on a Magstim. A single per-file device
    could describe neither, and v6.3 references the device per row for exactly
    this reason."""
    sets = [sp.StimParamSet("Mmax", nibs_type="PNS",
                            values={"Manufacturer": "Digitimer",
                                    "StimulationIntensity": 45}),
            sp.StimParamSet("MEP_120", nibs_type="TMS",
                            values={"Manufacturer": "Magstim",
                                    "StimulationIntensity": 60})]
    pf = _PF(str(tmp_path), _Item(sets, {"A": "Mmax", "G": "MEP_120"}))
    bidsify.write_nibs_sidecar(pf, load_schema())

    head, rows = _read_tsv(pf.nibs_tsv_path)
    assert "stimulator_id" in head
    assert {r["stimulator_id"] for r in rows} == {"Digitimer", "Magstim"}

    d = json.load(open(pf.nibs_json_path, encoding="utf-8"))
    ids = {e["StimulatorID"] for e in d["StimulatorSet"]}
    assert ids == {"Digitimer", "Magstim"}


def test_one_stimulator_needs_no_column(tmp_path):
    """A study on one device must not carry a column repeating its name on
    every row."""
    pf = _PF(str(tmp_path), _Item(_sets(), {"A": "Mmax", "G": "MEP_120"},
                                  values={"Manufacturer": "Digitimer"}))
    bidsify.write_nibs_sidecar(pf, load_schema())
    head, _ = _read_tsv(pf.nibs_tsv_path)
    assert "stimulator_id" not in head
    d = json.load(open(pf.nibs_json_path, encoding="utf-8"))
    assert d["StimulatorSet"]["Manufacturer"] == "Digitimer"


def test_a_set_without_an_override_inherits_the_shared_default(tmp_path):
    """Blank means "use the shared default", not "no device"."""
    sets = [sp.StimParamSet("Mmax", nibs_type="PNS",
                            values={"Manufacturer": "Digitimer"}),
            sp.StimParamSet("MEP_120", nibs_type="TMS")]
    pf = _PF(str(tmp_path), _Item(sets, {"A": "Mmax", "G": "MEP_120"},
                                  values={"Manufacturer": "Magstim"}))
    bidsify.write_nibs_sidecar(pf, load_schema())
    _, rows = _read_tsv(pf.nibs_tsv_path)
    by_id = {r["nibs_event_id"]: r["stimulator_id"] for r in rows}
    assert by_id["Mmax"] == "Digitimer"
    assert by_id["MEP_120"] == "Magstim"


def test_every_referenced_device_is_defined(tmp_path):
    """A dangling stimulator_id is worse than no column."""
    sets = [sp.StimParamSet("a", nibs_type="TMS",
                            values={"Manufacturer": "Magstim",
                                    "CoilModel": "D70"}),
            sp.StimParamSet("b", nibs_type="TMS",
                            values={"Manufacturer": "Magstim",
                                    "CoilModel": "DCC"})]
    pf = _PF(str(tmp_path), _Item(sets, {"A": "a", "G": "b"}))
    bidsify.write_nibs_sidecar(pf, load_schema())
    head, rows = _read_tsv(pf.nibs_tsv_path)
    d = json.load(open(pf.nibs_json_path, encoding="utf-8"))
    elems = d["ElementSet"]
    elems = elems if isinstance(elems, list) else [elems]
    known = {e["ElementID"] for e in elems}
    assert "nibs_element_id" in head
    for r in rows:
        assert r["nibs_element_id"] in known


def test_sets_sharing_a_device_share_its_entry(tmp_path):
    """Two protocols on one stimulator is one stimulator, not two."""
    sets = [sp.StimParamSet("a", nibs_type="TMS",
                            values={"Manufacturer": "Magstim"}),
            sp.StimParamSet("b", nibs_type="TMS",
                            values={"Manufacturer": "Magstim"}),
            sp.StimParamSet("c", nibs_type="PNS",
                            values={"Manufacturer": "Digitimer"})]
    pf = _PF(str(tmp_path), _Item(sets, {"A": "a", "G": "b", "C": "c"}))
    bidsify.write_nibs_sidecar(pf, load_schema())
    d = json.load(open(pf.nibs_json_path, encoding="utf-8"))
    assert len(d["StimulatorSet"]) == 2


def test_a_per_set_threshold_beats_the_shared_default(tmp_path):
    """A threshold measured for this target rather than the session's."""
    sets = [sp.StimParamSet("MEP_120", nibs_type="TMS",
                            values={"IntensityReference": "rMT",
                                    "RestingMotorThreshold": 62})]
    pf = _PF(str(tmp_path), _Item(sets, {"G": "MEP_120"},
                                  values={"RestingMotorThreshold": 50}))
    bidsify.write_nibs_sidecar(pf, load_schema())
    d = json.load(open(pf.nibs_json_path, encoding="utf-8"))
    assert d["IntensitySet"][0]["Value"] == 62


# ── backward compatibility ───────────────────────────────────────────────────

def test_a_study_without_parameter_sets_still_converts(tmp_path):
    """The old flat sidecar, written honestly rather than an empty table."""
    pf = _PF(str(tmp_path), _Item([], {}, values={"Manufacturer": "Digitimer"}))
    bidsify.write_nibs_sidecar(pf, load_schema())
    assert os.path.isfile(pf.nibs_json_path)
    assert not os.path.isfile(pf.nibs_tsv_path)
    d = json.load(open(pf.nibs_json_path, encoding="utf-8"))
    assert d["Manufacturer"] == "Digitimer"


def test_events_keep_their_old_shape_without_parameter_sets(tmp_path):
    pf = _PF(str(tmp_path), _Item([], {}))
    rec = _Rec([{"onset": 1.0, "duration": 0, "trial_type": "A"}])
    bidsify.write_events_tsv(pf, rec)
    head, _ = _read_tsv(pf.events_tsv_path)
    assert head == ["onset", "duration", "trial_type"]
