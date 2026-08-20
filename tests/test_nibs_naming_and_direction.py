"""
The stimulation description is *_nibs, and the two current directions are not
the same quantity.

The file was named per modality (*_tms.tsv), which the spec never does: it uses
*_nibs.tsv throughout and carries the modality in the nibs_type column. A
validator, or anyone else's script, looks for *_nibs.tsv and would not find the
description at all.

And CurrentDirection offered PA / AP / LM / ML -- the induced current in the
BRAIN, the convention TMS work reports -- while emitting `current_direction`,
which in v6.3 is the current in the COIL WINDING. Writing 'AP' there asserts
something about the winding nobody measured, and the two are not convertible in
either direction.
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
    def __init__(self, sets, code_sets, values=None):
        self.source_path = "rec.smr"
        self.modality = "TMS"
        self.sidecar_values = values or {}
        self.param_sets = sets
        self.code_sets = code_sets
        self.condition_rows = []
        self.metadata = _Meta()


class _PF:
    def __init__(self, tmp, item):
        self.item = item
        self.rel_dir = "sub-01/ses-01/emg"
        self.nibs_tsv_path = os.path.join(tmp, "x_nibs.tsv")
        self.nibs_json_path = os.path.join(tmp, "x_nibs.json")
        self.markers_tsv_path = os.path.join(tmp, "x_markers.tsv")
        self.markers_json_path = os.path.join(tmp, "x_markers.json")
        self.events_tsv_path = os.path.join(tmp, "x_events.tsv")


def _read_tsv(path):
    with open(path, encoding="utf-8") as fh:
        lines = [l.rstrip("\n") for l in fh if l.strip()]
    head = lines[0].split("\t")
    return head, [dict(zip(head, l.split("\t"))) for l in lines[1:]]


# ── the filename ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("modality", ["TMS", "tES", "TUS", "PNS", "anything"])
def test_the_suffix_is_always_nibs(modality):
    """The modality is a column, not a filename."""
    assert bidsify._suffix_for_modality(modality) == "nibs"


# ── the two current directions ───────────────────────────────────────────────

def test_the_winding_field_takes_the_specs_vocabulary():
    f = load_schema().field("CurrentDirection")
    assert set(f.enum) == {"cl", "co", "cl-co", "co-cl", "other"}
    assert f.emits == "current_direction"


def test_the_brain_direction_is_a_separate_field():
    """PA cannot be derived from the winding, nor the winding from PA."""
    f = load_schema().field("InducedCurrentDirection")
    assert set(f.enum) == {"PA", "AP", "LM", "ML", "other"}
    assert f.emits != "current_direction"


def test_the_brain_direction_goes_to_the_placement():
    """It is a consequence of geometry and handle orientation, which is what
    *_markers.tsv describes. v6.3 has no column of its own for it, and
    inventing one would be worse than describing it where placements are."""
    f = load_schema().field("InducedCurrentDirection")
    assert f.block == "markers.tsv"
    assert f.emits == "position_description"


def test_pa_never_reaches_the_winding_column(tmp_path):
    schema = load_schema()
    sets = [sp.StimParamSet("MEP", nibs_type="TMS", position="M1_left",
                            values={"InducedCurrentDirection": "PA",
                                    "StimulationIntensity": 60})]
    pf = _PF(str(tmp_path), _Item(sets, {"A": "MEP"}))
    bidsify.write_nibs_sidecar(pf, schema)
    head, rows = _read_tsv(pf.nibs_tsv_path)
    assert "PA" not in "\t".join(head)
    for r in rows:
        assert r.get("current_direction", "n/a") in ("n/a", "")


def test_the_brain_direction_is_written_to_the_markers(tmp_path):
    schema = load_schema()
    sets = [sp.StimParamSet("MEP", nibs_type="TMS", position="M1_left",
                            values={"InducedCurrentDirection": "PA"})]
    pf = _PF(str(tmp_path), _Item(sets, {"A": "MEP"}))
    bidsify.write_nibs_sidecar(pf, schema)
    _head, rows = _read_tsv(pf.markers_tsv_path)
    assert "PA" in rows[0]["position_description"]


# ── free text with more than one contributor ─────────────────────────────────

def test_several_sources_of_position_description_are_joined(tmp_path):
    """Three fields emit position_description. Keyed by emits alone they
    overwrite each other and whichever the schema lists last silently wins."""
    schema = load_schema()
    sets = [sp.StimParamSet("MEP", nibs_type="TMS", position="M1_left",
                            values={"InducedCurrentDirection": "PA",
                                    "TargetRegion": "M1 hand hotspot"})]
    pf = _PF(str(tmp_path), _Item(sets, {"A": "MEP"}))
    bidsify.write_nibs_sidecar(pf, schema)
    _head, rows = _read_tsv(pf.markers_tsv_path)
    desc = rows[0]["position_description"]
    assert "M1 hand hotspot" in desc and "PA" in desc


def test_a_placement_set_per_protocol_is_written(tmp_path):
    """A coil moved to a second site mid-recording. Reading only session values
    meant that placement was never written at all."""
    schema = load_schema()
    sets = [sp.StimParamSet("a", nibs_type="TMS", position="site_1",
                            values={"TargetRegion": "M1"}),
            sp.StimParamSet("b", nibs_type="TMS", position="site_2",
                            values={"TargetRegion": "SMA"})]
    pf = _PF(str(tmp_path), _Item(sets, {"A": "a", "G": "b"}))
    bidsify.write_nibs_sidecar(pf, schema)
    _head, rows = _read_tsv(pf.markers_tsv_path)
    by_pos = {r["nibs_position_id"]: r["position_description"] for r in rows}
    assert "M1" in by_pos["site_1"]
    assert "SMA" in by_pos["site_2"]


def test_the_session_value_is_still_the_fallback(tmp_path):
    schema = load_schema()
    sets = [sp.StimParamSet("a", nibs_type="TMS", position="site_1")]
    pf = _PF(str(tmp_path), _Item(sets, {"A": "a"},
                                  values={"TargetRegion": "M1"}))
    bidsify.write_nibs_sidecar(pf, schema)
    _head, rows = _read_tsv(pf.markers_tsv_path)
    assert "M1" in rows[0]["position_description"]
