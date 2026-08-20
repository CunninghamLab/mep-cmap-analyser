"""
A split code needs a parameter set for every half.

Half of 'A' at 100 mA and half at 150 mA is two protocols, not one. Judged on
the bare code, a file with one half assigned and the other not would be offered
for conversion and would write events referencing a set that is not there.
"""

import os

import pytest

from mep_cmap import events_model as em
from mep_cmap import stim_params as sp
from mep_cmap.bids_schema import load_schema
from mep_cmap.bidsify_state import (STATUS_INCOMPLETE, STATUS_READY,
                                    BidsifyState)

SPLITS = [("A", "100mA"), ("A", "150mA")]


def _state(tmp_path, code_sets, codes=("A", "G")):
    st = BidsifyState.load_or_create(str(tmp_path))
    st.param_sets = [sp.StimParamSet("A_100", nibs_type="PNS"),
                     sp.StimParamSet("A_150", nibs_type="PNS"),
                     sp.StimParamSet("MEP", nibs_type="TMS")]
    p = os.path.join(str(tmp_path), "a.smr")
    rec = st.record_for(p)
    rec.reviewed = True
    rec.marker_names = list(codes)
    rec.code_sets = dict(code_sets)
    return st, p


def test_both_halves_assigned_is_ready(tmp_path):
    st, p = _state(tmp_path, {em.pair_key("A", "100mA"): "A_100",
                              em.pair_key("A", "150mA"): "A_150",
                              "G": "MEP"})
    assert st.unassigned_codes(p, SPLITS) == []
    assert st.status(p, load_schema(), SPLITS) == STATUS_READY


def test_one_half_missing_is_incomplete(tmp_path):
    st, p = _state(tmp_path, {em.pair_key("A", "100mA"): "A_100", "G": "MEP"})
    assert st.unassigned_codes(p, SPLITS) == ["A / 150mA"]
    assert st.status(p, load_schema(), SPLITS) == STATUS_INCOMPLETE


def test_the_bare_code_does_not_satisfy_a_split(tmp_path):
    """An assignment on the whole code would silently describe both halves as
    one protocol, which is the thing the split says they are not."""
    st, p = _state(tmp_path, {"A": "A_100", "G": "MEP"})
    assert sorted(st.unassigned_codes(p, SPLITS)) == ["A / 100mA", "A / 150mA"]


def test_an_unsplit_code_still_uses_its_bare_assignment(tmp_path):
    st, p = _state(tmp_path, {em.pair_key("A", "100mA"): "A_100",
                              em.pair_key("A", "150mA"): "A_150"})
    assert st.unassigned_codes(p, SPLITS) == ["G"]


def test_without_splits_the_old_rule_holds(tmp_path):
    """A recording nobody grouped is judged per code, as before."""
    st, p = _state(tmp_path, {"A": "A_100", "G": "MEP"})
    assert st.unassigned_codes(p) == []
    assert st.status(p, load_schema()) == STATUS_READY


def test_an_unticked_code_is_not_demanded(tmp_path):
    st, p = _state(tmp_path, {em.pair_key("A", "100mA"): "A_100",
                              em.pair_key("A", "150mA"): "A_150"},
                   codes=("A",))
    assert st.unassigned_codes(p, SPLITS) == []


def test_a_half_assigned_split_is_not_offered_for_conversion(tmp_path):
    """The gate that matters: converting it would write events referencing a
    parameter set with no row."""
    st, p = _state(tmp_path, {em.pair_key("A", "100mA"): "A_100", "G": "MEP"})
    ready = st.ready_paths([p], load_schema(), lambda _p: SPLITS)
    assert ready == []
    both = dict(st.record_for(p).code_sets)
    both[em.pair_key("A", "150mA")] = "A_150"
    st.record_for(p).code_sets = both
    assert st.ready_paths([p], load_schema(), lambda _p: SPLITS) == [p]


def test_ready_paths_without_a_lookup_is_unchanged(tmp_path):
    st, p = _state(tmp_path, {"A": "A_100", "G": "MEP"})
    assert st.ready_paths([p], load_schema()) == [p]
