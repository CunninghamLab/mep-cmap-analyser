"""
Format sidecars live under derivatives, not in rawdata.

Three readers hold a configuration per recording: which channel Spike2 should
treat as EMG, how a generic TSV's columns map, what unit a pre-epoched MATLAB
file is in. Each wrote it beside the recording, putting one program's settings
into the raw data -- a tree that in a BIDS study is what the acquisition system
wrote, may be read-only, and should be handed to a collaborator without
carrying this tool's state along.

The move matters less than the migration. Changing where a file is looked for
makes every existing one invisible, and a tool that silently forgets what it
was told is worse than one that never remembered: the analyst is asked to
configure files they configured last week, with no explanation and no clue that
an answer exists.
"""

import json
import os

import pytest

from mep_cmap import sidecars as S

SUFFIX = ".smr_config.json"


@pytest.fixture
def study(tmp_path):
    raw = tmp_path / "rawdata" / "sub-01"
    raw.mkdir(parents=True)
    rec = raw / "sub-01_emg.smr"
    rec.write_bytes(b"")
    S.set_derivatives_root(str(tmp_path / "derivatives"))
    yield tmp_path, str(rec)
    S.set_derivatives_root(None)


def test_a_new_sidecar_goes_under_derivatives(study):
    _root, rec = study
    p = S.sidecar_path(rec, SUFFIX)
    assert "derivatives" in str(p)
    assert "rawdata" not in str(p)


def test_an_existing_sidecar_is_migrated_not_ignored(study):
    """A study configured last week must not be asked to configure itself
    again."""
    _root, rec = study
    old = S.legacy_path(rec, SUFFIX)
    old.write_text(json.dumps({"emg_channel": "EMG1"}), encoding="utf-8")

    got = S.read(rec, SUFFIX)
    assert got == {"emg_channel": "EMG1"}
    assert not old.exists(), "the old copy should have been moved, not copied"
    assert S.sidecar_path(rec, SUFFIX).exists()


def test_migration_does_not_lose_the_file_if_it_cannot_move(study, monkeypatch):
    """A read-only rawdata tree is exactly the case this change helps; being
    unable to tidy it is not a reason to refuse to read it."""
    _root, rec = study
    old = S.legacy_path(rec, SUFFIX)
    old.write_text(json.dumps({"emg_channel": "EMG1"}), encoding="utf-8")

    def _refuse(*_a, **_k):
        raise OSError("read-only")
    monkeypatch.setattr(S.shutil, "move", _refuse)

    assert S.read(rec, SUFFIX) == {"emg_channel": "EMG1"}
    assert old.exists(), "the original must survive a failed migration"


def test_with_no_derivatives_root_the_old_location_is_used(tmp_path):
    """Every earlier version wrote beside the recording, so a study that has
    not chosen a derivatives folder behaves exactly as it always has."""
    S.set_derivatives_root(None)
    rec = tmp_path / "x_emg.smr"
    rec.write_bytes(b"")
    assert S.sidecar_path(str(rec), SUFFIX) == S.legacy_path(str(rec), SUFFIX)


def test_derivatives_is_not_nested_twice(tmp_path):
    S.set_derivatives_root(str(tmp_path / "derivatives"))
    p = S.sidecar_path(str(tmp_path / "x_emg.smr"), SUFFIX)
    # Count path SEGMENTS, not occurrences in the string: pytest names its own
    # temporary directory after the test, so a substring count picks that up.
    assert [part for part in p.parts].count("derivatives") == 1
    S.set_derivatives_root(None)


def test_reading_a_missing_sidecar_returns_none(study):
    _root, rec = study
    assert S.read(rec, SUFFIX) is None


def test_writing_creates_the_folder(study):
    _root, rec = study
    assert S.write(rec, SUFFIX, {"emg_channel": "EMG2"}) is True
    assert S.read(rec, SUFFIX) == {"emg_channel": "EMG2"}


def test_removal_clears_both_locations(study):
    """Clearing only the new place would leave the old one to be migrated
    straight back, so a reset would restore what it had just discarded."""
    _root, rec = study
    S.legacy_path(rec, SUFFIX).write_text("{}", encoding="utf-8")
    S.write(rec, SUFFIX, {"a": 1})
    S.remove(rec, SUFFIX)
    assert not S.legacy_path(rec, SUFFIX).exists()
    assert not S.sidecar_path(rec, SUFFIX).exists()


# ── the readers go through it ────────────────────────────────────────────────

@pytest.mark.parametrize("mod,suffix", [
    ("spike2_smr", ".smr_config.json"),
    ("generic_tsv", ".tsv_config.json"),
    ("epoched_mat", ".epoched_config.json"),
])
def test_every_reader_uses_the_shared_locator(mod, suffix):
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent / "mep_cmap" /
           "formats" / f"{mod}.py").read_text(encoding="utf-8")
    assert "from ..sidecars import resolve" in src
    assert f'SIDECAR_SUFFIX = "{suffix}"' in src
    assert "with_suffix" not in src.split("def _sidecar_path")[1][:400]


def test_the_reader_round_trips_through_the_new_location(study):
    from mep_cmap.formats import spike2_smr as SMR
    _root, rec = study
    SMR.save_config(rec, "EMG1", "DigMark")
    assert SMR.has_config(rec)
    assert SMR.load_config(rec)["emg_channel"] == "EMG1"
    assert "derivatives" in str(S.sidecar_path(rec, SUFFIX))


def test_the_app_syncs_the_root_whenever_it_changes():
    """Four call sites set that path; a trace means none of them has to
    remember."""
    import pathlib
    app = (pathlib.Path(__file__).resolve().parent.parent / "mep_cmap" /
           "app.py").read_text(encoding="utf-8")
    assert "_sync_sidecar_root" in app
    assert "derivatives_path.trace_add" in app


def test_reset_clears_sidecars_from_both_places():
    import pathlib
    app = (pathlib.Path(__file__).resolve().parent.parent / "mep_cmap" /
           "app.py").read_text(encoding="utf-8")
    i = app.index("def _queue_reset_file")
    j = app.index("\n    def ", i + 10)
    assert "remove as _rm_sidecar" in app[i:j]
