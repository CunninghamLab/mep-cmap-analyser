"""
CED Signal MATLAB exports.

The fixtures here are written by the tests, not shipped: the recordings this
reader was developed against live in a GPL-licensed repository, and copying
them into an MIT one is a licensing question rather than a technical one.
Everything the reader depends on is reproduced instead — including the two
awkward shapes that only real files revealed.
"""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


# ── fixtures ─────────────────────────────────────────────────────────────────

def _write_signal_mat(path, *, var_name="wave_data", n_frame=6, n_chan=2,
                      n_point=1000, interval=0.0002, start=-0.1,
                      titles=("FDI", "APB"), units=("mV", "mV"),
                      labels=None, inline_cells=False):
    """Write a file shaped like Signal's export.

    ``inline_cells`` reproduces what MATLAB does for a ONE-element cell: the
    char array is stored directly instead of as a reference. A single-channel
    export takes that shape, and reading it as references fails silently --
    the channel simply loses its name and unit.
    """
    rng = np.random.default_rng(0)
    values = rng.standard_normal((n_frame, n_chan, n_point)) * 0.01
    stim_idx = int(round(-start / interval))
    for f in range(n_frame):
        for c in range(n_chan):
            values[f, c, stim_idx + 50:stim_idx + 150] += np.hanning(100) * (c + 1)

    labels = labels or [f"State {i % 3 + 1}" for i in range(n_frame)]

    with h5py.File(path, "w", userblock_size=512) as h:
        g = h.create_group(var_name)
        g.create_dataset("values", data=values)
        g.create_dataset("interval", data=np.array([[interval]]))
        g.create_dataset("start", data=np.array([[start]]))
        g.create_dataset("points", data=np.array([[n_point]], dtype=np.int32))
        g.create_dataset("frames", data=np.array([[n_frame]], dtype=np.int32))
        g.create_dataset("chans", data=np.array([[n_chan]], dtype=np.int32))

        def _cell(group, field, strings):
            grp = group.require_group(field.split("/")[0])
            key = field.split("/")[1]
            if inline_cells and len(strings) == 1:
                grp.create_dataset(
                    key, data=np.array([ord(ch) for ch in strings[0]],
                                       dtype=np.uint16))
                return
            refs = []
            for i, sv in enumerate(strings):
                d = h.create_dataset(
                    f"_s_{field.replace('/', '_')}_{i}",
                    data=np.array([ord(ch) for ch in sv], dtype=np.uint16))
                refs.append(d.ref)
            grp.create_dataset(key, data=np.array(refs, dtype=h5py.ref_dtype))

        _cell(g, "chaninfo/title", list(titles[:n_chan]))
        _cell(g, "chaninfo/units", list(units[:n_chan]))
        _cell(g, "frameinfo/label", labels)

    # MATLAB's v7.3 banner lives in the HDF5 user block, and the signature
    # sits at 512 rather than 0 -- which is what made the first version of the
    # detector decline every file it was written for.
    with open(path, "r+b") as fh:
        fh.seek(0)
        fh.write(b"MATLAB 7.3 MAT-file, written by test fixture")


@pytest.fixture
def signal_file(tmp_path):
    p = tmp_path / "sig.mat"
    _write_signal_mat(str(p))
    return str(p)


# ── detection ────────────────────────────────────────────────────────────────

def test_it_is_detected(signal_file):
    from mep_cmap.io import detect_format
    assert detect_format(signal_file) == "signal_mat"


def test_the_signature_is_found_behind_the_user_block(signal_file):
    """v7.3 opens with an ASCII banner; the HDF5 magic is 512 bytes in.

    Testing at offset 0 finds nothing at all.
    """
    with open(signal_file, "rb") as fh:
        head = fh.read(520)
    assert head.startswith(b"MATLAB 7.3")
    assert head[512:520] == b"\x89HDF\r\n\x1a\x0a"


def test_the_variable_name_need_not_be_wave_data(tmp_path):
    """Signal names it after the recording: V231115_SICF000_wave_data."""
    from mep_cmap.io import detect_format
    p = tmp_path / "named.mat"
    _write_signal_mat(str(p), var_name="V231115_SICF000_wave_data")
    assert detect_format(str(p)) == "signal_mat"


def test_a_plain_mat_is_not_claimed(tmp_path):
    from mep_cmap.formats.signal_mat import is_signal_mat
    p = tmp_path / "plain.mat"
    p.write_bytes(b"MATLAB 5.0 MAT-file" + b"\x00" * 200)
    assert is_signal_mat(str(p)) is False


def test_detection_never_raises_on_rubbish(tmp_path):
    from mep_cmap.formats.signal_mat import is_signal_mat
    p = tmp_path / "rubbish.mat"
    p.write_bytes(b"\x00\x01\x02not a mat file at all")
    assert is_signal_mat(str(p)) is False
    assert is_signal_mat(str(tmp_path / "does_not_exist.mat")) is False


# ── the contract ─────────────────────────────────────────────────────────────

def test_channels_come_from_the_file(signal_file):
    from mep_cmap.io import list_waveform_channels
    assert list_waveform_channels(signal_file) == ["FDI", "APB"]


def test_a_single_channel_export_still_names_its_channel(tmp_path):
    """MATLAB stores a one-element cell inline, not as a reference.

    Reading it as a reference fails silently, and the channel comes back as
    'Channel 1' with no unit -- which looks like a file that carries no
    metadata rather than a reader that could not read it.
    """
    from mep_cmap.io import list_waveform_channels
    from mep_cmap.formats.signal_mat import get_unit

    p = tmp_path / "one_chan.mat"
    _write_signal_mat(str(p), n_chan=1, titles=("FDI",), units=("mV",),
                      inline_cells=True)
    assert list_waveform_channels(str(p)) == ["FDI"]
    assert get_unit(str(p), 0) == "mV"


def test_epoch_bounds_match_the_frames(signal_file):
    """start = -0.1 s at 5 kHz over 1000 points is -100 to +100 ms."""
    from mep_cmap.io import get_epoch_bounds
    pre, post = get_epoch_bounds(signal_file)
    assert pre == pytest.approx(100.0)
    assert post == pytest.approx(100.0)


def test_bounds_are_reported_so_windows_can_be_clamped(signal_file):
    """A frame contains nothing outside itself.

    Without bounds an over-long window measures guard padding and reports it
    as recorded signal.
    """
    from mep_cmap.io import get_epoch_bounds
    assert get_epoch_bounds(signal_file) is not None


def test_units_are_stated_not_assumed(signal_file):
    from mep_cmap.io import units_assumed
    assert units_assumed(signal_file) is False


def test_the_sampling_rate_is_the_reciprocal_of_the_interval(signal_file):
    from mep_cmap.io import extract_emg_waveform_and_fs
    _emg, fs, unit = extract_emg_waveform_and_fs(signal_file, 0)
    assert fs == 5000
    assert unit == "mV"


def test_frames_become_stim_types(signal_file):
    from mep_cmap.io import extract_stim_times
    ev = extract_stim_times(signal_file, "A")
    assert set(ev) == {"State 1", "State 2", "State 3"}
    assert sum(len(v) for v in ev.values()) == 6


def test_every_frame_is_accounted_for(tmp_path):
    from mep_cmap.io import extract_stim_times
    from mep_cmap.formats.signal_mat import get_trial_count
    p = tmp_path / "many.mat"
    _write_signal_mat(str(p), n_frame=46)
    ev = extract_stim_times(str(p), "A")
    assert sum(len(v) for v in ev.values()) == get_trial_count(str(p)) == 46


def test_a_single_state_file_yields_one_type(tmp_path):
    from mep_cmap.io import extract_stim_times
    p = tmp_path / "one_state.mat"
    _write_signal_mat(str(p), n_frame=5, labels=["State 1"] * 5)
    assert list(extract_stim_times(str(p), "A")) == ["State 1"]


# ── stitching ────────────────────────────────────────────────────────────────

def test_the_trace_is_guard_padded_not_butted_together(signal_file):
    """Frames are concatenated with guard bands, as epoched_mat does.

    Without them a filter transient at the end of one trial reaches into the
    next, and an over-long window measures the neighbouring response.
    """
    from mep_cmap.formats.epoched_mat import GUARD_MS
    from mep_cmap.io import extract_emg_waveform_and_fs

    emg, fs, _u = extract_emg_waveform_and_fs(signal_file, 0)
    guard_n = int(round(GUARD_MS * fs / 1000.0))
    raw = 6 * 1000
    assert emg.size > raw + guard_n, "no guard band was added"


def test_stim_times_land_on_the_stimulus_in_the_stitched_trace(signal_file):
    """The response was written 50 samples after the trigger in each frame."""
    from mep_cmap.io import extract_emg_waveform_and_fs, extract_stim_times

    emg, fs, _u = extract_emg_waveform_and_fs(signal_file, 0)
    times = sorted(t for v in extract_stim_times(signal_file, "A").values()
                   for t in v)
    for t in times:
        i = int(round(t * fs))
        window = emg[i:i + 200]
        assert np.ptp(window) > 0.5, (
            f"no response at the reported stimulus time {t:.4f} s")


def test_channels_are_independent(signal_file):
    """Channel 2 was written at twice the amplitude of channel 1."""
    from mep_cmap.io import extract_emg_waveform_and_fs
    a, _fs, _u = extract_emg_waveform_and_fs(signal_file, 0)
    b, _fs2, _u2 = extract_emg_waveform_and_fs(signal_file, 1)
    assert np.ptp(b) > np.ptp(a) * 1.5


def test_an_out_of_range_channel_is_refused(signal_file):
    from mep_cmap.io import extract_emg_waveform_and_fs
    with pytest.raises(IndexError):
        extract_emg_waveform_and_fs(signal_file, 9)


# ── no borrowed code ─────────────────────────────────────────────────────────

def test_the_reader_needs_nothing_from_the_matlab_toolbox():
    """Written from the file's own structure, which keeps it MIT-compatible.

    The MATLAB reader for this format belongs to a GPL project, so the reader
    must depend on the format rather than on that source.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
           / "formats" / "signal_mat.py").read_text(encoding="utf-8")
    for borrowed in ("cfs2mat", "readcfs", "matcfs", "CEDS64", "ft_read"):
        assert borrowed not in src


# ── unsupported files must say why ───────────────────────────────────────────

def test_an_unreadable_file_fails_with_a_reason(tmp_path):
    """The file queue warms files up through list_waveform_channels.

    That runs before _browse_file_path and its unsupported-format message, so
    an unreadable file used to surface whatever the text reader said when
    handed binary -- on a build with the Rust extension, "stream did not
    contain valid UTF-8", which names neither the file nor the fix.
    """
    from mep_cmap.io import list_waveform_channels

    p = tmp_path / "mystery.mat"
    p.write_bytes(b"\x00\x01\x02\x03" + bytes(range(256)) * 4)
    with pytest.raises(ValueError) as exc:
        list_waveform_channels(str(p))
    assert "mystery.mat" in str(exc.value), "the message must name the file"
    assert "File" in str(exc.value), "and say where to look"


def test_a_signal_cfs_container_is_recognised_by_name(tmp_path):
    """Signal writes two different MATLAB files.

    Telling the analyst which one they have, and which menu item writes the
    other, is the difference between a dead end and a next step.
    """
    import scipy.io as sio

    from mep_cmap.io import list_waveform_channels

    p = tmp_path / "cfs_dump.mat"
    sio.savemat(str(p), {"CfsFile": np.zeros((2, 2))})
    with pytest.raises(ValueError) as exc:
        list_waveform_channels(str(p))
    msg = str(exc.value)
    assert "CFS-export" in msg
    assert "Export As" in msg, "it must name the menu item that helps"


def test_the_cfs_hint_points_at_the_matlab_export():
    from mep_cmap.io import UNREADABLE_FORMATS
    assert "MATLAB" in UNREADABLE_FORMATS[".cfs"], (
        "the .cfs message predates Signal MATLAB support and should now "
        "point at the export this tool reads")


# ── a missing package is not an unreadable file ──────────────────────────────

def test_a_v73_file_is_recognised_without_h5py(signal_file):
    """The banner check needs no HDF5 library at all."""
    from mep_cmap.formats.signal_mat import looks_like_mat73
    assert looks_like_mat73(signal_file) is True


def test_a_plain_mat_is_not_taken_for_v73(tmp_path):
    from mep_cmap.formats.signal_mat import looks_like_mat73
    p = tmp_path / "old.mat"
    p.write_bytes(b"MATLAB 5.0 MAT-file" + b"\x00" * 600)
    assert looks_like_mat73(str(p)) is False


def test_without_h5py_the_message_names_the_package(signal_file, monkeypatch):
    """It reported a supported format as unreadable.

    is_signal_mat must swallow the ImportError to stay total, so a machine
    without h5py saw "not in a format the tool can read" for a file the tool
    reads perfectly well once the package is there. One of those is
    actionable; the other sends the analyst looking for a different export.
    """
    from mep_cmap import io as _io
    from mep_cmap.formats import signal_mat as sm

    sm.clear_cache()

    def _no_h5py():
        raise ImportError("simulated: h5py is not installed")

    monkeypatch.setattr(sm, "_h5py", _no_h5py)
    with pytest.raises(ValueError) as exc:
        _io.list_waveform_channels(signal_file)
    msg = str(exc.value)
    assert "h5py" in msg
    assert "pip install h5py" in msg
    assert "not in a format the tool can read" not in msg


def test_h5py_is_a_declared_dependency():
    """A Signal export is a supported format, so its reader is not optional."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    assert "h5py" in (root / "requirements.txt").read_text(encoding="utf-8")
    assert "h5py" in (root / "pyproject.toml").read_text(encoding="utf-8")


def test_h5py_is_bundled_into_the_frozen_builds():
    """PyInstaller cannot see a lazily imported package by itself."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    for spec in ("MEP_CMAP_Windows.spec", "MEP_CMAP_Mac.spec",
                 "MEP_CMAP_Linux.spec"):
        text = (root / spec).read_text(encoding="utf-8")
        assert "h5py" in text, f"{spec} would ship without Signal support"
