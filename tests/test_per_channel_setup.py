"""
Tab 1a is per (channel, stimulus type), not per stimulus type alone.

An iSP recorded on one channel and a contralateral MEP on another are different
muscles under the same marker, so they need different latency profiles. Before
this, one table was applied to whichever channel happened to be selected --
silently, since nothing on the tab said which channel it belonged to.

The pipeline is unaffected: it still receives one flat map per run, and the
analysis hands it each channel's settings in turn. Widening the pipeline's own
maps would have reached the summary builder, the group merge, every add-on and
the Inspector at once.
"""

import ast
import pathlib
import re

APP = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "app.py").read_text(encoding="utf-8")


def _method(name):
    """Source of one method.

    Falls back to end of file: a method that happens to be the last one in the
    class has no following "def" to slice against, and index() then raises
    rather than returning what is there.
    """
    a = APP.index(f"def {name}(self")
    b = APP.find("\n    def ", a + 10)
    return APP[a:] if b == -1 else APP[a:b]


def test_settings_are_stored_per_channel():
    assert "self._chan_settings   = {}" in APP
    keys = APP[APP.index("_chan_settings_keys = ("):]
    keys = keys[:keys.index(")")]
    for expected in ("label_map", "gap_ms_map", "delay_ms_map", "csp_types",
                     "latency_map", "reference_map"):
        assert f'"{expected}"' in keys, f"{expected} is not stored per channel"


def test_switching_channel_saves_the_outgoing_table_first():
    """
    Edits made since the last Confirm Setup would otherwise be lost: the
    widgets hold them, and only harvesting writes them back.
    """
    body = _method("_on_channel_selected")
    harvest = body.index("_harvest_labels_tab")
    snap = body.index("_snapshot_chan_settings")
    switch = body.index("self.channel_idx = new_idx")
    assert harvest < snap < switch, (
        "the outgoing channel must be read and stored before the index moves"
    )


def test_switching_channel_shows_that_channels_own_confirmation():
    """
    Confirmation is per channel because the setup is. A channel already set up
    and confirmed must not read as unconfirmed merely because it was navigated
    away from and back -- and an unconfigured one must not inherit the previous
    channel's green button.
    """
    body = _method("_on_channel_selected")
    assert "_was = new_idx in self._chan_confirmed" in body
    assert "self._labels_tab_confirmed = _was" in body


def test_an_unconfigured_channel_starts_from_defaults():
    """
    Carrying the previous channel's table across was convenient in the
    abstract and wrong in practice: it is a way to inherit another muscle's
    latency profile without noticing, and the only symptom is onsets pinning
    at the bottom of a window the analyst never chose for that channel.

    Copying is now something asked for explicitly, not something that happens
    by moving between channels.
    """
    body = _method("_restore_chan_settings")
    assert "if stored is None:" in body
    assert "_reset_chan_settings_to_defaults()" in body


def test_resetting_clears_every_per_stimulus_map():
    """
    An empty map is how tab 1a signals "no choice made" -- labels fall back to
    the stimulus code, gap and delay to zero, the latency window to the profile
    for the default stimulus type. Clearing is therefore the same as choosing
    defaults, and a map left behind would be a setting the analyst never made.
    """
    body = _method("_reset_chan_settings_to_defaults")
    assert "for key in self._chan_settings_keys:" in body
    assert "set()" in body and "{}" in body, (
        "csp_types is a set and the rest are dicts; both must be handled"
    )


def test_copying_is_the_only_way_settings_move_between_channels():
    switch = _method("_on_channel_selected")
    assert "_reset_chan_settings_to_defaults" not in switch, (
        "the reset belongs in _restore_chan_settings, not the switch handler"
    )
    copy = _method("_copy_setup_to_all_channels")
    assert "_snapshot_chan_settings(idx)" in copy


def test_harvesting_is_separable_from_confirming():
    """Switching channel needs the first without the second."""
    assert "def _harvest_labels_tab(self)" in APP
    confirm = _method("_confirm_labels_tab")
    assert "self._harvest_labels_tab()" in confirm
    assert "_labels_tab_confirmed = True" in confirm


def test_confirming_stores_that_channels_settings():
    """Switching away and back must return what was confirmed."""
    confirm = _method("_confirm_labels_tab")
    assert "_snapshot_chan_settings(self.channel_idx)" in confirm


def test_the_tab_says_which_channel_it_configures():
    """The channel selector is in the file row, far from this table."""
    assert "_labels_chan_lbl" in APP
    assert "Setup below applies to:" in APP


def test_copy_to_all_channels_exists_and_harvests_first():
    body = _method("_copy_setup_to_all_channels")
    assert "_harvest_labels_tab" in body
    assert "_snapshot_chan_settings(idx)" in body


def test_the_1a_grid_has_no_row_collision():
    """
    Inserting the channel banner pushed the header down; the stimulus rows had
    to move with it. Overlapping rows would draw widgets on top of each other.
    """
    a = APP.index('text="Configure labels, colours, and analysis options')
    seg = APP[a:a + 18000]
    static = {int(m.group(1)) for m in re.finditer(r"\.grid\(row=(\d+)", seg)}
    m = re.search(r"enumerate\(sorted\(stim_types\), start=(\d+)\)", seg)
    assert m, "stim row loop not found"
    assert int(m.group(1)) > max(static), (
        f"stimulus rows start at {m.group(1)} but static rows go to "
        f"{max(static)}"
    )


def test_per_channel_settings_survive_a_session_save():
    assert '"chan_settings":' in APP
    assert "sess.get(\"chan_settings\")" in APP
    # JSON has no set type, so csp_types must be converted both ways.
    assert "sorted(vv) if isinstance(vv, set)" in APP
    assert 'set(_snap["csp_types"] or [])' in APP


def test_app_still_parses():
    ast.parse(APP)


# ── Choosing which channels to analyse ───────────────────────────────────────

def test_analysis_channels_default_to_the_selected_one():
    """A single-channel workflow must behave exactly as before, with no opt-in."""
    body = _method("_analysis_channel_indices")
    assert "if self.analyse_channels:" in body
    assert "return [self.channel_idx]" in body


def test_the_analysis_selection_is_separate_from_the_configure_selection():
    """
    One control cannot mean both. The combobox chooses the channel being
    configured and previewed; the analysis set is chosen separately, because
    configuring one channel while analysing four is the normal case.
    """
    assert "self.analyse_channels = set()" in APP
    assert "_choose_analysis_channels" in APP
    a = APP.index("self.channel_dd.bind(")
    b = APP.index("_marker_dd", a)
    assert "_analyse_btn_var" in APP[a:b], "the chooser must sit beside the combobox"


def test_run_analysis_start_still_enforces_the_guards():
    """The guards moved into _validate_analysis_setup; the call must remain.

    Extracting them let Preview detection reuse the same rules. If the call
    were ever dropped, every test below would still pass while an unconfigured
    channel ran straight through.
    """
    body = _method("run_analysis_start")
    assert "self._validate_analysis_setup(require_derivatives=True)" in body
    a = body.index("_validate_analysis_setup")
    b = body.index("threading.Thread", a)
    assert "return" in body[a:b], "an invalid setup must stop the run"


def test_running_is_blocked_when_a_selected_channel_has_no_setup():
    """
    Otherwise the channel silently inherits whichever table was on screen. For
    a different muscle that is the wrong latency profile, and nothing in the
    output would record it.
    """
    body = _method("_validate_analysis_setup")
    assert "_unconfigured_analysis_channels()" in body
    guard = body.index("_unconfigured_analysis_channels()")
    confirm = body.index("_labels_tab_confirmed")
    assert guard < confirm, "check the channels before the confirmation state"
    assert "return" in body[guard:confirm]


def test_the_warning_names_the_channels_and_the_way_out():
    body = _method("_validate_analysis_setup")
    a = body.index("_unconfigured_analysis_channels()")
    seg = body[a:a + 1200]
    assert "Copy this setup to all channels" in seg, (
        "the message should point at the shortcut for the shared case"
    )


def test_unconfigured_channels_are_listed_by_name():
    body = _method("_unconfigured_analysis_channels")
    assert "self._chan_settings" in body
    assert "names[i]" in body


def test_per_channel_setup_does_not_survive_a_new_file():
    """
    The store is keyed by channel INDEX, and an index means nothing across
    files: channel 0 of a LabChart export is not channel 0 of a Spike2
    recording, and may not even be the same muscle.

    Carrying it over restored a Vastus lateralis TMS profile (13-30 ms) onto an
    M-wave recording needing 1-12 ms. Every onset was then reported at 13.2 ms
    -- the bottom of the wrong profile, and a plausible enough number that
    nothing looked broken until the marker was seen sitting on the descending
    limb of the response.
    """
    body = _method("_reset_state_for_new_file")
    assert "self._chan_settings = {}" in body, (
        "per-channel setup must not carry across files"
    )
    assert "self.analyse_channels = set()" in body, (
        "the channel selection belongs to the file that was open"
    )


def test_the_flat_maps_still_persist_across_files():
    """
    Session-level persistence is deliberate -- it saves retyping the whole
    table for every file. Only the per-CHANNEL override is file-level.
    """
    body = _method("_reset_state_for_new_file")
    for key in ("self.label_map = {}", "self.latency_map = {}",
                "self.gap_ms_map = {}"):
        assert key not in body, (
            f"{key} was cleared; the flat maps are session-level by design"
        )


def test_a_saved_session_still_restores_its_own_per_channel_setup():
    """
    Clearing on file change must not defeat session restore. The reset runs
    first and the session for THAT file then repopulates the store, so
    reopening a file returns the setup it was analysed with -- while opening a
    different file starts clean.
    """
    a = APP.index("def _load_file_entry(self")
    b = APP.index("Restored session", a)
    body = APP[a:b]
    reset = body.index("_reset_state_for_new_file()")
    assert reset < len(body), "the reset must run inside the load path"
    # and the restore that follows repopulates it
    assert 'sess.get("chan_settings")' in APP


def test_every_map_the_tab_writes_is_stored_per_channel():
    """
    Storing some of tab 1a's maps but not others lets them drift apart.

    latency_map holds the latency numbers; latency_stim_map and
    latency_muscle_map hold the dropdown choices those numbers come from.
    Switching channel restored one channel's numbers alongside another's
    dropdowns, so the tab read "Peripheral nerve / Upper limb (M-wave)" while
    the window was still the TMS 13-30 ms one. Onsets for those stimulus types
    then pinned at 13 ms -- a plausible latency that was simply the bottom of a
    profile the tab was no longer showing.

    Deriving the list from the harvest method means a map added to the tab
    cannot be forgotten here.
    """
    import re

    a = APP.index("def _harvest_labels_tab")
    b = APP.index("def _set_confirm_state")
    harvested = set(re.findall(r"self\.(\w+)\s*=", APP[a:b]))

    a2 = APP.index("_chan_settings_keys = (")
    stored = set(re.findall(r'"(\w+)"', APP[a2:APP.index(")", a2)]))

    missing = harvested - stored
    assert not missing, (
        f"tab 1a writes these but they are not stored per channel, so they "
        f"will drift out of step with the ones that are: {sorted(missing)}"
    )


def test_a_latency_window_contradicting_its_muscle_group_is_reported():
    """
    A saved window wins over the profile, because a typed value must not be
    overwritten. But it can then contradict the dropdowns above it --
    "Peripheral nerve / Upper limb (M-wave)" sitting over a 13-30 ms TMS
    window -- and the only visible symptom is onsets pinning at 13 ms, which
    looks like a detector fault rather than a settings one.
    """
    assert "_lat_mismatch" in APP
    a = APP.index("_profile = LATENCY_PROFILES.get((_prev_stype, _prev_muscle))")
    seg = APP[a:a + 700]
    assert "_lat_mismatch.append" in seg
    assert "0.05" in seg, "an exact float comparison would fire on rounding"


def test_the_mismatch_warning_says_which_values_are_in_use():
    a = APP.index("Latency window does not match the muscle group")
    seg = APP[a:a + 600]
    assert "values shown are being used" in seg, (
        "the analyst must know which of the two contradicting settings won"
    )


# ── Confirm walks through the selected channels ──────────────────────────────

def test_confirming_moves_to_the_next_channel_that_needs_setup():
    """
    Each channel has its own table, so confirming one says nothing about the
    rest. Jumping straight to filtering made it easy to select four channels
    and configure one.
    """
    body = _method("_confirm_labels_tab")
    assert "_pending = [c for c in self._analysis_channel_indices()" in body
    assert "if _pending:" in body
    assert "self._on_channel_selected()" in body
    # and it stays on 1a rather than moving to filtering
    a = body.index("if _pending:")
    b = body.index("return", a)
    assert "tab1b_frame" in body[a:b]


def test_filtering_is_reached_only_once_every_channel_is_confirmed():
    body = _method("_confirm_labels_tab")
    a = body.index("if _pending:")
    b = body.index("return", a)
    tail = body[b:]
    assert "self.nb_stage1.select(self.tab_filter)" in tail
    assert "every selected" in tail


def test_running_requires_every_selected_channel_confirmed():
    """Confirming the last channel visited says nothing about the others."""
    body = _method("_validate_analysis_setup")
    assert "_unconfirmed = [c for c in self._analysis_channel_indices()" in body
    assert "_chan_confirmed" in body


def test_editing_after_confirmation_un_confirms_that_channel():
    body = _method("_set_confirm_state")
    assert "_chan_confirmed.add(self.channel_idx)" in body
    assert "_chan_confirmed.discard(self.channel_idx)" in body


def test_a_new_file_clears_confirmations():
    body = _method("_reset_state_for_new_file")
    assert "self._chan_confirmed = set()" in body


def test_there_is_only_one_place_a_session_payload_is_built():
    """
    There were two, and they drifted. The manual save carried thirteen fewer
    settings than the automatic one -- latency_map, both latency dropdowns, the
    onset method and every onset detector parameter -- so a manually saved
    session came back without the latency profiles and with the detector reset.

    Nothing announced it: the file loaded and most settings were right.

    The earlier version of this test asserted that BOTH writers carried the
    per-channel keys, which treated the duplication as a fact to work around
    rather than the fault itself. One builder cannot drift from itself.
    """
    import re

    assert "def _session_payload(self" in APP
    # The only place a session dict is assembled is inside the builder itself.
    a = APP.index("def _session_payload(self")
    b = APP.index("\n    def ", a + 10)
    outside = APP[:a] + APP[b:]
    builders = re.findall(r'^\s*session\s*=\s*\{', outside, re.M)
    assert not builders, (
        f"{len(builders)} session dictionary/ies are still built outside "
        f"_session_payload; they will drift from it"
    )
    assert APP.count("self._session_payload(") >= 2, (
        "both the automatic and the manual save should use the builder"
    )


def test_the_single_builder_carries_the_per_channel_state():
    a = APP.index("def _session_payload(self")
    b = APP.index("\n    def ", a + 10)
    body = APP[a:b]
    for key in ('"chan_settings"', '"chan_segment_meta"', '"chan_confirmed"',
                '"analyse_channels"', '"latency_map"', '"latency_stim_map"',
                '"latency_muscle_map"'):
        assert key in body, f"{key} is missing from the session payload"


def test_the_confirmation_state_and_selection_survive_a_reload():
    assert '"chan_confirmed"' in APP
    assert '"analyse_channels"' in APP
    assert 'sess.get("chan_confirmed")' in APP
    assert 'sess.get("analyse_channels")' in APP
