"""Retag nibs_bep037.json for NIBS-BIDS v6.3.

One-shot migration. Additive only: every existing field key is kept, so a
bidsify_state.json written by 1.4.0 still loads. Two keys are added per field:

    block  where the value serialises: a JSON block (StimulatorSet, ElementSet,
           IntensitySet, NavigationSystem), a tabular file (nibs.tsv,
           markers.tsv, events.tsv, coordsystem.json), or "" for a top-level
           key of *_nibs.json.
    emits  the v6.3 name the value is written under. "" means the key itself.

and `scope` is corrected: v6.3 puts most stimulation parameters on the
stimulation parameter set, not on the recording.

Keys are NOT renamed. The spec has already renamed things twice; a rename here
would break every saved state file, so the spec name lives in `emits` and the
tool's own key stays put.

Safe to re-run: the mapping is absolute, not relative to current values.
"""
import json, sys, collections

PATH = sys.argv[1] if len(sys.argv) > 1 else "mep_cmap/schema/nibs_bep037.json"

# key -> (block, emits, scope)
M = {
 # legacy: still read from old state files, no longer written
 "StimulationModality":          ("", "nibs_type", "session"),
 "PulseConfiguration":           ("", "", "file"),

 # top level of *_nibs.json
 "StimulationDescription":       ("", "NIBSDescription", "file"),
 "TargetMuscle":                 ("", "", "file"),

 # StimulatorSet (the box that makes the waveform)
 "Manufacturer":                 ("StimulatorSet", "Manufacturer", "session"),
 "ManufacturersModelName":       ("StimulatorSet", "ManufacturerModelName", "session"),
 "DeviceSerialNumber":           ("StimulatorSet", "ManufacturerSerialNumber", "session"),
 "SoftwareVersions":             ("StimulatorSet", "SoftwareVersion", "session"),

 # ElementSet (the thing on the participant)
 "CoilModel":                    ("ElementSet", "ModelName", "session"),
 "CoilType":                     ("ElementSet", "CoilShape", "session"),
 "CoilManufacturer":             ("ElementSet", "Manufacturer", "session"),
 "ElectrodeType":                ("ElementSet", "ElectrodeShape", "session"),
 "ElectrodeSize":                ("ElementSet", "ElectrodeArea", "session"),
 "TransducerModel":              ("ElementSet", "ModelName", "session"),

 # IntensitySet (measured dosing references)
 "RestingMotorThreshold":        ("IntensitySet", "Value", "session"),
 "ActiveMotorThreshold":         ("IntensitySet", "Value", "session"),
 "MotorThresholdMethod":         ("IntensitySet", "Algorithm", "session"),

 # nibs.tsv, per parameter set
 "PulseShape":                   ("nibs.tsv", "stimulus_shape", "parameter_set"),
 "CurrentDirection":             ("nibs.tsv", "current_direction", "parameter_set"),
 "StimulationIntensity":         ("nibs.tsv", "stimulus_intensity", "parameter_set"),
 "StimulationIntensityUnits":    ("nibs.tsv", "", "parameter_set"),
 "ConditioningStimulusIntensity":("nibs.tsv", "pattern1_intensity", "parameter_set"),
 "TestStimulusIntensity":        ("nibs.tsv", "pattern1_intensity", "parameter_set"),
 "InterPulseInterval":           ("nibs.tsv", "pattern1_interval", "parameter_set"),
 "InterStimulusInterval":        ("nibs.tsv", "pattern1_interval", "parameter_set"),
 "NumberOfPulses":               ("nibs.tsv", "stimulus_pulses_number", "parameter_set"),
 "PulsesPerBurst":               ("nibs.tsv", "pattern1_count", "parameter_set"),
 "BurstFrequency":               ("nibs.tsv", "pattern1_frequency", "parameter_set"),
 "PulsesPerTrain":               ("nibs.tsv", "pattern2_count", "parameter_set"),
 "RepetitionRate":               ("nibs.tsv", "pattern2_frequency", "parameter_set"),
 "NumberOfTrains":               ("nibs.tsv", "pattern3_count", "parameter_set"),
 "InterTrainInterval":           ("nibs.tsv", "pattern3_interval", "parameter_set"),
 "StimulationType":              ("nibs.tsv", "stimulus_shape", "parameter_set"),
 "Current":                      ("nibs.tsv", "stimulus_intensity", "parameter_set"),
 "CurrentDensity":               ("nibs.tsv", "", "parameter_set"),
 "StimulationDuration":          ("nibs.tsv", "stimulus_duration", "parameter_set"),
 "FadeInDuration":               ("nibs.tsv", "ramp_up", "parameter_set"),
 "FadeOutDuration":              ("nibs.tsv", "ramp_down", "parameter_set"),
 "Frequency":                    ("nibs.tsv", "frequency", "parameter_set"),
 "PhaseOffset":                  ("nibs.tsv", "starting_phase", "parameter_set"),
 "FundamentalFrequency":         ("nibs.tsv", "frequency", "parameter_set"),
 "AcousticIntensitySPPA":        ("nibs.tsv", "spatial_peak_pulse_average_intensity", "parameter_set"),
 "MechanicalIndex":              ("nibs.tsv", "mechanical_index", "parameter_set"),
 "PulseRepetitionFrequency":     ("nibs.tsv", "pattern1_frequency", "parameter_set"),
 "DutyCycle":                    ("nibs.tsv", "duty_cycle", "parameter_set"),
 "SonicationDuration":           ("nibs.tsv", "stimulus_duration", "parameter_set"),

 # markers.tsv / coordsystem.json / NavigationSystem
 "TargetRegion":                 ("markers.tsv", "position_description", "parameter_set"),
 "TargetingMethod":              ("markers.tsv", "coil_positioning_method", "parameter_set"),
 "CoilOrientation":              ("markers.tsv", "coil_handle_direction", "parameter_set"),
 "TargetCoordinates":            ("markers.tsv", "", "parameter_set"),
 "MontageDescription":           ("markers.tsv", "position_description", "session"),
 "AnodeLocation":                ("markers.tsv", "position_label", "parameter_set"),
 "CathodeLocation":              ("markers.tsv", "position_label", "parameter_set"),
 "NeuronavigationSystem":        ("NavigationSystem", "NavigationSystemName", "session"),
 "StructuralMRI":                ("coordsystem.json", "IntendedFor", "session"),
 "TargetCoordinateSystem":       ("coordsystem.json", "NIBSCoordinateSystem", "session"),
}

LEGACY = {"StimulationModality", "PulseConfiguration"}

d = json.load(open(PATH, encoding="utf-8"))

d["schema_version"] = "0.3.0"
d["based_on"] = (
    "NIBS-BIDS (BEP037) proposal v6.3, from the development repository at "
    "https://github.com/nigelrogasch/nibs-bids (specification/nibs-bids.md), "
    "cross-checked against the preprint at doi:10.5281/zenodo.19337642. "
    "Earlier versions of this schema were a reconstruction of v6.2; field "
    "names are now taken from the spec text itself. Field keys below are this "
    "tool's own stable identifiers and are NOT renamed when the spec renames "
    "things: 'emits' carries the v6.3 output name and 'block' says which file "
    "or JSON block it is written into, so a spec rename is an 'emits' change "
    "and never breaks a saved bidsify_state.json."
)
if "PNS" not in d.get("modalities", []):
    d["modalities"].append("PNS")
d["scopes"] = ["session", "file", "parameter_set"]
d["_field_notes"] += (
    " block: where the value serialises (StimulatorSet | ElementSet | "
    "IntensitySet | NavigationSystem | nibs.tsv | markers.tsv | events.tsv | "
    "coordsystem.json | '' for a top-level *_nibs.json key). emits: the v6.3 "
    "name written under, '' meaning the key itself. legacy: read from older "
    "saved state, never written. scope gains 'parameter_set' = varies by "
    "stimulation parameter set (per stim code), which is where v6.3 puts most "
    "stimulation parameters."
)

missing, counts = [], collections.Counter()
for f in d["fields"]:
    k = f["key"]
    if k not in M:
        missing.append(k)
        continue
    block, emits, scope = M[k]
    f["block"] = block
    f["emits"] = emits
    if scope:
        f["scope"] = scope
    if k in LEGACY:
        f["legacy"] = True
    counts[scope or f.get("scope")] += 1

    if k == "StimulationModality" and "PNS" not in (f.get("enum") or []):
        f["enum"] = list(f["enum"]) + ["PNS"]

json.dump(d, open(PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print("schema_version:", d["schema_version"])
print("fields tagged :", len(d["fields"]) - len(missing), "/", len(d["fields"]))
print("by scope      :", dict(counts))
if missing:
    print("NOT IN MAPPING (left untouched):", missing)
