"""Enable PNS in nibs_bep037.json.

v6.3 added PNS as a nibs_type, but every field's `modalities` list predates it,
so a PNS parameter set had no fields at all: the dialogue showed "No parameter
fields apply to PNS" and a peripheral M-wave intensity could not be entered.

Two changes, both additive:

  1. PNS is added to the `modalities` of the fields that genuinely describe
     peripheral electrical stimulation. Coil-specific and field-specific
     fields (current direction in the winding, coil orientation, acoustic
     indices) are deliberately NOT included.
  2. `mA` joins the StimulationIntensityUnits vocabulary. Peripheral stimulators
     are dosed in milliamps and the list held only TMS units, so the number had
     nowhere legal to go. Safe to extend: this field has a blank `emits`, so it
     is this tool's own units declaration and not a spec-closed vocabulary.

Safe to re-run.
"""
import json, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "mep_cmap/schema/nibs_bep037.json"

# Generic pulse-train and dosing description: true of any stimulator that
# delivers discrete pulses, magnetic or electrical.
PNS_FIELDS = [
    "StimulationDescription",
    "StimulationIntensity",
    "StimulationIntensityUnits",
    "PulseShape",
    "StimulationDuration",          # pulse width for peripheral stimulation
    "NumberOfPulses",
    "InterPulseInterval",
    "InterStimulusInterval",
    "ConditioningStimulusIntensity",  # paired-pulse H-reflex conditioning
    "TestStimulusIntensity",
    "PulsesPerBurst",
    "BurstFrequency",
    "PulsesPerTrain",
    "NumberOfTrains",
    "InterTrainInterval",
    "RepetitionRate",
    "Frequency",
    "TargetMuscle",
    "TargetRegion",                 # the stimulated nerve
    "TargetingMethod",
    "AnodeLocation",                # peripheral stimulation uses electrodes
    "CathodeLocation",
    "ElectrodeType",
    "ElectrodeSize",
]

d = json.load(open(PATH, encoding="utf-8"))

added, units_added = [], False
for f in d["fields"]:
    if f["key"] in PNS_FIELDS:
        mods = list(f.get("modalities") or [])
        if "common" not in mods and "PNS" not in mods:
            mods.append("PNS")
            f["modalities"] = mods
            added.append(f["key"])
    if f["key"] == "StimulationIntensityUnits":
        enum = list(f.get("enum") or [])
        if "mA" not in enum:
            # After the TMS units, before the electrical ones already present.
            enum.insert(enum.index("T") if "T" in enum else len(enum), "mA")
            f["enum"] = enum
            units_added = True

json.dump(d, open(PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print("PNS added to", len(added), "field(s)")
for k in added:
    print("   ", k)
print("mA added to StimulationIntensityUnits:", units_added)
missing = [k for k in PNS_FIELDS
           if k not in {f["key"] for f in d["fields"]}]
if missing:
    print("NOT FOUND IN SCHEMA:", missing)
