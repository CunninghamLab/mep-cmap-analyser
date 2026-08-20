"""Reconcile the stimulus-shape vocabulary and add the missing dosing fields.

Two problems, both left over from the v6.2 reconstruction.

1. `PulseShape` emits `stimulus_shape`, which v6.3 defines as a CLOSED
   vocabulary: Monophasic, Biphasic, Halfsine, Rectangle, Sinusoid, Noise,
   Custom. The tool offered lowercase v6.2 terms plus `biphasic-burst` and
   `other`, none of which are conformant, and offered nothing a peripheral
   stimulator could use -- a square pulse is `Rectangle`.

   `biphasic-burst` maps to `Biphasic`: in v6.3 the burst structure is not part
   of the shape, it is the pattern layers, so a theta-burst protocol is
   Biphasic with pattern1/pattern2 describing the burst and train.
   `other` maps to `Custom`, which v6.3 requires be accompanied by a free-text
   `stimulus_description`.

2. The fields that actually answer "120% of resting motor threshold" were
   missing. v6.3 separates three things the tool was conflating in
   StimulationIntensityUnits: the delivered amplitude (`stimulus_intensity`),
   what it was dosed against (`intensity_reference`), and the multiplier
   (`intensity_scaling`). Without the latter two, %RMT could only be recorded
   as a unit, which loses the measured threshold and the scaling factor.

Safe to re-run.
"""
import json, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "mep_cmap/schema/nibs_bep037.json"

#: v6.3 closed vocabulary for stimulus_shape.
SHAPES = ["Monophasic", "Biphasic", "Halfsine", "Rectangle", "Sinusoid",
          "Noise", "Custom"]

#: Old value -> new. Applied to the enum only; saved values are reported.
SHAPE_MIGRATION = {
    "monophasic": "Monophasic", "biphasic": "Biphasic",
    "halfsine": "Halfsine", "biphasic-burst": "Biphasic",
    "other": "Custom", "square": "Rectangle", "rectangular": "Rectangle",
}

NEW_FIELDS = [
    ("StimulationIntensityUnits", {
        "key": "IntensityReference",
        "modalities": ["common"],
        "group": "parameters",
        "level": "recommended",
        "type": "string",
        "units": None,
        "enum": ["rMT", "aMT", "1mV", "e-field", "absolute"],
        "advanced": False,
        "description": ("What the intensity was dosed against. 'rMT' and 'aMT' "
                        "reference the measured motor thresholds; 'absolute' "
                        "means the amplitude was set directly. The measured "
                        "value of the reference is recorded once per session "
                        "(RestingMotorThreshold / ActiveMotorThreshold)."),
        "scope": "parameter_set",
        "block": "nibs.tsv",
        "emits": "intensity_reference",
    }),
    ("IntensityReference", {
        "key": "IntensityScaling",
        "modalities": ["common"],
        "group": "parameters",
        "level": "recommended",
        "type": "string",
        "units": None,
        "enum": None,
        "advanced": False,
        "description": ("Multiplier applied to the reference, e.g. 1.2 for 120% "
                        "of resting motor threshold, or 'absolute' when the "
                        "amplitude was specified directly. Delivered amplitude "
                        "= reference value x scaling."),
        "scope": "parameter_set",
        "block": "nibs.tsv",
        "emits": "intensity_scaling",
    }),
    ("PulseShape", {
        "key": "FirstInflection",
        "modalities": ["TMS"],
        "group": "parameters",
        "level": "optional",
        "type": "string",
        "units": None,
        "enum": ["rising", "descending"],
        "advanced": True,
        "description": ("Direction of the first deflection of the induced "
                        "current, corresponding to the normal or reverse "
                        "setting on the stimulator."),
        "scope": "parameter_set",
        "block": "nibs.tsv",
        "emits": "first_inflection",
    }),
    ("FirstInflection", {
        "key": "StimulusDescription",
        "modalities": ["common"],
        "group": "parameters",
        "level": "optional",
        "type": "string",
        "units": None,
        "enum": None,
        "advanced": True,
        "description": ("Free-text description of a waveform not covered by the "
                        "shape vocabulary. Required by v6.3 when PulseShape is "
                        "'Custom'."),
        "scope": "parameter_set",
        "block": "nibs.tsv",
        "emits": "stimulus_description",
    }),
]

d = json.load(open(PATH, encoding="utf-8"))
keys = {f["key"] for f in d["fields"]}

old_shapes = []
for f in d["fields"]:
    if f["key"] == "PulseShape":
        old_shapes = list(f.get("enum") or [])
        f["enum"] = list(SHAPES)
        f["description"] = (
            "Waveform shape of the base stimulus, from the closed vocabulary in "
            "NIBS-BIDS v6.3. Monophasic/Biphasic/Halfsine apply to TMS; "
            "Rectangle covers square pulses (tES blocks and peripheral nerve "
            "stimulation); Sinusoid covers tACS and TUS carriers; Custom "
            "requires StimulusDescription.")
        # Rectangle and Sinusoid make it meaningful beyond TMS.
        for m in ("tES", "TUS", "PNS"):
            if m not in f["modalities"] and "common" not in f["modalities"]:
                f["modalities"].append(m)

added = []
for after_key, spec in NEW_FIELDS:
    if spec["key"] in keys:
        continue
    idx = next((i for i, f in enumerate(d["fields"]) if f["key"] == after_key),
               len(d["fields"]) - 1)
    d["fields"].insert(idx + 1, spec)
    keys.add(spec["key"])
    added.append(spec["key"])

json.dump(d, open(PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print("PulseShape enum:", old_shapes, "->", SHAPES)
print("fields added   :", added or "(none, already present)")
print("total fields   :", len(d["fields"]))
print()
print("Saved values needing re-selection, if any exist:")
for k, v in SHAPE_MIGRATION.items():
    print(f"   {k!r} -> {v!r}")
