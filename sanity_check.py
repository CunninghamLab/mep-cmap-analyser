"""
sanity_check.py  —  throwaway dev check for the new BIDS-ify pieces.

Place this BESIDE the package (in modular_build\\, next to the mep_cmap\\ folder),
NOT inside mep_cmap\\. It is read-only: it reads data through your existing
io.py and writes nothing.

Usage (Windows CMD, from D:\\MEP-CMAP_Analyser\\modular_build):

    rem schema check only:
    "<venv>\\Scripts\\python.exe" sanity_check.py

    rem schema check + Recording check on one real file:
    "<venv>\\Scripts\\python.exe" sanity_check.py "D:\\path\\to\\recording.txt"

    rem also pull events for one or more stim marker labels:
    "<venv>\\Scripts\\python.exe" sanity_check.py "D:\\path\\to\\recording.txt" A B

What to look for
----------------
Schema:    the "default TMS prompt" list = exactly the fields a TMS user is asked
           for up front. Is anything you always record missing? Anything useless
           cluttering it? (Edit nibs_bep037.json, not code.)
Recording: channel names/units match what the GUI shows; fs and duration match the
           recording you know; event count is roughly the number of stims (if you
           passed a marker); warnings list is empty or explainable.
"""

import sys
import json


def schema_check():
    print("=" * 70)
    print("SCHEMA CHECK  (mep_cmap.bids_schema + schema/nibs_bep037.json)")
    print("=" * 70)
    from mep_cmap.bids_schema import load_schema
    s = load_schema()
    print(f"schema: {s.schema_name} v{s.schema_version}")
    print(f"based on: {s.based_on[:90]}...")
    print(f"total fields: {len(s.fields)} | modalities: {s.modalities}")
    for m in s.modalities:
        print(f"  applicable to {m}: {len(s.fields_for(m))}")

    print("\n-- default TMS prompt (recommended/required, non-advanced) --")
    for f in s.fields_for("TMS", include_advanced=False):
        units = f" [{f.units}]" if f.units else ""
        enum = f"  one of {list(f.enum)}" if f.enum else ""
        print(f"  [{f.level:11s}] {f.group:10s} {f.key}{units}{enum}")

    print("\n-- advanced TMS fields (collapsed by default in the dialog) --")
    adv = [f for f in s.fields_for("TMS") if f.advanced]
    print("  " + ", ".join(f.key for f in adv))

    print("\n-- demo validate: a typical single-pulse MEP setup, no neuronav --")
    demo = {
        "StimulationModality": "TMS",
        "Manufacturer": "Magstim",
        "CoilType": "figure-of-eight",
        "PulseShape": "monophasic",
        "CurrentDirection": "PA",
        "PulseConfiguration": "single-pulse",
        "TargetMuscle": "FDI",
        "StimulationIntensity": "120",
        "StimulationIntensityUnits": "%RMT",
        "RestingMotorThreshold": "45",
        "TargetRegion": "left M1 hand knob",
        "TargetingMethod": "scalp-heuristic",
    }
    res = s.validate(demo, modality="TMS")
    print(f"  validates: {res.ok}  | errors: {res.errors}")
    print(f"  recommended-but-missing: {len(res.warnings)}")
    print("\n-- resulting nibs sidecar --")
    print(json.dumps(s.ordered_sidecar(demo, modality="TMS"), indent=2))


def recording_check(path, markers):
    print("\n" + "=" * 70)
    print(f"RECORDING CHECK  (build_recording on {path})")
    print("=" * 70)
    try:
        from mep_cmap.recording import build_recording
    except Exception as exc:
        print("Could not import mep_cmap.recording / io — run this from the")
        print("modular_build folder (the parent of mep_cmap\\). Error:")
        print(f"  {type(exc).__name__}: {exc}")
        return

    try:
        rec = build_recording(path, marker_names=markers or None)
    except Exception as exc:
        print(f"build_recording failed: {type(exc).__name__}: {exc}")
        print("If the signature differs from your live io.py, adjust the calls")
        print("inside build_recording() in recording.py.")
        return

    print("summary       :", rec.summary())
    print("source format :", rec.source_format)
    print("channels      :", rec.channel_names)
    print("units         :", rec.units)
    print("sampling freq :", rec.sampling_frequency, "Hz")
    print("n_samples     :", rec.n_samples, f"(~{rec.duration_s:.2f} s)")
    print("equal length  :", rec.channels_equal_length())
    print("events        :", len(rec.events), "(markers:", markers or "none requested", ")")
    if rec.events:
        rows = rec.events_table()[:6]
        print("first events  :", [(round(r["onset"], 4), r["trial_type"]) for r in rows])
    if rec.warnings:
        print("WARNINGS:")
        for w in rec.warnings:
            print("   -", w)
    else:
        print("warnings      : none")

    try:
        mtx = rec.data_matrix(on_length_mismatch="truncate")
        print("data_matrix   :", mtx.shape, mtx.dtype)
    except Exception as exc:
        print("data_matrix   : could not assemble:", exc)

    print("signature     :")
    print(json.dumps(rec.signature(), indent=2))


if __name__ == "__main__":
    schema_check()
    if len(sys.argv) > 1:
        recording_check(sys.argv[1], sys.argv[2:])
    else:
        print("\n(no data file given — pass a path as the first argument to also")
        print(" run the Recording check, e.g. sanity_check.py \"D:\\data\\file.txt\")")
