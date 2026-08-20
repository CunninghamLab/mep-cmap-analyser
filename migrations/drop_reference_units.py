"""Remove the reference-as-unit values from StimulationIntensityUnits.

`%RMT`, `%AMT` and `%SI1mV` are v6.2 thinking: they encode WHAT the intensity
was dosed against into the UNIT of the delivered amplitude, which loses both
the measured threshold and the scaling factor. v6.3 separates the two, and the
tool now has IntensityReference and IntensityScaling to carry them.

Leaving both routes in place would mean two ways to say the same thing and only
one of them conformant, and the wrong one is shorter -- so it would get picked.

Nothing has been saved against these values (confirmed with the maintainer), so
this is a straight removal rather than a migration.

The remaining vocabulary is units of a delivered amplitude only:
    %MSO    percent of maximum stimulator output   (TMS)
    mA      milliamps                              (peripheral, tES)
    T       tesla                                  (TMS, measured field)
    A/us    amps per microsecond                   (TMS, coil current slope)
    V       volts

Safe to re-run.
"""
import json, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "mep_cmap/schema/nibs_bep037.json"

DROP = ["%RMT", "%AMT", "%SI1mV"]

d = json.load(open(PATH, encoding="utf-8"))

removed = []
for f in d["fields"]:
    if f["key"] != "StimulationIntensityUnits":
        continue
    enum = list(f.get("enum") or [])
    removed = [v for v in enum if v in DROP]
    f["enum"] = [v for v in enum if v not in DROP]
    f["description"] = (
        "Units of the delivered amplitude in StimulationIntensity. This is the "
        "unit only: what the intensity was dosed against goes in "
        "IntensityReference, and the multiplier in IntensityScaling. So 120% of "
        "a resting motor threshold of 50 %MSO is recorded as intensity 60, "
        "units %MSO, reference rMT, scaling 1.2.")

json.dump(d, open(PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print("removed:", removed or "(none, already gone)")
for f in d["fields"]:
    if f["key"] == "StimulationIntensityUnits":
        print("remaining:", f["enum"])
