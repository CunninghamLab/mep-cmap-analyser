"""StimulationIntensityUnits is a declaration, not table data.

The spec is explicit that units are stated in the sidecar and never inferred
from the numbers in the table, so the unit belongs to the COLUMN, not to each
row. Tagged `block: nibs.tsv` with a blank `emits`, it fell through the writer
as a column of its own and was written twice: once as data, once (correctly) as
the Units of stimulus_intensity.

Moved to the sidecar block. It is still read per parameter set, because a PNS
row in mA and a TMS row in %MSO share one column and the writer needs both to
decide what to declare.

Safe to re-run.
"""
import json
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "mep_cmap/schema/nibs_bep037.json"

d = json.load(open(PATH, encoding="utf-8"))
changed = 0
for f in d["fields"]:
    if f["key"] == "StimulationIntensityUnits" and f.get("block") != "":
        f["block"] = ""
        f["emits"] = ""
        f["description"] = (
            "Units of the delivered amplitude in StimulationIntensity. Declared "
            "in *_nibs.json against the stimulus_intensity column, never written "
            "as a column of its own: the spec states units in the sidecar so a "
            "number in the table is never ambiguous. What the intensity was "
            "dosed against goes in IntensityReference, and the multiplier in "
            "IntensityScaling.")
        changed += 1

json.dump(d, open(PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("StimulationIntensityUnits moved to the sidecar:", bool(changed))
