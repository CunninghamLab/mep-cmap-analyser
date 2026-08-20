"""Separate coil winding direction from induced current direction.

`CurrentDirection` offered PA / AP / LM / ML, which describe the direction of
the induced current in the BRAIN -- the near-universal convention in the TMS
literature. It emitted `current_direction`, which in v6.3 means something else
entirely: the direction of current flow in the COIL WINDING, read in a stated
viewing frame, `cl`/`co` for a circular coil and `cl-co`/`co-cl` naming the left
then right wing of a figure-of-eight.

Writing 'AP' into that column asserts something about the winding that nobody
measured, and a reader following the spec would misread it. The two are not
convertible in either direction: winding direction cannot be derived from PA,
and PA cannot be derived from the winding alone.

So:

  * CurrentDirection keeps the column and takes the spec's closed vocabulary.
    Most analysts will leave it blank, because most do not record it.
  * InducedCurrentDirection is new, holds PA / AP / LM / ML, and goes to
    *_markers.tsv as part of the placement description -- which is what it
    actually is, a consequence of coil geometry and handle orientation. v6.3
    has no column for it, and inventing one would be worse than describing it
    where placements are described.

Safe to re-run.
"""
import json
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "mep_cmap/schema/nibs_bep037.json"

#: v6.3 closed vocabulary, from "Coil current direction".
WINDING = ["cl", "co", "cl-co", "co-cl", "other"]

INDUCED = {
    "key": "InducedCurrentDirection",
    "modalities": ["TMS"],
    "group": "targeting",
    "level": "recommended",
    "type": "string",
    "units": None,
    "enum": ["PA", "AP", "LM", "ML", "other"],
    "advanced": False,
    "description": ("Direction of the current induced in the brain, the "
                    "convention normally reported in TMS work. This is a "
                    "consequence of coil geometry, handle orientation and "
                    "winding direction rather than an independent setting, so "
                    "v6.3 has no column for it: it is written into the "
                    "placement description in *_markers.tsv. Not the same as "
                    "CurrentDirection, which is the current in the coil "
                    "winding."),
    "scope": "parameter_set",
    "block": "markers.tsv",
    "emits": "position_description",
}

d = json.load(open(PATH, encoding="utf-8"))
keys = {f["key"] for f in d["fields"]}

fixed = False
for f in d["fields"]:
    if f["key"] != "CurrentDirection":
        continue
    f["enum"] = list(WINDING)
    f["level"] = "optional"
    f["description"] = (
        "Direction of current flow in the coil winding, from the closed "
        "vocabulary in NIBS-BIDS v6.3. Read in a fixed viewing frame: hold the "
        "coil with its stimulating face away from you and the handle pointing "
        "away, then report the flow in each wing as seen from the top face. "
        "'cl' and 'co' describe a circular coil, clockwise and "
        "counterclockwise; 'cl-co' and 'co-cl' describe a figure-of-eight, "
        "naming the left wing then the right. Leave blank unless you know it — "
        "for the PA/AP convention reported in most TMS work, use "
        "InducedCurrentDirection.")
    fixed = True

added = False
if INDUCED["key"] not in keys:
    idx = next((i for i, f in enumerate(d["fields"])
                if f["key"] == "CurrentDirection"), len(d["fields"]) - 1)
    d["fields"].insert(idx + 1, INDUCED)
    added = True

json.dump(d, open(PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("CurrentDirection vocabulary corrected:", fixed)
print("InducedCurrentDirection added        :", added)
print("total fields                         :", len(d["fields"]))
