#!/usr/bin/env python
"""
Bump the release version everywhere it appears.

    python bump_version.py 1.2.9
    python bump_version.py 1.2.9 --dry-run

Each location is matched by an anchored pattern rather than by substituting the
old version string, so a file that has drifted out of sync (MEP_CMAP_Mac.spec
sat at 1.2.3 for two releases) is corrected rather than skipped.

DOIs, dependency pins and the CFF schema version are never touched, because
nothing here matches them.

Run check_release.py afterwards to confirm.
"""

import datetime
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
V = r"\d+\.\d+\.\d+"


def rules(new, today, month_year):
    """(filename, description, pattern, replacement) for every version site."""
    return [
        ("mep_cmap/bids.py", "TOOL_VERSION (source of truth)",
         rf'^(TOOL_VERSION\s*=\s*["\'])({V})(["\'])', rf'\g<1>{new}\g<3>'),

        ("pyproject.toml", "project version",
         rf'^(version\s*=\s*["\'])({V})(["\'])', rf'\g<1>{new}\g<3>'),

        ("CITATION.cff", "version",
         rf'^(version:\s*["\'])({V})(["\'])', rf'\g<1>{new}\g<3>'),
        ("CITATION.cff", "date-released",
         r'^(date-released:\s*["\'])([\d-]+)(["\'])', rf'\g<1>{today}\g<3>'),
        ("CITATION.cff", "archived-release description",
         rf'(Archived release of version )({V})', rf'\g<1>{new}'),

        ("zenodo.json", "version",
         rf'("version":\s*")({V})(")', rf'\g<1>{new}\g<3>'),
        ("zenodo.json", "changelog heading",
         rf'(Changes in version )({V})', rf'\g<1>{new}'),

        ("MEP_CMAP_Windows.spec", "bundle name (appears twice)",
         rf"(name='MEP-CMAP Analyser v)({V})(')", rf'\g<1>{new}\g<3>'),

        ("MEP_CMAP_Mac.spec", "CFBundleVersion",
         rf"('CFBundleVersion':\s*')({V})(')", rf'\g<1>{new}\g<3>'),
        ("MEP_CMAP_Mac.spec", "CFBundleShortVersionString",
         rf"('CFBundleShortVersionString':\s*')({V})(')", rf'\g<1>{new}\g<3>'),

        ("README.md", "title version and date",
         rf'^(\*\*Version )({V})( \| )(\w+ \d{{4}})',
         rf'\g<1>{new}\g<3>{month_year}'),
        ("README.md", "point-releases range end",
         rf'(Point releases \({V}[\u2013-])({V})(\))', rf'\g<1>{new}\g<3>'),
        ("README.md", "Zenodo citation version",
         rf'(\(Version v)({V})(\))', rf'\g<1>{new}\g<3>'),
    ]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    dry = "--dry-run" in sys.argv or "-n" in sys.argv

    if len(args) != 1 or not re.fullmatch(V, args[0]):
        sys.exit("usage: python bump_version.py <version>  [--dry-run]\n"
                 "       e.g. python bump_version.py 1.2.9")
    new = args[0]

    now = datetime.date.today()
    today = now.isoformat()
    month_year = now.strftime("%B %Y")

    print(f"Bumping to {new}" + ("  (dry run)" if dry else ""))
    print(f"  date-released -> {today}")
    print(f"  README date   -> {month_year}\n")

    edits = {}
    missing = []
    for name, what, pat, repl in rules(new, today, month_year):
        path = ROOT / name
        if not path.is_file():
            if name not in missing:
                missing.append(name)
            continue
        text = edits.get(name, path.read_text(encoding="utf-8"))
        out, n = re.subn(pat, repl, text, flags=re.M)
        status = f"{n} change(s)" if n else "no match"
        print(f"  {'ok ' if n else '-- '} {name:24} {what:34} {status}")
        if n:
            edits[name] = out
        elif name in edits:
            pass

    if missing:
        print("\n  absent:", ", ".join(missing))

    if not edits:
        print("\nNothing to change.")
        return 0

    if dry:
        print(f"\nDry run — {len(edits)} file(s) would be written.")
        return 0

    for name, text in edits.items():
        (ROOT / name).write_text(text, encoding="utf-8")
    print(f"\nWrote {len(edits)} file(s).")

    print("\nStill needs a human:")
    print("  - zenodo.json: the changelog body still describes the previous")
    print("    release. Rewrite it, including any reprocess warning.")
    print("  - README.md: the point-releases sentence needs its text updated,")
    print("    not just the version number.")
    print("  - CITATION.cff / README: the version-specific DOI can only be set")
    print("    after Zenodo mints it for this release.")
    print("\nNext:  python check_release.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
