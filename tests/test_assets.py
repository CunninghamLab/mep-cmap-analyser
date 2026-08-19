"""
Bundled images.

Decoration, but decoration that has to survive being packaged: the ways an
image fails -- a frozen build that did not bundle the folder, a Tk without PNG
support, a corrupted file -- are all invisible until run time, and none of them
should stop the application starting.
"""

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "mep_cmap" / "assets"
APP = (ROOT / "mep_cmap" / "app.py").read_text(encoding="utf-8")
SPLASH = (ROOT / "splash_screen.py").read_text(encoding="utf-8")


def test_the_logo_is_present_at_the_sizes_used():
    for name in ("tmsmultilab.png", "tmsmultilab_22.png",
                 "tmsmultilab_32.png", "tmsmultilab_40.png",
                 "tmsmultilab_64.png"):
        assert (ASSETS / name).is_file(), f"{name} missing"


def test_loading_never_raises(tmp_path, monkeypatch):
    """A missing or unreadable image returns None and the caller omits it."""
    from mep_cmap.assets import load_photo
    assert load_photo("does_not_exist.png") is None


def test_a_corrupt_file_returns_none(tmp_path, monkeypatch):
    from mep_cmap import assets
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not a png")
    monkeypatch.setattr(assets, "_HERE", str(tmp_path))
    assets._CACHE.clear()
    assert assets.load_photo("broken.png") is None


def test_the_loader_caches():
    """Tk holds only a weak reference to a PhotoImage; an image whose only
    reference is a local variable is collected and the widget draws nothing."""
    from mep_cmap import assets
    assert "_CACHE" in dir(assets)
    src = (ASSETS / "__init__.py").read_text(encoding="utf-8")
    assert "_CACHE[name] = img" in src


def test_widgets_keep_their_own_reference():
    """Belt and braces: the cache keeps it alive, and so does the widget."""
    # Any name: what matters is that a widget holds a reference, not which
    # local variable it came from.
    import re
    assert re.search(r"\.image = \w+", APP), \
        "no widget in app.py keeps a reference to its image"
    assert "badge.image = logo" in SPLASH


def test_the_assets_are_shipped_in_the_wheel():
    txt = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "assets/*.png" in txt


def test_the_assets_are_bundled_into_the_frozen_builds():
    for spec in ("MEP_CMAP_Windows.spec", "MEP_CMAP_Mac.spec",
                 "MEP_CMAP_Linux.spec"):
        txt = (ROOT / spec).read_text(encoding="utf-8")
        assert "mep_cmap/assets" in txt, f"{spec} would ship without the logo"


def test_every_use_of_the_logo_is_guarded():
    """A decorative mark is never worth preventing a window from opening."""
    for src, name in ((APP, "app.py"), (SPLASH, "splash_screen.py")):
        i = 0
        while True:
            i = src.find("tmsmultilab_logo", i + 1)
            if i < 0:
                break
            before = src[max(0, i - 400):i]
            assert "try:" in before, f"{name}: unguarded use of the logo"


# ── authorship ───────────────────────────────────────────────────────────────

def test_the_author_credit_is_gone_from_the_settings_tab():
    """Authorship belongs in CITATION.cff and the Zenodo record, which is
    where anyone citing the tool looks."""
    assert 'text="Author: Justin Andrushko PhD"' not in APP


def test_about_names_the_authors_from_one_string():
    """A name typed into each window falls out of step with the citation file.
    """
    assert "AUTHORS_LINE" in APP
    assert APP.count("AUTHORS_LINE =") == 1


def test_the_gui_and_the_citation_file_agree():
    import ast
    import re

    tree = ast.parse(APP)
    line = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and \
                any(getattr(t, "id", "") == "AUTHORS_LINE" for t in node.targets):
            line = ast.literal_eval(node.value)
    assert line, "AUTHORS_LINE not found"

    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    families = re.findall(r'family-names:\s*"([^"]+)"', cff)
    assert families, "no authors in CITATION.cff"
    for family in families:
        assert family in line, (
            f"{family} is an author in CITATION.cff but not in the About box")


def test_every_author_has_a_valid_orcid():
    """The checksum is ISO 7064 MOD 11-2 over the first fifteen digits.

    A mistyped ORCID does not fail loudly: it resolves to a different
    researcher, or to nothing, and credits the wrong person on a record that
    is meant to be permanent.
    """
    import re

    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    found = re.findall(r"orcid:\s*\"https://orcid\.org/([0-9X-]+)\"", cff)
    assert len(found) >= 2, "expected an ORCID for each author"

    for orcid in found:
        digits = orcid.replace("-", "")
        assert len(digits) == 16, f"{orcid} is not sixteen characters"
        total = 0
        for ch in digits[:15]:
            total = (total + int(ch)) * 2
        check = (12 - total % 11) % 11
        expected = "X" if check == 10 else str(check)
        assert digits[15].upper() == expected, \
            f"{orcid} fails its checksum — it is not a valid ORCID"


def test_the_two_metadata_files_name_the_same_authors():
    """Zenodo reads one and GitHub's citation widget the other; they diverge
    silently, and only a reader comparing them would notice."""
    import json
    import re

    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    families = set(re.findall(r'family-names:\s*"([^"]+)"', cff))
    zen = json.loads((ROOT / "zenodo.json").read_text(encoding="utf-8"))
    names = {c["name"].split(",")[0].strip() for c in zen["creators"]}
    assert families == names, f"CITATION.cff has {families}, zenodo.json has {names}"


def test_the_orcids_agree_between_the_two_files():
    import json
    import re

    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    cff_ids = set(re.findall(r"orcid:\s*\"https://orcid\.org/([0-9X-]+)\"", cff))
    zen = json.loads((ROOT / "zenodo.json").read_text(encoding="utf-8"))
    zen_ids = {c["orcid"] for c in zen["creators"] if "orcid" in c}
    assert cff_ids == zen_ids


# ── where the mark lives ─────────────────────────────────────────────────────

def test_the_mark_sits_above_the_notebook():
    """Outside every tab, so it is present on all of them.

    The derivatives bar it used to sit on changes colour with the folder
    state, so the mark was on red as often as green.
    """
    i = APP.index("_brand = tk.Frame(self.root)")
    j = APP.index("self.notebook = ttk.Notebook(self.root)")
    assert i < j, "the branding strip must be packed before the notebook"


def test_there_is_only_one_copy_of_the_mark_in_the_main_window():
    """Two copies reads as an oversight rather than as branding."""
    import re
    assert len(re.findall(r"tmsmultilab_logo\(\d+\)", APP)) == 2, \
        "one mark in the header, one in the About box"


def test_every_size_asked_for_is_shipped():
    """load_photo falls back to the full-size file, which Tk would then draw at
    140 px -- a mark five times the size intended, rather than a missing one.
    """
    import re
    from mep_cmap.assets import asset_path
    import os
    for size in set(re.findall(r"tmsmultilab_logo\((\d+)\)", APP)):
        assert os.path.isfile(asset_path(f"tmsmultilab_{size}.png")), \
            f"app.py asks for {size} px and no such file is shipped"


def test_the_mark_is_labelled_and_links_out():
    """A mark alone says nothing to a reader who does not recognise it."""
    assert 'text="TMSMultiLab"' in APP
    assert "TMSMultiLab/wiki" in APP


def test_opening_a_link_cannot_raise():
    """A decorative link failing is not worth a dialogue, and the environments
    where it fails -- a headless session, a locked-down desktop -- are ones
    where a traceback would be the more confusing outcome.

    Checked in the source: app.py needs a working matplotlib Tk backend and
    cannot be imported by the suite.
    """
    import ast

    tree = ast.parse(APP)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_open_url":
            body = ast.unparse(node)
            assert "try:" in body and "except Exception:" in body
            return
    raise AssertionError("_open_url not found")


# ── the About window ─────────────────────────────────────────────────────────

def _about():
    i = APP.index("def _show_about")
    j = APP.index("\n    # ", i)
    return APP[i:j]


def test_the_description_names_what_the_tool_measures():
    """It said MEP and cSP, which is two of the several evoked responses this
    quantifies: M-waves, H-reflexes and CMAPs are not MEPs."""
    body = _about()
    assert "BIDS-compliant EMG neurophysiology" in body
    assert "CMAP" in body and "cSP" in body


def test_the_mark_in_about_links_out_like_the_one_in_the_header():
    """The same mark behaving differently in two places is a small puzzle for
    no reason."""
    body = _about()
    assert "_open_url(_TMSML_URL)" in body
    assert 'cursor="hand2"' in body


def test_the_mark_in_about_is_labelled():
    body = _about()
    assert 'text="TMSMultiLab"' in body


def test_the_address_is_written_once():
    """Three widgets now link to it; three copies is how one of them ends up
    pointing somewhere that has moved."""
    assert APP.count('"https://github.com/TMSMultiLab/TMSMultiLab/wiki"') == 1
    assert APP.count("_TMSML_URL") >= 4


def test_the_about_mark_keeps_its_reference():
    body = _about()
    assert "_w.image = _l" in body
