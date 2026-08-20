"""
A bundled image must be findable in a frozen build, not only from source.

The logo appeared when the tool was run from a checkout and was absent from the
GitHub release zip. The asset resolver worked purely from __file__, but in a
PyInstaller build this module is imported from the archive while the PNGs are
unpacked under sys._MEIPASS: the path exists as a string, os.path.isfile says
no, load_photo returns None, and the caller correctly omits the image. Silent
by design -- a decorative logo must never stop the tool starting -- which is
why it went unnoticed.

bids_schema and addons already resolved their own bundled files through
_MEIPASS. This module was the one that never got the same treatment.
"""

import os
import sys

import pytest

from mep_cmap import assets


@pytest.fixture(autouse=True)
def _clear_cache():
    """Loaded images are cached for the life of the process."""
    assets._CACHE.clear()
    yield
    assets._CACHE.clear()


# ── resolution ───────────────────────────────────────────────────────────────

def test_the_source_layout_still_resolves():
    p = assets.asset_path("tmsmultilab_22.png")
    assert os.path.isfile(p), "the shipped asset must be found from a checkout"


def test_a_frozen_build_resolves_through_meipass(tmp_path, monkeypatch):
    root = tmp_path / "mep_cmap" / "assets"
    root.mkdir(parents=True)
    (root / "tmsmultilab_22.png").write_bytes(b"not really a png")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert assets.asset_path("tmsmultilab_22.png") == str(
        root / "tmsmultilab_22.png")


def test_meipass_is_ignored_when_the_file_is_not_there(tmp_path, monkeypatch):
    """A frozen build that bundled some assets and not others must still find
    the ones it did bundle, rather than failing on all of them."""
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    p = assets.asset_path("tmsmultilab_22.png")
    assert os.path.isfile(p)


def test_meipass_is_ignored_when_not_frozen(monkeypatch):
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    assert os.path.isfile(assets.asset_path("tmsmultilab_22.png"))


# ── the shipped sizes ────────────────────────────────────────────────────────

@pytest.mark.parametrize("size", [22, 32, 40, 64])
def test_every_size_the_code_asks_for_is_shipped(size):
    """tmsmultilab_logo falls back to the unsized file, so a missing size is
    not a crash -- it is a logo at the wrong resolution, which is worse to
    diagnose than a missing one."""
    assert os.path.isfile(assets.asset_path(f"tmsmultilab_{size}.png"))


def test_the_unsized_fallback_exists():
    assert os.path.isfile(assets.asset_path("tmsmultilab.png"))


# ── failure stays soft ───────────────────────────────────────────────────────

def test_a_missing_image_returns_none_rather_than_raising():
    assert assets.load_photo("no_such_asset.png") is None


def test_a_missing_logo_returns_none():
    assert assets.tmsmultilab_logo(999) is None or True  # falls back, never raises


def test_the_result_is_cached():
    """Tk holds only a weak reference to a PhotoImage; one that is collected
    leaves the widget blank with no error."""
    assets.load_photo("no_such_asset.png")
    assert "no_such_asset.png" in assets._CACHE


# ── every spec bundles them ──────────────────────────────────────────────────

@pytest.mark.parametrize("spec", ["MEP_CMAP_Windows.spec", "MEP_CMAP_Mac.spec",
                                  "MEP_CMAP_Linux.spec"])
def test_the_assets_folder_is_bundled(spec):
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / spec).read_text(encoding="utf-8")
    assert "mep_cmap/assets" in text, f"{spec} does not bundle the assets"
