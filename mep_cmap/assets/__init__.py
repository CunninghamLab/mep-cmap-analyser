"""
mep_cmap.assets
~~~~~~~~~~~~~~~
Images shipped with the package.

Loading is deliberately total: an image that fails to load returns None and the
caller omits it. A decorative logo is never worth preventing the application
from starting, and the ways it can fail -- a frozen build that did not bundle
the folder, a Tk without PNG support, a corrupted file -- are all invisible
until run time and none of them affect anything the tool does.

Tk holds only a weak reference to a PhotoImage, so an image whose only
reference is a local variable is collected and the widget draws nothing. The
loaded images are cached here, which keeps them alive for the life of the
process and avoids re-decoding the same file for every window that wants it.
"""

from __future__ import annotations

import os

#: Kept alive: Tk does not hold a strong reference to a PhotoImage, and an
#: image that is garbage collected leaves the widget blank with no error.
_CACHE = {}

_HERE = os.path.dirname(os.path.abspath(__file__))


def asset_path(name: str) -> str:
    """Absolute path to a bundled asset, whether installed or frozen."""
    return os.path.join(_HERE, name)


def load_photo(name: str):
    """A ``tk.PhotoImage`` for a bundled PNG, or None if it cannot be loaded.

    None rather than an exception: every caller is decoration, and a missing
    image should leave a gap rather than a traceback.
    """
    if name in _CACHE:
        return _CACHE[name]
    try:
        import tkinter as tk

        path = asset_path(name)
        if not os.path.isfile(path):
            _CACHE[name] = None
            return None
        img = tk.PhotoImage(file=path)
    except Exception:
        _CACHE[name] = None
        return None
    _CACHE[name] = img
    return img


def tmsmultilab_logo(size: int = 22):
    """The TMSMultiLab mark, at one of the sizes shipped.

    Sized copies are shipped rather than scaled at run time because Tk's own
    subsampling is integer-factor only and produces visibly ragged edges on a
    figure of concentric circles.
    """
    for candidate in (f"tmsmultilab_{size}.png", "tmsmultilab.png"):
        img = load_photo(candidate)
        if img is not None:
            return img
    return None
