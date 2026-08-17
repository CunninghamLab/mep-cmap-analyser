"""
mep_cmap.tooltips
~~~~~~~~~~~~~~~~~
A hover-and-click tooltip for Tk widgets.

Why this exists
---------------
Settings that need a paragraph of explanation had only one place to receive
it: a block of prose above the table. That has two costs. It occupies the
vertical space the table needs, and it is nowhere near the field it describes,
so the reader has to hold a column name in mind while scrolling to find the
paragraph about it. Explaining a setting beside the setting is both shorter and
easier to follow.

Behaviour
---------
Hovering shows the text after a short delay; leaving hides it. Clicking pins it
open, so a long explanation can be read without holding the mouse still, and
clicking again or pressing Escape closes it. Both are offered because hovering
suits a quick reminder and pinning suits actually reading.

Tk has no tooltip widget, so this is a borderless Toplevel positioned under the
target. It is deliberately small: no images, no markup, no arrow. Everything
here is guarded against TclError, because a tooltip may outlive the widget it
describes when a tab is rebuilt underneath it, and a tooltip is never important
enough to raise.
"""

from __future__ import annotations

import tkinter as tk

#: How long the pointer must rest before an unpinned tooltip appears. Long
#: enough not to flash while the pointer crosses the header on its way
#: somewhere else, short enough not to feel unresponsive.
HOVER_DELAY_MS = 450

#: Wrap width in pixels. These texts are paragraphs, not labels.
WRAP_PX = 420

#: The glyph marking "there is an explanation here". Defined once: a tooltip
#: nobody knows is there explains nothing, so the marker has to be consistent
#: enough to be learned.
INFO_ICON = "\u24d8"


class Tooltip:
    """Attach explanatory text to a widget."""

    def __init__(self, widget, text: str, delay_ms: int = HOVER_DELAY_MS,
                 wrap_px: int = WRAP_PX, pin_on_click: bool = True):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wrap_px = wrap_px
        self._tip = None
        self._after_id = None
        self._pinned = False
        # Pinning is for widgets that do nothing else when clicked. On a
        # control -- a checkbutton, an entry -- the click belongs to the
        # widget, and intercepting it left the box impossible to tick: the
        # tooltip opened and the state never changed.
        self._pin_on_click = bool(pin_on_click)

        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        if self._pin_on_click:
            widget.bind("<Button-1>", self._on_click, add="+")
        widget.bind("<Destroy>", lambda _e: self._hide(force=True), add="+")

    def set_text(self, text: str):
        """Replace the explanation, for settings whose meaning depends on state.

        A tooltip describing anchoring as off while it is on is worse than no
        tooltip, so this exists rather than the caller constructing a second
        Tooltip and leaking the first. An open window is refreshed in place.
        """
        self.text = text or ""
        if self._tip is not None:
            was_pinned = self._pinned
            self._hide()
            self._pinned = was_pinned
            if was_pinned:
                self._show()

    # ── events ───────────────────────────────────────────────────────────────

    def _on_enter(self, _event=None):
        if self._pinned:
            return
        self._cancel()
        try:
            self._after_id = self.widget.after(self.delay_ms, self._show)
        except tk.TclError:
            pass

    def _on_leave(self, _event=None):
        self._cancel()
        if not self._pinned:
            self._hide()

    def _on_click(self, _event=None):
        # Pinning is a toggle rather than a separate control: the icon is
        # already the thing being pointed at, and a close button on a tooltip
        # is more chrome than the text it wraps.
        self._cancel()
        if self._pinned:
            self._pinned = False
            self._hide()
        else:
            self._pinned = True
            self._show()
        # Deliberately no "break": even on a label, swallowing the click stops
        # anything else that legitimately wants it, and there is nothing here
        # worth protecting from a second handler.

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    # ── window ───────────────────────────────────────────────────────────────

    def _show(self):
        self._after_id = None
        if self._tip is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx()
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except tk.TclError:
            return
        try:
            tip = tk.Toplevel(self.widget)
            tip.wm_overrideredirect(True)
            tip.attributes("-topmost", True)
        except tk.TclError:
            return
        frame = tk.Frame(tip, background="#B8B8B8", borderwidth=0)
        frame.pack()
        tk.Label(
            frame, text=self.text, justify="left", wraplength=self.wrap_px,
            background="#FFFFE0", foreground="#222222",
            relief="flat", padx=8, pady=6,
        ).pack(padx=1, pady=1)

        # Keep the window on screen: a tooltip on the last column would
        # otherwise open past the right edge and be unreadable.
        try:
            tip.update_idletasks()
            sw = tip.winfo_screenwidth()
            w = tip.winfo_width()
            if x + w > sw - 8:
                x = max(8, sw - w - 8)
        except tk.TclError:
            pass
        try:
            tip.wm_geometry(f"+{int(x)}+{int(y)}")
            if self._pinned:
                tip.bind("<Button-1>", lambda _e: self._on_click())
                self.widget.winfo_toplevel().bind(
                    "<Escape>", lambda _e: self._on_click(), add="+")
        except tk.TclError:
            pass
        self._tip = tip

    def _hide(self, force: bool = False):
        if force:
            self._pinned = False
        tip, self._tip = self._tip, None
        if tip is None:
            return
        try:
            tip.destroy()
        except tk.TclError:
            pass


def attach_info_icon(parent, text, **grid_kw):
    """Place a small ⓘ beside a heading and return it.

    A separate widget rather than binding the heading itself, so that the
    presence of an explanation is visible before the pointer arrives -- a
    tooltip nobody knows is there explains nothing.
    """
    lbl = tk.Label(parent, text=INFO_ICON, fg="#1F3864", cursor="hand2")
    if grid_kw:
        lbl.grid(**grid_kw)
    # Kept on the widget so a caller can update the text later without having
    # to hold a second reference alongside every label it creates.
    lbl.tooltip = Tooltip(lbl, text)
    return lbl


def label_with_help(parent, text, help_text, suffix=INFO_ICON, **label_kw):
    """A field label that carries its own explanation.

    The icon is appended to the label rather than placed in its own cell,
    because these sit in fixed grid positions and inserting a widget beside
    each one would renumber every column to its right -- a change with more
    ways to go wrong than the thing it is documenting.
    """
    lbl = tk.Label(parent, text=f"{text} {suffix}" if help_text else text,
                   cursor="hand2" if help_text else "", **label_kw)
    lbl.tooltip = Tooltip(lbl, help_text) if help_text else None
    return lbl


def check_with_help(parent, text, help_text, **kw):
    """A checkbutton that carries its own explanation.

    Separate from label_with_help because the icon has to go inside the
    button's own text: a Checkbutton is label and control in one widget, and
    an icon beside it would look like a second thing to click.
    """
    label = "{} {}".format(text, INFO_ICON) if help_text else text
    btn = tk.Checkbutton(parent, text=label,
                         cursor="hand2" if help_text else "", **kw)
    # Hover only: the click toggles the box, and a tooltip that pinned itself
    # instead would make the control unusable.
    btn.tooltip = (Tooltip(btn, help_text, pin_on_click=False)
                   if help_text else None)
    return btn
