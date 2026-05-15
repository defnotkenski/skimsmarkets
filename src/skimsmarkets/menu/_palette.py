"""Palette shared across the menu's renderers.

The gold/clay tones echo the banner's ember gradient so the menu reads
as one piece with the wordmark above it. Green/red are reserved for
the form's `▶ run` row — green when safe, red when a danger choice
(currently `execute · mode = LIVE`) is armed and the run will trigger
a confirmation prompt.
"""

from __future__ import annotations

GOLD = "#e8b563"   # focused / selected accent + cycle arrows
CLAY = "#cf6e50"   # rounded panel border
GREEN = "#5cb85c"  # ▶ run row when no danger choice is armed
RED = "#d9534f"    # ▶ run row + value tint when a danger choice is armed
