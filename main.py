"""Convenience shim so `python main.py ...` keeps working (e.g. PyCharm run button).

Prefer `uv run skims ...` for day-to-day use.
"""

from __future__ import annotations

from skimsmarkets.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
