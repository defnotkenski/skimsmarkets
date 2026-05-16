"""One-shot CLI wrapper around the xAI SDK — lets Claude Code (or any
shell user) invoke Grok as a "tool" without standing up an MCP server.

Honors the same `XAI_API_KEY` env var the production fetcher
(`agents/fetchers/grok.py`) uses, so no separate auth setup. Uses the
xAI sync client (the fetcher uses async because it fans out across
events; a one-shot CLI doesn't need that and the sync API is simpler
to read).

Usage:

    uv run python scripts/ask_grok.py "what's new in tennis serve analytics?"
    uv run python scripts/ask_grok.py "..." --search       # web + X search
    uv run python scripts/ask_grok.py "..." --search --code  # also enable code_execution
    uv run python scripts/ask_grok.py "..." --model grok-4-fast  # cheaper

Tools (`web_search`, `x_search`, `code_execution`) are off by default
because the cheapest Grok call is a text-only completion; opt in via
flags when the question actually needs them. Same posture as the
fetcher (which always enables them) but inverted for one-off CLI use
where you usually know whether you need fresh data.

Exits 0 on success, 2 on missing API key, 1 on SDK error.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import system, user
from xai_sdk.tools import code_execution, web_search, x_search

_DEFAULT_MODEL = "grok-4.3"
_DEFAULT_SYSTEM = (
    "You are a focused research assistant. Answer concisely with concrete "
    "facts. When you use web/X search, cite the URLs you actually retrieved "
    "inline. Do not pad with throat-clearing or hedged caveats."
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask Grok a question from the CLI.",
    )
    parser.add_argument("prompt", help="The question to ask Grok.")
    parser.add_argument(
        "--search",
        action="store_true",
        help="Enable web_search + x_search tools (current web/X content).",
    )
    parser.add_argument(
        "--code",
        action="store_true",
        help="Enable code_execution tool (numeric derivations).",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"xAI model name (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--system",
        default=_DEFAULT_SYSTEM,
        help="System prompt prefix.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if not api_key:
        print(
            "ERROR: XAI_API_KEY not set in environment or .env",
            file=sys.stderr,
        )
        return 2

    tools = []
    if args.search:
        tools.extend([web_search(), x_search()])
    if args.code:
        tools.append(code_execution())

    client = Client(api_key=api_key)
    try:
        chat = client.chat.create(
            model=args.model,
            messages=[system(args.system)],
            tools=tools or None,
        )
        chat.append(user(args.prompt))
        response = chat.sample()
    except Exception as e:  # noqa: BLE001 — top-level CLI; surface anything
        print(f"ERROR: xAI SDK call failed: {e}", file=sys.stderr)
        return 1
    finally:
        client.close()

    print(response.content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
