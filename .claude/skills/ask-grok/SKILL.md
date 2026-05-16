---
name: ask-grok
description: Use ONLY when the user explicitly requests research literature — academic papers, journal articles, arxiv preprints, conference proceedings, or citation lookups on a specific topic. Invokes the `scripts/ask_grok.py` wrapper via Bash. Do NOT use for general knowledge questions, recent news, fact-checking, code lookups, or any other purpose — research literature lookups only.
---

# ask-grok skill

Narrow-scope bridge to Grok for one specific use case: looking up research literature (papers, preprints, citations) on a topic the user named. Grok's web search reliably surfaces recent arxiv / journal / conference content with URLs you can quote back.

## When to fire this skill

Only fires when the user EXPLICITLY asks for research literature. Example phrasings that match:

- "what's the latest research on X"
- "find me papers about Y"
- "what's the literature on Z"
- "any recent publications on W"
- "what does the literature say about V"
- "look up citations for U"

If the user's request doesn't clearly map to "find me academic / research publications on a topic," DO NOT fire this skill. Default to answering from training data.

## When NOT to fire

- Recent news, X activity, fact-checking against live web — NOT this skill's scope, even though Grok could do it
- General knowledge questions (definitions, explanations, comparisons) — Claude answers directly
- Code questions, library lookups, "how does X library work" — use WebFetch or training data
- Anything the user didn't frame as a research / literature request
- Highly sensitive prompts — Grok API logs requests; treat like any external service. If unsure, ask the user before sending.
- Inside a tight loop / fan-out without checking cost first (see "cost note" below)

## Invocation

```bash
uv run python scripts/ask_grok.py "<the user's question>" [--search] [--code]
```

Flags:
- `--search` — enables `web_search` + `x_search`. Use whenever the answer needs current / recent info. Default OFF because the cheapest call is text-only.
- `--code` — enables `code_execution` (Grok runs Python). Use when the question needs numeric derivations the LLM shouldn't eyeball.
- `--model <name>` — defaults to `grok-4.3`. Cheaper option: `grok-4-fast`. Reasoning-heavy: `grok-4` (slower, costlier).
- `--system "<prompt>"` — override the default focused-research-assistant system prompt.

## Reading the output

Bash returns Grok's text response on stdout. Read it, then summarize for the user. Don't paste the entire response verbatim unless it's already concise — pull out the load-bearing facts, preserve any URLs Grok cited so the user can verify.

Exit codes: `0` = success, `2` = missing `XAI_API_KEY`, `1` = SDK / network error (stderr will have the message).

## Cost note

Each call is roughly $0.01-0.05 depending on tool use and response length. Fine for one-off questions. Before fanning out (e.g. "ask Grok about each of these 10 events"), mention the rough total ("~$0.30 across all 10 — okay to proceed?").

## Architectural fit

The skill + script pair is the deliberate middle ground between (1) ad-hoc Bash improvisation and (2) a full MCP server for xAI. It reuses your existing `XAI_API_KEY` and the `xai_sdk` dep your production fetcher already pulls in, so no new infrastructure. If usage grows to "every session, often, with structured params," upgrade to an MCP server. Until then this is the right shape.
