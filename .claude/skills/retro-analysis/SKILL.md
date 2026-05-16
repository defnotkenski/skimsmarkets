---
name: retro-analysis
description: Use when the user asks for a retro analysis pass — phrases like "do a retro analysis", "do a retro pass", "look at recent results", "what's retro saying", "analyze settled events", "could you analyze the logs". Auto-fires the conversational retro flow so the user doesn't have to point at the playbook file each time.
---

# retro-analysis skill

Conversational retro analysis — the LLM half of `skims retro --step analyze`, run in chat to save the LLM-API cost the operator deliberately skips.

## Authoritative source

The full discipline, anti-hindsight rules, output shape, and per-sport notes live in `playbooks/retro-analysis.md`. **Read and follow that file end-to-end** before producing a report. This skill is a thin pointer that ensures you remember to consult the playbook on every retro invocation; the playbook itself is the source of truth.

## Quick reminders (the playbook elaborates each one)

- **Run `uv run skims retro --step calibrate`** for the deterministic cuts (free; no LLM). Do NOT run `--step analyze` — that's the path the operator skips on cost grounds.
- **Fetch post-match data if the cache is empty** for the dates being analyzed. The playbook has the exact `fetch_post_match_for_settled` invocation. Without it the high-yield `divergence_*` cuts can't be computed.
- **Anti-hindsight discipline.** Only report patterns that recur across MULTIPLE losses AND are clearly less frequent in wins.
- **Small-N honesty.** At n<10 losses, label everything "directional only." Push back on prompt-edit recommendations regardless of how confident the pattern looks.
- **Honor CLAUDE.md invariants** on any prompt-edit recommendation (confidence-as-contingency-robustness, market-as-prior, canonical names, lens silos, no edge/buy/pass framing).

## Output shape

Mirrors `logs/retro/*.report.md`:

1. **Headline** — `N settled / M unique markets, K correct (X%)`.
2. **Hit-rate cuts** — from calibrate output (case bucket / confidence / favorite-vs-underdog / negative-edge / director-vs-GBT divergence).
3. **Per-sport patterns** — `recurring_patterns`, optional `lens_underperformance` (omit when no clean attribution), `prompt_recommendations` (2-5 concrete edits targeting specific surfaces).
4. **Pushbacks** — explicit list of which recommendations NOT to adopt and why. At small N this section is usually the longest.

## What NOT to do

- Don't run `skims retro --step analyze` (the LLM-API path the operator skips).
- Don't recommend prompt edits at n<5 losses without flagging "directional only — not actionable yet."
- Don't suggest building the Phase A/A.5/B/C self-improvement infrastructure unless the user explicitly revisits it (parked plan in `~/.claude/projects/-Users-kennylao-PycharmProjects-SkimsMarkets/memory/project_retro_lineage_parked.md`).
- Don't ask the user to confirm before doing the analysis. They asked — that IS the confirmation. Read + report.
