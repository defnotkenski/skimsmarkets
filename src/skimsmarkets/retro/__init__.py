"""Retro / self-improvement layer.

Reads `logs/runs/<run_id>.jsonl` prediction logs, resolves outcomes
against gamma, computes deterministic hit-rate cuts, and (for tennis)
fetches post-match box scores so a single batched LLM call can find
patterns across wins vs losses. Output is operator reading material —
no auto-editing of prompts.

Three steps; see CLI `skims retro --step {resolve,calibrate,analyze,all}`
and `~/.claude/plans/cached-tinkering-pudding.md` for design context.
"""
