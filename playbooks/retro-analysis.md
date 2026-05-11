# Retro analysis playbook

Use when the operator asks for a retro pass — anything like "do a retro analysis", "look at recent results", "what's retro saying", "analyze settled events".

This playbook is the conversational equivalent of `skims retro --step analyze`. The operator runs `--step calibrate` for the deterministic cuts (no LLM cost) and asks for analysis instead of paying for the in-pipeline LLM call. Don't suggest `--step analyze` — that's the path the operator deliberately skips.

## What to read

1. Calibrate output — either the operator's recent stdout or run `uv run skims retro --step calibrate` yourself.
2. `logs/runs/*.jsonl` — per-event detail (lens reports, defensibility, predicted/market probabilities, negative_edge flag). Recent files first; older ones contextually.
3. `logs/runs/*.resolutions.jsonl` — settled outcomes (winning side, predicted_correct).
4. `CLAUDE.md` — load-bearing invariants any prompt-edit recommendation must respect.

## Discipline

- **Anti-hindsight.** Don't explain individual losses. Anything plausible can be constructed for any single loss. Only report patterns that recur across MULTIPLE losses AND are clearly less frequent in wins.
- **No base-rate facts.** "Many losses were tennis" isn't a finding when the slate is mostly tennis.
- **Cross-check claims against data.** Don't take any LLM framing for granted. Pull the actual rows; verify counts, means, ranges before asserting.
- **Small-N honesty.** At n<10 losses, label everything "directional only." Push back on "fix this prompt" recommendations regardless of how confident the pattern looks.
- **Concrete, not generic.** Each recommendation is a sentence or two of specific guidance the operator can translate into a prompt edit. "The system overweights ranking" is an observation, not a recommendation; "tighten DIRECTOR_SPORT_HINTS to require X when Y" is the right shape. Vague platitudes get cut.
- **Honor CLAUDE.md invariants** in any recommendation. Reject (with reasoning) any proposed edit that:
  - Re-ties confidence to probability magnitude (confidence is contingency robustness, not lopsidedness)
  - Softens "bid/ask is a prior" toward "bid/ask is a ceiling"
  - Reformats canonical names
  - Pipes one lens's output into another's context
  - Reintroduces edge / buy / pass framing

## Output shape

Mirror `logs/retro/*.report.md` structure:

1. **Headline.** "N settled / M unique markets, K correct (X%)" — pull from calibrate output.
2. **Hit-rate cuts.** From calibrate (case bucket / confidence / favorite-vs-underdog / negative-edge). Already in the calibrate output; copy or summarize.
3. **Per-sport patterns** (only sports with settled events). Three fields, mirroring the `RetroFindings` schema the in-pipeline analyze step would have emitted:
   - **`recurring_patterns`** — 3-7 short bullets, each naming one loss-overrepresented feature with its rough magnitude (e.g. "edge_over_market mean +0.015 in losses vs +0.043 in wins").
   - **`lens_underperformance`** — for each registered lens with a clear failure attribution, one short sentence on the recurring failure mode. Omit the field entirely when no clear attribution emerges from the data. Use exact registered lens names (see "Per-sport notes" below).
   - **`prompt_recommendations`** — 2-5 concrete edits, each targeting one specific surface:
     - `DIRECTOR_SPORT_HINTS` (in `src/skimsmarkets/agents/sport_hints.py`)
     - `LensSpec.reasoner_sport_hint` — name the lens
     - `LensSet.director_system_tail` (in `src/skimsmarkets/agents/sports/<sport>/prompts.py`)
4. **Pushbacks.** Explicitly call out which (if any) recommendations should NOT be adopted, and why (sample size, invariant violation, conflates concepts, etc.). At small N this section is usually the longest.

## Per-sport notes

### Tennis

- **Registered lens names** — use these EXACTLY when attributing failures; don't invent or abbreviate:
  - `tennis_form_and_surface`
  - `tennis_matchup_and_clutch`
  - `tennis_conditions_and_context`

- **Divergence columns are the highest-yield signal.** The `divergence_*` fields on the prediction row capture how a player's actual match performance differed from their pre-match career baseline (`actual - baseline`; positive = overperformed). **Large negative divergence on the predicted side, paired with a loss, is the high-signal pattern**: the player underperformed their baseline AND nothing in pre-match flagged it. When tennis losses cluster these collapses, look for whether ANY pre-match feature (surface, confidence tier, market favorite/underdog, defensibility score) correlates — that's a real lens-attribution candidate. Wins without divergence data don't disprove the pattern; wins WITH positive divergence on the predicted side reinforce it.

## What NOT to do

- Don't run `skims retro --step analyze` (that's the LLM-API path the operator wants to skip).
- Don't suggest building Phase A/A.5/B/C lineage infrastructure unless the operator explicitly revisits the parked plan (see `memory/project_retro_lineage_parked.md`).
- Don't recommend prompt edits at n<5 losses without flagging "these are not actionable yet — directional only."
- Don't ask the operator to confirm before doing the analysis. They asked, that's the confirmation. Read + report.
