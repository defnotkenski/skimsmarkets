"""Deterministic post-rank filters for `skims execute`.

Each row from `logs/runs/<run_id>.jsonl` passes through these
predicates before the matcher / order paths run. The filter set is
intentionally narrow — confidence tier, defensibility cutoff,
negative-edge exclusion, sport gate — and corresponds 1:1 to CLI
flags. No LLM here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Literal

from skimsmarkets.retro.models import PredictionRow


def filter_rows(
    rows: Iterable[PredictionRow],
    *,
    confidence: list[Literal["low", "medium", "high"]] | None = None,
    min_defensibility: float | None = None,
    no_negative_edge: bool = False,
    sports: list[str] | None = None,
    risk_buckets: list[str] | None = None,
    min_market_implied_prob: float | None = None,
) -> Iterator[PredictionRow]:
    """Yield rows that pass every active filter.

    Filters compose as AND. A filter is "inactive" when its parameter
    is None / empty / False; inactive filters pass every row through.

    - `confidence`: keep rows whose `confidence` is in the set.
    - `min_defensibility`: drop rows below the cutoff. `None` score
      (judge failure) → DROP, since we can't verify defensibility.
      This is a SAFETY filter; absence of evidence isn't evidence
      of safety.
    - `no_negative_edge`: drop rows where `negative_edge is True` OR
      `is None`. None means we couldn't compute the flag (market
      implied missing) — treat as unsafe.
    - `sports`: keep rows whose `sport_type` is in the set.
    - `risk_buckets`: keep rows whose `risk_bucket` is in the set.
      Rows with `risk_bucket=None` (Unrated / classifier failure) →
      DROP when the filter is active; same safety stance as
      `min_defensibility`.
    - `min_market_implied_prob`: drop rows whose
      `polymarket_implied_probability` (market's prob for the
      predicted winner) is below the cutoff. Default 0.50 catches
      directional disagreement — market and model picking opposite
      sides. None implied prob → DROP, same safety stance.
    """
    conf_set: frozenset[str] | None = (
        frozenset(confidence) if confidence else None
    )
    sport_set: frozenset[str] | None = (
        frozenset(s.lower() for s in sports) if sports else None
    )
    bucket_set: frozenset[str] | None = (
        frozenset(risk_buckets) if risk_buckets else None
    )
    for row in rows:
        if conf_set is not None and row.confidence not in conf_set:
            continue
        if min_defensibility is not None:
            if (
                row.defensibility_score is None
                or row.defensibility_score < min_defensibility
            ):
                continue
        if no_negative_edge:
            # `negative_edge` is True | False | None. Drop True and None.
            if row.negative_edge is not False:
                continue
        if sport_set is not None:
            sport = (row.sport_type or "").lower()
            if sport not in sport_set:
                continue
        if bucket_set is not None:
            if row.risk_bucket is None or row.risk_bucket not in bucket_set:
                continue
        if min_market_implied_prob is not None:
            if (
                row.polymarket_implied_probability is None
                or row.polymarket_implied_probability < min_market_implied_prob
            ):
                continue
        yield row
