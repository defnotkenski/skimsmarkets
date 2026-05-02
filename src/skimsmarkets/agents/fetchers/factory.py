"""Provider factory — `build_provider(name, config)` returns a configured
`FetcherProvider`. Centralises the SDK-import boundary so a Grok-only run
never needs `google-genai` to be importable, and vice versa.
"""

from __future__ import annotations

from skimsmarkets import config as cfg
from skimsmarkets.agents.fetchers.base import FetcherProvider


def build_provider(name: str, config: cfg.Config) -> FetcherProvider:
    """Instantiate the named provider with credentials from `config`.

    Raises a clear error on unknown names (validated against
    `cfg.FETCHER_PROVIDERS` which is the single source of truth) and on
    missing per-provider keys (already validated in `Config.from_env`,
    but re-checked here so direct construction paths fail loud too).
    """
    if name == "grok":
        if not config.xai_api_key:
            raise RuntimeError(
                "Cannot build GrokProvider: xai_api_key is unset. "
                "Set XAI_API_KEY in the environment."
            )
        # Local import keeps `xai_sdk` off the import path of Gemini-only
        # runs (and vice versa for `google-genai` below).
        from skimsmarkets.agents.fetchers.grok import GrokProvider

        return GrokProvider(api_key=config.xai_api_key)
    if name == "gemini":
        if not config.google_api_key:
            raise RuntimeError(
                "Cannot build GeminiProvider: google_api_key is unset. "
                "Set GOOGLE_API_KEY in the environment."
            )
        from skimsmarkets.agents.fetchers.gemini import GeminiProvider

        return GeminiProvider(api_key=config.google_api_key)
    raise RuntimeError(
        f"Unknown fetcher provider {name!r}. "
        f"Valid: {', '.join(cfg.FETCHER_PROVIDERS)}."
    )
