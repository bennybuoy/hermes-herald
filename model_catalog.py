"""Local and remote model catalog discovery and route verification for Hermes Herald.

Separates provider authority (fail-closed: the selected provider must be configured
or authenticated on the target/local profile) from model-ID membership (open for
unregistered/new/custom models on a configured provider, while surfacing catalog
hits and multi-route ambiguities explicitly).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from . import config as cfg
from .yaml_compat import safe_load

logger = logging.getLogger(__name__)

_VIRTUAL_OR_AMBIGUOUS_PROVIDERS = frozenset({"auto", "moa", "main"})
_SELF_HOSTED_PROVIDERS = frozenset({
    "custom",
    "ollama",
    "lmstudio",
    "vllm",
    "local",
    "llamacpp",
    "llama.cpp",
    "llama-cpp",
})
_PARTIAL_CATALOG_PROVIDERS = frozenset({"openrouter"})
_OPENROUTER_ROUTING_SUFFIXES = (
    ":free",
    ":nitro",
    ":floor",
    ":exacto",
    ":extended",
    ":thinking",
    ":online",
)
_DIGIT_DOT_RE = re.compile(r"(\d)\.(\d)")
_MAX_MODEL_ID_LEN = 200


def provider_allows_whitespace_models(
    provider: str,
    *,
    is_user_defined: bool = False,
) -> bool:
    """True when a provider may legitimately serve model IDs containing spaces."""
    if is_user_defined:
        return True
    slug = str(provider or "").strip().lower()
    if not slug:
        return False
    return slug.startswith("custom:") or slug in _SELF_HOSTED_PROVIDERS


def validate_model_id(
    raw: Any,
    *,
    allow_whitespace: bool = False,
) -> tuple[str, Optional[str]]:
    """Validate a model ID string without rewriting or restricting unregistered slugs."""
    if not isinstance(raw, str):
        return "", "'model' must be a non-empty string."
    model = raw.strip()
    if not model:
        return "", "'model' must be a non-empty string."
    if len(model) > _MAX_MODEL_ID_LEN:
        return "", f"'model' exceeds the maximum length of {_MAX_MODEL_ID_LEN} characters."
    if re.search(r"[\r\n\x00]", model):
        return "", "'model' must not contain control characters."
    if not allow_whitespace and any(ch.isspace() for ch in model):
        return "", (
            f"Model '{model}' contains whitespace, which is only allowed for "
            "custom or self-hosted endpoints."
        )
    return model, None


def bare_model_name(model: str) -> str:
    """Return the lowercased model slug after any ``vendor/`` prefix."""
    return str(model or "").strip().lower().rsplit("/", 1)[-1]


def normalized_model_family_key(model: str) -> str:
    """Canonical grouping key for detecting multi-route versions of the same model.

    Used ONLY for discovery hints and disambiguation — wire model IDs are never
    rewritten. Strips ``vendor/`` prefixes, strips OpenRouter routing modifiers
    such as ``:free``/``:nitro``, and normalizes digit separators (``4.6`` vs
    ``4-6``) so direct Anthropic and OpenRouter entries group together.
    """
    bare = bare_model_name(model)
    for suffix in _OPENROUTER_ROUTING_SUFFIXES:
        if bare.endswith(suffix):
            bare = bare[: -len(suffix)]
            break
    return _DIGIT_DOT_RE.sub(r"\1-\2", bare)


def query_matches_model(needle: str, *, provider: str = "", model: str = "", alias: str = "") -> bool:
    """Case-insensitive substring match across provider, model, alias, and bare family key."""
    if not needle:
        return True
    norm_needle = _DIGIT_DOT_RE.sub(r"\1-\2", needle)
    candidates = [
        provider.lower(),
        model.lower(),
        alias.lower(),
        bare_model_name(model),
        normalized_model_family_key(model),
    ]
    return any(needle in text or norm_needle in text for text in candidates if text)


def _provider_catalog_note(
    slug: str,
    *,
    is_user_defined: bool,
    model_count: int,
) -> tuple[bool, str]:
    """Return ``(catalog_partial, note)`` explaining catalog completeness for a provider."""
    low = slug.strip().lower()
    if low in _PARTIAL_CATALOG_PROVIDERS:
        return (
            True,
            "Curated subset only; any valid OpenRouter org/model slug (including "
            ":free/:nitro variants or @preset/*) may be passed with provider='openrouter'.",
        )
    if is_user_defined and model_count == 0:
        return (
            True,
            "Custom endpoint returned no /models listing; pass any model ID served "
            "by this endpoint with this provider.",
        )
    if is_user_defined:
        return (
            True,
            "Custom endpoint; unregistered or aliased models served by this endpoint "
            "may also be passed with this provider.",
        )
    return False, ""


def _provider_match_keys(row: dict) -> set[str]:
    """Lowercased slugs/aliases accepted for a configured provider row."""
    keys: set[str] = set()
    slug = str(row.get("provider") or row.get("slug") or "").strip().lower()
    if slug:
        keys.add(slug)
        if slug.startswith("custom:"):
            suffix = slug.split(":", 1)[1].strip()
            if suffix:
                keys.add(suffix)
        elif row.get("is_user_defined"):
            keys.add(f"custom:{slug}")
    for alias in row.get("aliases") or ():
        if isinstance(alias, str) and alias.strip():
            keys.add(alias.strip().lower())
    return keys


def match_configured_provider(
    requested_provider: str,
    providers: Iterable[dict],
) -> Optional[dict]:
    """Match *requested_provider* against configured/authenticated provider rows.

    Returns the matched provider dict (whose ``provider`` field is the canonical
    slug) or ``None``. Virtual/ambiguous routing names are always rejected.
    """
    token = str(requested_provider or "").strip().lower()
    if not token or token in _VIRTUAL_OR_AMBIGUOUS_PROVIDERS:
        return None
    exact: Optional[dict] = None
    aliased: Optional[dict] = None
    for row in providers:
        if not isinstance(row, dict):
            continue
        slug = str(row.get("provider") or row.get("slug") or "").strip()
        if not slug or slug.lower() in _VIRTUAL_OR_AMBIGUOUS_PROVIDERS:
            continue
        if slug.lower() == token:
            exact = row
            break
        if aliased is None and token in _provider_match_keys(row):
            aliased = row
    return exact or aliased


def build_multi_route_models(providers: Iterable[dict]) -> list[dict]:
    """Return models reachable via more than one configured provider.

    Surfaces overlapping models (including direct vs OpenRouter versions such as
    ``claude-sonnet-4-6`` on ``anthropic`` and ``anthropic/claude-sonnet-4.6``
    on ``openrouter``) so callers explicitly pin ``provider=``.
    """
    groups: dict[str, list[dict[str, str]]] = {}
    seen_pairs: set[tuple[str, str, str]] = set()
    for row in providers:
        if not isinstance(row, dict):
            continue
        provider = str(row.get("provider") or row.get("slug") or "").strip()
        if not provider or provider.lower() in _VIRTUAL_OR_AMBIGUOUS_PROVIDERS:
            continue
        for model in row.get("models") or ():
            if not isinstance(model, str) or not model.strip():
                continue
            model_id = model.strip()
            key = normalized_model_family_key(model_id)
            if not key:
                continue
            dedupe = (key, provider.lower(), model_id)
            if dedupe in seen_pairs:
                continue
            seen_pairs.add(dedupe)
            groups.setdefault(key, []).append({
                "provider": provider,
                "model": model_id,
            })

    multi: list[dict] = []
    for key, routes in sorted(groups.items()):
        unique_providers = {route["provider"].lower() for route in routes}
        if len(unique_providers) > 1:
            routes_sorted = sorted(
                routes,
                key=lambda item: (item["provider"].lower(), item["model"]),
            )
            multi.append({"model_key": key, "routes": routes_sorted})
    return multi


def _normalize_provider_rows(raw_rows: Iterable[dict]) -> list[dict]:
    """Normalize raw inventory/options provider rows into Herald's public shape."""
    providers: list[dict] = []
    seen_slugs: set[str] = set()
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        slug = str(row.get("slug") or row.get("provider") or "").strip()
        if not slug or slug.lower() in _VIRTUAL_OR_AMBIGUOUS_PROVIDERS:
            continue
        if row.get("authenticated") is False:
            continue
        raw_models = row.get("models") or []
        advertised_models = sorted({
            str(model).strip()
            for model in raw_models
            if isinstance(model, str) and model.strip()
        })
        if row.get("source") == "canonical" and not advertised_models:
            continue
        low = slug.lower()
        if low in seen_slugs:
            continue
        seen_slugs.add(low)
        is_user_defined = bool(row.get("is_user_defined")) or low.startswith("custom:") or low == "custom"
        aliases = sorted({
            str(a).strip()
            for a in (row.get("aliases") or ())
            if isinstance(a, str) and a.strip() and str(a).strip().lower() != low
        })
        catalog_partial, note = _provider_catalog_note(
            slug,
            is_user_defined=is_user_defined,
            model_count=len(advertised_models),
        )
        entry: Dict[str, Any] = {
            "provider": slug,
            "models": advertised_models,
            "model_count": len(advertised_models),
            "is_user_defined": is_user_defined,
            "catalog_partial": catalog_partial,
        }
        name = str(row.get("name") or "").strip()
        if name:
            entry["name"] = name
        if row.get("is_current") is True:
            entry["is_current"] = True
        if aliases:
            entry["aliases"] = aliases
        if note:
            entry["note"] = note
        providers.append(entry)
    providers.sort(key=lambda item: item["provider"].lower())
    return providers


# ---------------------------------------------------------------------------
# Local profile inventory
# ---------------------------------------------------------------------------


def _load_local_via_core_inventory(
    *,
    current_provider: str = "",
    current_model: str = "",
    current_base_url: str = "",
    probe_custom_providers: bool = True,
) -> dict:
    """Build local profile inventory via Hermes's explicit-only inventory builder."""
    from hermes_cli.inventory import ConfigContext, build_models_payload, load_picker_context

    configured = load_picker_context()
    live_provider = (current_provider or configured.current_provider or "").strip()
    live_model = (current_model or configured.current_model or "").strip()
    live_base_url = (current_base_url or configured.current_base_url or "").strip()
    user_providers = getattr(configured, "user_providers", {}) or {}
    custom_providers = getattr(configured, "custom_providers", []) or []

    context = ConfigContext(
        current_provider=live_provider,
        # Do not force the active model into the catalog list; catalog membership
        # is reported separately so unregistered defaults stay transparent.
        current_model="",
        current_base_url=live_base_url,
        user_providers=user_providers,
        custom_providers=custom_providers,
        excluded_providers=getattr(configured, "excluded_providers", []) or [],
    )
    payload = build_models_payload(
        context,
        explicit_only=True,
        include_unconfigured=False,
        for_picker=False,
        probe_custom_providers=probe_custom_providers,
        probe_current_custom_provider=probe_custom_providers,
    )
    providers = _normalize_provider_rows(payload.get("providers") or [])

    default_provider_row = match_configured_provider(live_provider, providers)
    configured_default = (
        {"provider": live_provider, "model": live_model}
        if default_provider_row is not None and live_model
        else None
    )
    default_in_catalog = bool(
        configured_default is not None
        and live_model in (default_provider_row.get("models") or ())
    )

    return {
        "configured_default": configured_default,
        "configured_default_in_catalog": default_in_catalog,
        "providers": providers,
        "user_providers": user_providers,
        "custom_providers": custom_providers,
        "discovery_source": "hermes_inventory",
    }


def _read_declared_models(raw: Any) -> list[str]:
    """Extract declared model IDs from a YAML ``models`` list, dict, or string."""
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                mid = str(item.get("id") or item.get("name") or item.get("model") or "").strip()
                if mid:
                    out.append(mid)
        return out
    if isinstance(raw, dict):
        return [str(k).strip() for k in raw.keys() if str(k).strip()]
    return []


def _load_local_via_public_apis(
    *,
    current_provider: str = "",
    current_model: str = "",
    current_base_url: str = "",
) -> dict:
    """Public-API-only fallback when ``hermes_cli.inventory`` is unavailable.

    Uses only modules listed in Hermes's published ``plugin_api.json``
    (``hermes_cli.config``, ``providers``, ``agent.secret_scope``,
    ``agent.credential_pool``, ``hermes_constants``) plus stdlib / ``yaml_compat``.
    """
    from agent.secret_scope import get_secret
    from hermes_constants import get_hermes_home

    full_cfg: dict = {}
    try:
        from hermes_cli.config import load_config_readonly
        loaded = load_config_readonly()
        if isinstance(loaded, dict):
            full_cfg = loaded
    except Exception:
        try:
            config_path = get_hermes_home() / "config.yaml"
            with open(config_path, encoding="utf-8") as handle:
                loaded = safe_load(handle)
            if isinstance(loaded, dict):
                full_cfg = loaded
        except Exception:
            full_cfg = {}

    model_cfg = full_cfg.get("model")
    if isinstance(model_cfg, dict):
        cfg_model = str(model_cfg.get("default") or model_cfg.get("name") or "").strip()
        cfg_provider = str(model_cfg.get("provider") or "").strip()
        cfg_base_url = str(model_cfg.get("base_url") or "").strip()
    else:
        cfg_model = str(model_cfg or "").strip()
        cfg_provider = ""
        cfg_base_url = ""

    live_provider = (current_provider or cfg_provider).strip()
    live_model = (current_model or cfg_model).strip()
    live_base_url = (current_base_url or cfg_base_url).strip()

    user_providers = full_cfg.get("providers") if isinstance(full_cfg.get("providers"), dict) else {}
    custom_providers = (
        full_cfg.get("custom_providers")
        if isinstance(full_cfg.get("custom_providers"), list)
        else []
    )
    model_catalog_cfg = full_cfg.get("model_catalog")
    excluded_raw = (
        model_catalog_cfg.get("excluded_providers")
        if isinstance(model_catalog_cfg, dict)
        else None
    )
    excluded = {
        str(p).strip().lower()
        for p in (excluded_raw if isinstance(excluded_raw, list) else [])
        if isinstance(p, str) and p.strip()
    }

    raw_rows: list[dict] = []
    seen: set[str] = set()

    # Registered providers with configured env keys, pools, or active selection.
    try:
        from providers import list_providers
        profiles = list(list_providers())
    except Exception:
        profiles = []

    pool_loader = None
    try:
        from agent.credential_pool import load_pool
        pool_loader = load_pool
    except Exception:
        pool_loader = None

    for profile in profiles:
        slug = str(getattr(profile, "name", "") or "").strip()
        if not slug or slug.lower() in excluded or slug.lower() in _VIRTUAL_OR_AMBIGUOUS_PROVIDERS:
            continue
        aliases = tuple(
            str(a).strip() for a in (getattr(profile, "aliases", ()) or ())
            if isinstance(a, str) and a.strip()
        )
        all_names = {slug.lower(), *(a.lower() for a in aliases)}
        if all_names & excluded:
            continue
        env_vars = tuple(getattr(profile, "env_vars", ()) or ())
        has_env = any(bool(str(get_secret(var, "") or "").strip()) for var in env_vars if isinstance(var, str))
        has_pool = False
        if pool_loader is not None and not has_env:
            try:
                pool = pool_loader(slug)
                has_pool = bool(pool is not None and pool.has_credentials())
            except Exception:
                has_pool = False
        is_current = live_provider.lower() in all_names
        user_entry = user_providers.get(slug) if isinstance(user_providers, dict) else None
        if not (has_env or has_pool or is_current or isinstance(user_entry, dict)):
            continue
        if isinstance(user_entry, dict) and user_entry.get("enabled") is False:
            continue
        models = list(
            getattr(profile, "static_models", ()) or getattr(profile, "fallback_models", ()) or ()
        )
        if isinstance(user_entry, dict):
            models = [*_read_declared_models(user_entry.get("models")), *models]
        raw_rows.append({
            "slug": slug,
            "name": str(getattr(profile, "display_name", "") or slug),
            "is_current": is_current,
            "is_user_defined": False,
            "authenticated": True,
            "models": models,
            "aliases": list(aliases),
        })
        seen.update(all_names)

    # User-defined providers: dict entries.
    for name, entry in user_providers.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            continue
        if entry.get("enabled") is False:
            continue
        slug = name.strip()
        if not slug or slug.lower() in seen or slug.lower() in excluded:
            continue
        models = _read_declared_models(entry.get("models"))
        for key in ("default_model", "model"):
            val = str(entry.get(key) or "").strip()
            if val and val not in models:
                models.append(val)
        raw_rows.append({
            "slug": slug,
            "name": str(entry.get("name") or slug).strip(),
            "is_current": live_provider.lower() in {slug.lower(), f"custom:{slug.lower()}"},
            "is_user_defined": True,
            "authenticated": True,
            "models": models,
            "aliases": [f"custom:{slug}"],
        })
        seen.add(slug.lower())

    # Legacy custom_providers: list entries.
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        raw_name = str(entry.get("name") or "").strip()
        if not raw_name:
            continue
        slug_body = re.sub(r"[^a-z0-9]+", "-", raw_name.lower()).strip("-") or raw_name.lower()
        slug = f"custom:{slug_body}"
        if slug.lower() in seen or slug_body in seen or slug.lower() in excluded:
            continue
        models = _read_declared_models(entry.get("models"))
        val = str(entry.get("model") or "").strip()
        if val and val not in models:
            models.append(val)
        raw_rows.append({
            "slug": slug,
            "name": raw_name,
            "is_current": live_provider.lower() in {slug.lower(), slug_body, raw_name.lower()},
            "is_user_defined": True,
            "authenticated": True,
            "models": models,
            "aliases": [slug_body, raw_name],
        })
        seen.add(slug.lower())

    # Active provider fallback when configured in model: block (e.g. bare custom or OAuth).
    if live_provider and live_provider.lower() not in seen and live_provider.lower() not in _VIRTUAL_OR_AMBIGUOUS_PROVIDERS:
        raw_rows.append({
            "slug": live_provider,
            "name": live_provider,
            "is_current": True,
            "is_user_defined": live_provider.lower() == "custom" or live_provider.lower().startswith("custom:") or bool(live_base_url),
            "authenticated": True,
            "models": [],
        })

    providers = _normalize_provider_rows(raw_rows)
    default_provider_row = match_configured_provider(live_provider, providers)
    configured_default = (
        {"provider": live_provider, "model": live_model}
        if default_provider_row is not None and live_model
        else None
    )
    default_in_catalog = bool(
        configured_default is not None
        and live_model in (default_provider_row.get("models") or ())
    )
    return {
        "configured_default": configured_default,
        "configured_default_in_catalog": default_in_catalog,
        "providers": providers,
        "user_providers": user_providers,
        "custom_providers": custom_providers,
        "discovery_source": "public_config_fallback",
    }


def load_local_model_inventory(
    *,
    current_provider: str = "",
    current_model: str = "",
    current_base_url: str = "",
    probe_custom_providers: bool = True,
) -> dict:
    """Return configured local provider/model inventory for this profile.

    Prefers Hermes's explicit-only inventory builder when present and falls back
    to public-API config/provider discovery if ``hermes_cli.inventory`` is
    unavailable.
    """
    try:
        return _load_local_via_core_inventory(
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
            probe_custom_providers=probe_custom_providers,
        )
    except (ImportError, ModuleNotFoundError):
        logger.debug("hermes_cli.inventory unavailable; using public-API local model discovery")
        return _load_local_via_public_apis(
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
        )


def summarize_llm_direct_endpoints() -> list[dict]:
    """Return credential-free metadata for configured ``llm_direct`` endpoints."""
    if not cfg.llm_direct_enabled():
        return []
    section = cfg._load_config().get("llm_direct")
    if not isinstance(section, dict):
        return []
    endpoints = section.get("endpoints")
    if not isinstance(endpoints, dict):
        return []
    default_endpoint = cfg.get_default_direct_endpoint()
    summary: list[dict] = []
    for name in sorted(endpoints.keys()):
        entry = endpoints.get(name)
        if not isinstance(name, str) or not name.strip() or not isinstance(entry, dict):
            continue
        allowed = entry.get("allowed_models")
        allowed_list = (
            [m for m in allowed if isinstance(m, str) and m.strip()]
            if isinstance(allowed, list)
            else None
        )
        summary.append({
            "endpoint": name.strip(),
            "is_default": name.strip() == default_endpoint,
            "default_model": str(entry.get("default_model") or "").strip(),
            "allowed_models": allowed_list,
            "allows_unregistered_models": allowed_list is None,
        })
    return summary


def verify_local_llm_route(
    inventory: dict,
    provider_override: str,
    model_override: str,
) -> tuple[dict, Optional[str]]:
    """Validate an explicit ``(provider, model)`` override for local ``llm_call``.

    Fails closed when the provider is virtual, ambiguous, or unconfigured on
    this profile. Permits unregistered models on a configured provider so
    OpenRouter slugs, custom proxies, and newly released snapshots work through
    Hermes's ``ctx.llm.complete()`` trust policy.
    """
    providers = [
        row for row in (inventory.get("providers") or [])
        if isinstance(row, dict)
    ]
    configured_names = sorted({
        str(row.get("provider") or "").strip()
        for row in providers
        if str(row.get("provider") or "").strip()
    })
    avail_text = ", ".join(configured_names) if configured_names else "(none)"

    low_provider = provider_override.strip().lower()
    if low_provider in _VIRTUAL_OR_AMBIGUOUS_PROVIDERS:
        return {}, (
            f"Provider '{provider_override}' is a virtual/ambiguous routing mode "
            f"and cannot be passed to llm_call. Configured providers: {avail_text}. "
            "Call list_profile_models without a profile."
        )

    matched = match_configured_provider(provider_override, providers)
    if matched is None:
        openai_hint = ""
        if low_provider == "openai":
            openai_hint = (
                " Hermes's bare 'openai' alias is ambiguous (use 'openai-codex', "
                "'openai-api', or 'openrouter' if configured)."
            )
        return {}, (
            f"Route provider='{provider_override}', model='{model_override}' is "
            f"not configured for this profile.{openai_hint} Configured providers: "
            f"{avail_text}. Call list_profile_models without a profile and copy "
            "a configured provider slug."
        )

    canonical_provider = str(matched.get("provider") or provider_override).strip()
    allow_ws = provider_allows_whitespace_models(
        canonical_provider,
        is_user_defined=bool(matched.get("is_user_defined")),
    )
    clean_model, model_err = validate_model_id(model_override, allow_whitespace=allow_ws)
    if model_err:
        return {}, model_err

    advertised_models = {
        str(m).strip()
        for m in (matched.get("models") or ())
        if isinstance(m, str) and m.strip()
    }
    in_catalog = clean_model in advertised_models
    return {
        "provider": canonical_provider,
        "model": clean_model,
        "in_catalog": in_catalog,
        "unregistered_model": not in_catalog,
    }, None


# ---------------------------------------------------------------------------
# Remote profile catalog & route verification
# ---------------------------------------------------------------------------


def entry_route_provider(entry: dict) -> str:
    """Credential-free provider pin from a ``/v1/models`` row, if advertised."""
    for key in ("provider", "owned_by"):
        raw = entry.get(key)
        if isinstance(raw, str):
            value = raw.strip()
            if value and value.lower() not in {"hermes", "hermes-agent"}:
                return value
    return ""


def parse_model_options_payload(response: Any) -> Optional[dict]:
    """Parse Hermes's ``GET /api/model/options`` payload into normalized routes."""
    if not isinstance(response, dict) or not isinstance(response.get("providers"), list):
        return None
    providers = _normalize_provider_rows(response["providers"])
    default_provider = str(response.get("provider") or "").strip()
    default_model = str(response.get("model") or "").strip()
    target_default = (
        {"provider": default_provider, "model": default_model}
        if default_provider or default_model
        else None
    )
    routes = sorted(
        (
            {"provider": row["provider"], "model": model}
            for row in providers
            for model in row["models"]
        ),
        key=lambda item: (item["provider"].lower(), item["model"]),
    )
    return {
        "target_default": target_default,
        "configured_providers": providers,
        "available_routes": routes,
        "route_count": len(routes),
        "multi_route_models": build_multi_route_models(providers),
    }


def _fetch_remote_v1_models(
    profile: str,
    pcfg: dict,
    get_json: Callable,
    *,
    error_context: str,
) -> tuple[dict, Optional[str]]:
    """Fetch and parse authenticated ``GET /v1/models`` from a target profile."""
    models_url = f"{pcfg['url']}/v1/models"
    try:
        response = get_json(models_url, pcfg["api_key"], timeout=10.0)
    except RuntimeError as e:
        msg = str(e)
        if "401" in msg or "403" in msg:
            return {}, (
                f"{error_context}: authentication to {models_url} failed. "
                f"Check the profile api_key. ({msg})"
            )
        if "Cannot reach" in msg:
            return {}, (
                f"{error_context}: the target API server is unreachable. ({msg})"
            )
        return {}, f"{error_context}. ({msg})"
    except Exception as e:
        return {}, (
            f"{error_context}: invalid response from {models_url}. "
            f"({type(e).__name__})"
        )

    entries = response.get("data") if isinstance(response, dict) else None
    if not isinstance(entries, list):
        return {}, (
            f"{error_context}: the target does not expose a compatible "
            "authenticated /v1/models route listing."
        )

    primary = next(
        (
            entry for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("id"), str)
            and not entry.get("parent")
        ),
        None,
    )
    route_entries = [
        entry for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and entry["id"].strip()
        and isinstance(entry.get("parent"), str)
        and entry["parent"].strip()
    ]
    dispatchable = sorted(
        (
            {
                "alias": entry["id"].strip(),
                "pass_as": entry["id"].strip(),
                "resolved_model": entry["root"].strip(),
                "provider": entry_route_provider(entry),
                "provider_pinned": bool(entry_route_provider(entry)),
            }
            for entry in route_entries
            if isinstance(entry.get("root"), str) and entry["root"].strip()
        ),
        key=lambda item: item["alias"],
    )
    return {
        "entries": entries,
        "route_entries": route_entries,
        "primary_id": primary["id"].strip() if primary else "",
        "dispatchable": dispatchable,
    }, None


def _fetch_remote_model_options(pcfg: dict, get_json: Callable) -> Optional[dict]:
    """Best-effort fetch of authenticated ``GET /api/model/options`` from a target."""
    options_url = f"{pcfg['url']}/api/model/options?include_unconfigured=false"
    try:
        response = get_json(options_url, pcfg["api_key"], timeout=10.0)
    except Exception:
        return None
    return parse_model_options_payload(response)


def fetch_remote_model_catalog(
    profile: str,
    pcfg: dict,
    get_json: Callable,
) -> tuple[dict, Optional[str]]:
    """Fetch combined ``/v1/models`` aliases + ``/api/model/options`` provider catalog."""
    v1_data, err = _fetch_remote_v1_models(
        profile,
        pcfg,
        get_json,
        error_context=f"Cannot list dispatchable models for {profile}",
    )
    if err:
        return {}, err

    options = _fetch_remote_model_options(pcfg, get_json)
    return {
        **v1_data,
        "options_available": options is not None,
        "target_default": options["target_default"] if options else None,
        "configured_providers": options["configured_providers"] if options else [],
        "available_routes": options["available_routes"] if options else [],
        "route_count": options["route_count"] if options else 0,
        "multi_route_models": options["multi_route_models"] if options else [],
    }, None


def _find_model_provider_matches(
    requested_model: str,
    providers: Iterable[dict],
) -> list[dict[str, str]]:
    """Find all configured providers on a target that advertise *requested_model*.

    Checks every configured provider (preferring exact ID within a provider, then
    normalized family match so direct Anthropic ``claude-sonnet-4-6`` and
    OpenRouter ``anthropic/claude-sonnet-4.6`` are BOTH surfaced as ambiguous
    candidates when the caller omits ``provider=``).
    """
    matches: list[dict[str, str]] = []
    target_lower = requested_model.strip().lower()
    target_key = normalized_model_family_key(requested_model)
    for row in providers:
        if not isinstance(row, dict):
            continue
        provider = str(row.get("provider") or "").strip()
        if not provider:
            continue
        models = [m for m in (row.get("models") or ()) if isinstance(m, str) and m.strip()]
        exact = next((m for m in models if m.lower() == target_lower), None)
        if exact:
            matches.append({"provider": provider, "model": exact})
            continue
        if target_key:
            fam = next(
                (m for m in models if normalized_model_family_key(m) == target_key),
                None,
            )
            if fam:
                matches.append({"provider": provider, "model": fam})
    return matches


def verify_remote_model_route(
    profile: str,
    pcfg: dict,
    requested_model: str,
    requested_provider: str = "",
    *,
    get_json: Callable,
) -> tuple[dict, Optional[str]]:
    """Verify a remote ``dispatch_agent`` / ``dispatch_chat`` model/provider request.

    Supports both:
    1. Exact ``model_routes`` aliases advertised by ``GET /v1/models`` (existing contract).
    2. Direct ``provider`` + ``model`` pairs verified against the target's authenticated
       providers from ``GET /api/model/options`` — including unregistered/new/custom models
       and disambiguating multi-route models (such as direct vs OpenRouter).

    Bare unaliased ``model`` without ``provider`` fails closed with actionable
    disambiguation hints so ``/v1/chat/completions`` never silently falls back to
    default and ``/v1/runs`` never routes a slug to the wrong provider.
    """
    requested_model = (requested_model or "").strip()
    requested_provider = (requested_provider or "").strip()

    # Provider-only override (no model alias/ID): validate against target options when available.
    if not requested_model:
        if not requested_provider:
            return {}, None
        if requested_provider.lower() in _VIRTUAL_OR_AMBIGUOUS_PROVIDERS:
            return {}, (
                f"Provider '{requested_provider}' is a virtual/ambiguous routing mode "
                f"and cannot be sent to {profile}. No task was started."
            )
        options = _fetch_remote_model_options(pcfg, get_json)
        if options is None:
            return {
                "requested_model": "",
                "resolved_model": "",
                "resolution_source": "target_provider_override",
                "provider": requested_provider,
                "provider_pinned": True,
                "unregistered_model": False,
            }, None
        matched_provider = match_configured_provider(
            requested_provider,
            options["configured_providers"],
        )
        if matched_provider is None:
            configured_names = ", ".join(
                row["provider"] for row in options["configured_providers"]
            ) or "(none)"
            return {}, (
                f"Provider '{requested_provider}' is not authenticated on {profile}. "
                f"Authenticated providers: {configured_names}. Call "
                f"list_profile_models(profile={profile!r}). No task was started."
            )
        return {
            "requested_model": "",
            "resolved_model": "",
            "resolution_source": "target_provider_override",
            "provider": matched_provider["provider"],
            "provider_pinned": True,
            "unregistered_model": False,
        }, None

    v1_data, err = _fetch_remote_v1_models(
        profile,
        pcfg,
        get_json,
        error_context=f"Cannot verify model route '{requested_model}' for {profile}",
    )
    if err:
        return {}, f"{err} No task was started."

    route_entries = v1_data["route_entries"]
    matched_alias = next(
        (entry for entry in route_entries if entry["id"].strip() == requested_model),
        None,
    )

    # Path 1: exact target model_routes alias (works even when /api/model/options is absent).
    if matched_alias is not None:
        if requested_provider and requested_provider.lower() in _VIRTUAL_OR_AMBIGUOUS_PROVIDERS:
            return {}, (
                f"Provider '{requested_provider}' is a virtual/ambiguous routing mode "
                f"and cannot be sent to {profile}. No task was started."
            )
        resolved_model = matched_alias.get("root")
        if isinstance(resolved_model, str) and resolved_model.strip():
            pinned_provider = entry_route_provider(matched_alias)
            if (
                requested_provider
                and pinned_provider
                and requested_provider.lower() != pinned_provider.lower()
            ):
                return {}, (
                    f"Model route '{requested_model}' on {profile} is pinned to "
                    f"provider '{pinned_provider}'. Remove provider= or pass "
                    f"provider='{pinned_provider}'. No task was started."
                )
            return {
                "requested_model": requested_model,
                "resolved_model": resolved_model.strip(),
                "resolution_source": "target_model_routes",
                "provider": requested_provider or pinned_provider,
                "provider_pinned": bool(pinned_provider),
                "unregistered_model": False,
            }, None
        return {}, (
            f"Cannot verify model route '{requested_model}' for {profile}: "
            "the target advertised the alias without a resolved root model. "
            "No task was started."
        )

    entries = v1_data["entries"]
    advertised_primary = any(
        isinstance(entry, dict)
        and entry.get("id") == requested_model
        and not entry.get("parent")
        for entry in entries
    )
    available_aliases = sorted(entry["id"].strip() for entry in route_entries)
    available_aliases_text = ", ".join(available_aliases) if available_aliases else "(none)"

    # Path 2: direct provider + model route (catalog OR unregistered model).
    if requested_provider:
        if advertised_primary:
            return {}, (
                f"Model override '{requested_model}' is only {profile}'s virtual "
                "API identity (advertised_primary.model), not an upstream model ID. "
                "Pass a real model ID with provider=, or omit model= to use the "
                "target default. No task was started."
            )
        if requested_provider.lower() in _VIRTUAL_OR_AMBIGUOUS_PROVIDERS:
            return {}, (
                f"Provider '{requested_provider}' is a virtual/ambiguous routing mode "
                f"and cannot be sent to {profile}. Call list_profile_models("
                f"profile={profile!r}) to inspect configured providers. No task was started."
            )
        options = _fetch_remote_model_options(pcfg, get_json)
        if options is None:
            return {}, (
                f"Cannot verify direct provider route provider='{requested_provider}', "
                f"model='{requested_model}' for {profile}: target does not expose "
                f"/api/model/options. pass_as_model aliases: {available_aliases_text}. "
                "Use an exact alias from pass_as_model or upgrade the target API server. "
                "No task was started."
            )
        configured_providers = options["configured_providers"]
        matched_provider = match_configured_provider(
            requested_provider,
            configured_providers,
        )
        if matched_provider is None:
            configured_names = ", ".join(
                row["provider"] for row in configured_providers
            ) or "(none)"
            openai_hint = ""
            if requested_provider.lower() == "openai":
                openai_hint = (
                    " Hermes's bare 'openai' alias is ambiguous; use an exact "
                    "configured provider slug such as 'openai-codex', 'openai-api', "
                    "or 'openrouter'."
                )
            return {}, (
                f"Provider '{requested_provider}' is not authenticated on {profile}."
                f"{openai_hint} Authenticated providers: {configured_names}. "
                f"pass_as_model aliases: {available_aliases_text}. Call "
                f"list_profile_models(profile={profile!r}). No task was started."
            )
        canonical_provider = matched_provider["provider"]
        allow_ws = provider_allows_whitespace_models(
            canonical_provider,
            is_user_defined=bool(matched_provider.get("is_user_defined")),
        )
        clean_model, model_err = validate_model_id(
            requested_model,
            allow_whitespace=allow_ws,
        )
        if model_err:
            return {}, f"{model_err} No task was started."
        in_catalog = clean_model in set(matched_provider.get("models") or ())
        return {
            "requested_model": clean_model,
            "resolved_model": clean_model,
            "resolution_source": (
                "target_provider_catalog"
                if in_catalog
                else "target_provider_unregistered"
            ),
            "provider": canonical_provider,
            "provider_pinned": True,
            "unregistered_model": not in_catalog,
        }, None

    # Path 3: bare model without provider, not a model_routes alias -> fail closed
    # with smart disambiguation hints from /api/model/options when available.
    if advertised_primary:
        return {}, (
            f"Model override '{requested_model}' is not supported for {profile}: "
            "it is only the target's advertised default/API identity, not a "
            f"configured model_routes alias. pass_as_model: {available_aliases_text}. "
            f"Call list_profile_models(profile={profile!r}). Omit model to use "
            "the target default. No task was started."
        )

    options = _fetch_remote_model_options(pcfg, get_json)
    if options is not None:
        configured_providers = options["configured_providers"]
        provider_matches = _find_model_provider_matches(
            requested_model,
            configured_providers,
        )
        configured_names = ", ".join(
            row["provider"] for row in configured_providers
        ) or "(none)"
        if len(provider_matches) == 1:
            match = provider_matches[0]
            return {}, (
                f"Model '{requested_model}' is not a model_routes alias on {profile}, "
                f"so passing model= alone is refused to prevent misrouting. It is "
                f"available via provider '{match['provider']}' (model '{match['model']}'). "
                f"Retry with provider='{match['provider']}', model='{match['model']}'. "
                "No task was started."
            )
        if len(provider_matches) > 1:
            choices = ", ".join(
                f"provider='{m['provider']}' model='{m['model']}'"
                for m in provider_matches
            )
            return {}, (
                f"Model '{requested_model}' is ambiguous on {profile} because it "
                f"exists on multiple configured providers: {choices}. Pass provider= "
                "explicitly alongside model= to pin the exact route. No task was started."
            )
        return {}, (
            f"Model override '{requested_model}' is not an exact configured "
            f"model_routes alias on {profile} (pass_as_model: {available_aliases_text}). "
            "To run a catalog or unregistered model directly without an alias, pass "
            f"provider= explicitly alongside model=. Authenticated providers on "
            f"{profile}: {configured_names}. Call list_profile_models("
            f"profile={profile!r}). Omit model to use the target default. "
            "No task was started."
        )

    return {}, (
        f"Model override '{requested_model}' is not supported for {profile}: "
        f"it is not an exact configured model_routes alias. pass_as_model: "
        f"{available_aliases_text}. Call list_profile_models(profile={profile!r}) "
        "and copy one exact alias from pass_as_model into model=, or pass both "
        "provider= and model= on targets supporting /api/model/options. "
        "Omit model to use the target default. No task was started."
    )
