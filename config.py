"""Config loading for Hermes Herald plugin.

Reads profile endpoint config from config.yaml under the ``hermes_herald``
key. Supports env var interpolation in ``api_key`` values via ``${VAR_NAME}``.
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

# Cache the loaded config so we don't re-read config.yaml on every tool call.
# The gateway restarts to pick up config changes, so a process-lifetime cache
# is fine.
_config_cache: Optional[dict] = None


def _resolve_hermes_home() -> Path:
    """Find the active HERMES_HOME directory."""
    return Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))


def _load_config() -> dict:
    """Load the hermes_herald section from config.yaml.

    Returns an empty dict if the section is missing or the file is unreadable.
    The result is cached for the process lifetime.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    hermes_home = _resolve_hermes_home()
    config_path = hermes_home / "config.yaml"
    try:
        import yaml
        with open(config_path) as f:
            full_config = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("hermes-herald: cannot load %s: %s", config_path, e)
        _config_cache = {}
        return _config_cache

    _config_cache = full_config.get("hermes_herald", {}) or {}
    if not isinstance(_config_cache, dict):
        _config_cache = {}
    return _config_cache


def _resolve_env_var(value: str) -> str:
    """Resolve a ``${VAR_NAME}`` string from os.environ.

    If the value doesn't match the env var pattern, return it as-is.
    If the env var is not set, return an empty string (the API server will
    reject the auth, which produces a clear error).
    """
    m = _ENV_VAR_RE.match(value.strip())
    if m:
        return os.environ.get(m.group(1), "")
    return value


def get_profile_config(profile: str) -> Optional[Dict[str, Any]]:
    """Return the endpoint config for a named profile.

    Returns ``None`` if the profile is not configured. The returned dict
    has keys: ``url``, ``api_key``, and optionally ``model``. The model value
    is an exact target ``model_routes`` alias accepted by ``dispatch_agent``
    and ``dispatch_chat``; it is not an arbitrary model name.
    """
    cfg = _load_config()
    profiles = cfg.get("profiles", {})
    entry = profiles.get(profile)
    if entry is None:
        return None

    resolved = dict(entry)
    # Resolve env var in api_key
    raw_key = entry.get("api_key", "")
    if isinstance(raw_key, str):
        resolved["api_key"] = _resolve_env_var(raw_key)
    return resolved


def list_profiles() -> list:
    """Return the list of configured profile names."""
    cfg = _load_config()
    return sorted((cfg.get("profiles") or {}).keys())


def get_active_profile_name() -> str:
    """Return the current Hermes profile identity for topology/audit records."""
    explicit = _load_config().get("origin_name")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    try:
        from hermes_cli.profiles import get_active_profile_name as _active_profile

        return str(_active_profile() or "default")
    except Exception:
        return "default"


def get_route_capabilities(profile: str) -> list[str]:
    """Return explicitly allowed outbound operations for one configured target.

    Missing or malformed grants fail closed. Unknown capability names are
    ignored so they cannot accidentally grant a route Herald does not implement.
    This is caller-side policy, not target-side caller authentication.
    """
    cfg = _load_config()
    entry = (cfg.get("profiles") or {}).get(profile) or {}
    raw = entry.get("capabilities")
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    allowed = {str(item).strip().lower() for item in raw}
    return [name for name in ("dispatch", "chat") if name in allowed]


def allow_self_routing() -> bool:
    """Whether the active profile may call its own explicitly named route."""
    return _load_config().get("allow_self") is True


def describe_topology() -> dict:
    """Return credential-free outbound topology for the active origin."""
    origin = get_active_profile_name()
    outbound = []
    for profile in list_profiles():
        outbound.append({
            "profile": profile,
            "capabilities": get_route_capabilities(profile),
            "self": profile == origin,
        })
    return {
        "origin_profile": origin,
        "allow_self": allow_self_routing(),
        "declared_scope": "current_origin_only",
        "configured_outbound": outbound,
    }


def get_default_profile() -> Optional[str]:
    """Return the configured default profile, if any."""
    cfg = _load_config()
    return cfg.get("default_profile")


def get_state_file_path() -> Path:
    """Return the path to the run-state JSON file."""
    cfg = _load_config()
    raw = cfg.get("state_file")
    if not raw:
        return _resolve_hermes_home() / "hermes-herald-runs.json"
    return Path(os.path.expandvars(os.path.expanduser(str(raw))))


def get_ledger_file_path() -> Path:
    """Return the durable SQLite dispatch-ledger path.

    Point several same-filesystem origin profiles at one absolute path to get a
    combined observed graph. The default remains profile-local.
    """
    cfg = _load_config()
    raw = cfg.get("ledger_file")
    if not raw:
        return _resolve_hermes_home() / "hermes-herald.db"
    return Path(os.path.expandvars(os.path.expanduser(str(raw))))


def get_chat_timeout() -> float:
    """Return the default dispatch_chat activity-stall timeout in seconds.

    Configured via ``hermes_herald.chat_timeout`` in config.yaml.
    Defaults to 600 (10 minutes) — agentic tasks with tool calls can take a while.
    """
    cfg = _load_config()
    raw = cfg.get("chat_timeout", 600)
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 600.0
    if val <= 0:
        return 600.0
    return val
