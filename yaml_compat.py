"""Load YAML using the parser supplied by the host Hermes installation."""

from __future__ import annotations


def safe_load(stream):
    """Use Hermes' YAML API, or the declared legacy PyYAML dependency."""
    try:
        from hermes_yaml import safe_load as host_safe_load
    except ModuleNotFoundError as exc:
        if exc.name != "hermes_yaml":
            raise
        import yaml

        host_safe_load = yaml.safe_load
    return host_safe_load(stream)
