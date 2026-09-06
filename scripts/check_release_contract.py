#!/usr/bin/env python3
"""Fail fast when public installation metadata drifts from the plugin identity."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    init_source = (ROOT / "__init__.py").read_text(encoding="utf-8")

    assert manifest["name"] == "hermes-herald"
    assert "hermes plugins install bennybuoy/hermes-herald --enable" in readme
    assert "hermes plugins enable hermes-herald" in readme
    assert "Upgrading from Agent Dispatch 1.x" in readme
    assert "identity from `agent-dispatch` to" in readme
    assert "- hermes-herald" in readme

    advertised = manifest["provides_tools"]
    assert isinstance(advertised, list) and advertised
    assert len(advertised) == len(set(advertised))
    assert "llm_direct" in advertised
    for tool_name in advertised:
        assert f'("{tool_name}",' in init_source
    count_line = f"hermes-herald: registered {len(advertised)} tools"
    assert count_line in readme, f"README missing {count_line!r}"
    heading = f"## The {len(advertised)} tools"
    assert heading in readme, f"README missing {heading!r}"

    print(
        f"release contract: OK (hermes-herald, {len(advertised)} tools, "
        "opt-in documented)"
    )


if __name__ == "__main__":
    main()
