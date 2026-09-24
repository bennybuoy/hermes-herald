#!/usr/bin/env python3
"""Fail fast when public installation metadata drifts from the plugin identity."""
import re
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

    # Version consistency: the manifest is the single source of truth; the skill
    # frontmatter and the README badge must not drift from it. v1.2.0 shipped with
    # 1.2.0 metadata and 1.1.0 artwork because nothing compared these surfaces.
    version = str(manifest.get("version", "")).strip()
    assert version, "plugin.yaml is missing its version"

    skill_text = (ROOT / "skills" / "agent-dispatch" / "SKILL.md").read_text(encoding="utf-8")
    # SKILL.md is YAML frontmatter (first --- … --- block) followed by a markdown body,
    # so it is a multi-document stream: parse the frontmatter block only.
    skill_version = ""
    front = re.match(r"\A---\n(.*?)\n---\n", skill_text, re.DOTALL)
    assert front, "bundled skill is missing its YAML frontmatter block"
    front_fields = yaml.safe_load(front.group(1)) or {}
    skill_version = str(front_fields.get("version", "")).strip()
    assert skill_version == version, (
        f"version drift: plugin.yaml says {version} but the bundled skill says "
        f"{skill_version or '(none)'} — bump both together"
    )

    badge_match = re.search(r"img\.shields\.io/badge/version-([0-9A-Za-z.]+)-", readme)
    assert badge_match, "README has no img.shields.io version badge to check"
    badge_version = badge_match.group(1)
    assert badge_version == version, (
        f"version drift: plugin.yaml says {version} but the README badge says {badge_version}"
    )

    # Artwork is binary; the script cannot read it. Fail only when the files the
    # release card/hero are built from are missing, and remind the human to eyeball
    # the visible version strings.
    for asset in ("assets/hero-banner.png", "assets/release-x-card.png"):
        assert (ROOT / asset).is_file(), f"missing release artwork: {asset}"

    print(
        f"release contract: OK (hermes-herald, {len(advertised)} tools, "
        f"version {version} consistent across plugin.yaml / skill / README badge; "
        "opt-in documented; artwork present — eyeball its version text yourself)"
    )


if __name__ == "__main__":
    main()
