"""YAML loading for the configuration files: safe, and strict about repeated keys (code rule C2).

PyYAML keeps the last of two equal keys in a mapping without a word, so a
value written twice in a configuration file would silently override the
first. The loader here refuses such files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class _StrictLoader(yaml.SafeLoader):  # type: ignore[misc, unused-ignore]  # PyYAML may lack type stubs
    """A safe loader that rejects mappings with a repeated key."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":  # '<<: *anchor', expanded by SafeLoader below
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                repeated = key in seen
            except TypeError:  # an unhashable key; PyYAML reports it below
                continue
            if repeated:
                raise ValueError(f"duplicate key {key!r} {key_node.start_mark}".strip())
            seen.add(key)
        result: dict[Any, Any] = super().construct_mapping(node, deep=deep)
        return result


def load_yaml(path: str | Path) -> Any:
    """Parse a YAML file with :class:`yaml.SafeLoader`'s rules, rejecting repeated keys.

    Raises:
        ValueError: If a mapping repeats a key.
    """
    with Path(path).open(encoding="utf-8") as f:
        return yaml.load(f, Loader=_StrictLoader)  # noqa: S506 - a SafeLoader subclass
