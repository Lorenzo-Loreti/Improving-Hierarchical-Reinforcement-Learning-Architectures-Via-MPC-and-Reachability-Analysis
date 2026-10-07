"""Tests of the YAML loader of the configuration files (code rule C2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hrlmpc.utils.strict_yaml import load_yaml

MERGED = """
base: &base {a: 1, b: 2}
derived:
  <<: *base
  b: 3
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_files_load_as_with_the_safe_loader(tmp_path: Path) -> None:
    text = "physics: {dt: 0.1, a_max: 2.5}\nobstacles:\n  - {x: [4.0, 5.0]}\nflag: true\nlr: 3.0e-4\n"
    assert load_yaml(_write(tmp_path, text)) == yaml.safe_load(text)


def test_merge_keys_are_expanded_and_may_be_overridden(tmp_path: Path) -> None:
    assert load_yaml(_write(tmp_path, MERGED)) == yaml.safe_load(MERGED) == {
        "base": {"a": 1, "b": 2},
        "derived": {"a": 1, "b": 3},
    }


@pytest.mark.parametrize("text", ["a: 1\na: 2\n", "outer:\n  x: 1\n  y: 2\n  x: 3\n", "<<: {a: 1}\nb: 1\nb: 2\n"])
def test_a_repeated_key_is_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        load_yaml(_write(tmp_path, text))


def test_only_safe_tags_are_accepted(tmp_path: Path) -> None:
    with pytest.raises(yaml.YAMLError):
        load_yaml(_write(tmp_path, "x: !!python/object/apply:os.getcwd []\n"))
