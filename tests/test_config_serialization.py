"""Test Config serialization methods: to_str, from_str, to_bytes, from_bytes,
to_disk, from_disk, copy, merge, interpolate."""

import pytest

from confection import Config
from confection._errors import ConfectionError

# -- to_str / from_str basics (beyond the hypothesis tests) --


def test_to_str_from_str_roundtrip():
    data = {"training": {"lr": 0.001, "epochs": 10}, "model": {"name": "cnn"}}
    config = Config(data)
    restored = Config().from_str(config.to_str(interpolate=False), interpolate=False)
    assert dict(restored["training"]) == {"lr": 0.001, "epochs": 10}
    assert dict(restored["model"]) == {"name": "cnn"}


def test_empty_config():
    config = Config()
    assert config.to_str(interpolate=False) == ""
    restored = Config().from_str("", interpolate=False)
    assert dict(restored) == {}


# -- to_bytes / from_bytes --


def test_to_bytes():
    config = Config({"a": {"x": 1}})
    b = config.to_bytes(interpolate=False)
    assert isinstance(b, bytes)
    assert b"[a]" in b
    assert b"x = 1" in b


def test_from_bytes_roundtrip():
    config = Config({"a": {"x": 1, "y": "hello"}})
    b = config.to_bytes(interpolate=False)
    restored = Config().from_bytes(b, interpolate=False)
    assert restored["a"]["x"] == 1
    assert restored["a"]["y"] == "hello"


def test_from_bytes_with_overrides():
    config = Config({"a": {"x": 1}})
    b = config.to_bytes(interpolate=False)
    restored = Config().from_bytes(b, interpolate=False, overrides={"a.x": 99})
    assert restored["a"]["x"] == 99


# -- to_disk / from_disk --


def test_to_disk_from_disk_roundtrip(tmp_path):
    config = Config({"section": {"key": "value", "num": 42}})
    path = tmp_path / "config.cfg"
    config.to_disk(path, interpolate=False)
    assert path.exists()
    restored = Config().from_disk(path, interpolate=False)
    assert restored["section"]["key"] == "value"
    assert restored["section"]["num"] == 42


def test_to_disk_str_path(tmp_path):
    config = Config({"a": {"x": 1}})
    path = str(tmp_path / "config.cfg")
    config.to_disk(path, interpolate=False)
    restored = Config().from_disk(path, interpolate=False)
    assert restored["a"]["x"] == 1


def test_from_disk_with_overrides(tmp_path):
    config = Config({"a": {"x": 1}})
    path = tmp_path / "config.cfg"
    config.to_disk(path, interpolate=False)
    restored = Config().from_disk(path, interpolate=False, overrides={"a.x": 99})
    assert restored["a"]["x"] == 99


# -- copy --


def test_copy_is_deep():
    config = Config({"a": {"x": [1, 2, 3]}})
    copied = config.copy()
    copied["a"]["x"].append(4)
    assert config["a"]["x"] == [1, 2, 3]
    assert copied["a"]["x"] == [1, 2, 3, 4]


def test_copy_preserves_metadata():
    config = Config({"a": {"x": 1}}, is_interpolated=False, section_order=["a"])
    copied = config.copy()
    assert copied.is_interpolated is False
    assert copied.section_order == ["a"]


# -- interpolate --


def test_interpolate():
    config = Config().from_str(
        "[a]\nx = 1\n\n[b]\ny = ${a.x}",
        interpolate=False,
    )
    assert config["b"]["y"] == "${a.x}"
    interpolated = config.interpolate()
    assert interpolated["b"]["y"] == 1
    # Original should be unchanged
    assert config["b"]["y"] == "${a.x}"


def test_interpolate_returns_new_config():
    config = Config().from_str("[a]\nx = 1", interpolate=False)
    interpolated = config.interpolate()
    assert interpolated is not config


# -- merge --


def test_merge_basic():
    base = Config({"a": {"x": 1, "y": 2}})
    updates = {"a": {"x": 99}}
    merged = base.merge(updates)
    assert merged["a"]["x"] == 99
    assert merged["a"]["y"] == 2


def test_merge_adds_new_keys():
    base = Config({"a": {"x": 1}})
    updates = {"a": {"x": 1, "y": 2}}
    merged = base.merge(updates)
    assert merged["a"]["y"] == 2


def test_merge_remove_extra():
    """remove_extra filters keys from updates that aren't in defaults."""
    base = Config({"a": {"x": 1}})
    updates = {"a": {"x": 99, "extra": "gone"}}
    merged = base.merge(updates, remove_extra=True)
    assert merged["a"]["x"] == 99
    assert "extra" not in merged["a"]


def test_merge_does_not_mutate_original():
    base = Config({"a": {"x": 1}})
    updates = {"a": {"x": 99}}
    base.merge(updates)
    assert base["a"]["x"] == 1


def test_merge_deep():
    base = Config({"a": {"sub": {"x": 1, "y": 2}}})
    updates = {"a": {"sub": {"x": 99}}}
    merged = base.merge(updates)
    assert merged["a"]["sub"]["x"] == 99
    assert merged["a"]["sub"]["y"] == 2


# -- __init__ --


def test_init_from_dict():
    config = Config({"a": {"x": 1}})
    assert config["a"]["x"] == 1
    assert config.is_interpolated is True


def test_init_from_config():
    original = Config({"a": {"x": 1}}, is_interpolated=False, section_order=["a"])
    copy = Config(original)
    assert copy["a"]["x"] == 1
    assert copy.is_interpolated is False
    assert copy.section_order == ["a"]


def test_init_bad_data():
    with pytest.raises(ConfectionError, match="Expected dict"):
        Config([1, 2, 3])


def test_init_override_metadata():
    original = Config({"a": {"x": 1}}, is_interpolated=False)
    copy = Config(original, is_interpolated=True)
    assert copy.is_interpolated is True


# -- overrides with interpolation --

# -- deep_merge_configs edge cases --


def test_merge_leaf_vs_dict():
    """When config has a leaf where defaults has a dict, skip merging."""
    base = Config({"a": {"sub": {"x": 1}}})
    updates = Config({"a": {"sub": "leaf_value"}})
    merged = base.merge(updates)
    assert merged["a"]["sub"] == "leaf_value"


def test_merge_different_promise():
    """Blocks with different @registry functions should not merge."""
    base = Config({"a": {"@cats": "meow.v1", "x": 1}})
    updates = Config({"a": {"@cats": "woof.v1", "y": 2}})
    merged = base.merge(updates)
    # Updates win — base defaults not merged in because different function
    assert merged["a"]["@cats"] == "woof.v1"
    assert merged["a"]["y"] == 2
    assert "x" not in merged["a"]


def test_merge_promise_not_in_defaults():
    """Promise in updates but not in defaults should not merge defaults."""
    base = Config({"a": {"x": 1}})
    updates = Config({"a": {"@cats": "meow.v1", "y": 2}})
    merged = base.merge(updates)
    assert merged["a"]["@cats"] == "meow.v1"
    assert "x" not in merged["a"]


# -- overrides with interpolation --


def test_overrides_with_interpolation():
    result = Config().from_str(
        "[a]\nx = 1\n\n[b]\ny = ${a.x}",
        interpolate=True,
        overrides={"a.x": 42},
    )
    assert result["a"]["x"] == 42
    assert result["b"]["y"] == 42
