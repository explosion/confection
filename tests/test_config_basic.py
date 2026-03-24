"""Test basic config sections.

* No variable interpolation
* No promises

Just basic structure and JSON-encoded values.
"""

from configparser import InterpolationDepthError

import pytest
from hypothesis import HealthCheck, given, settings

from confection import Config
from tests.strategies import (
    circular_interpolated_config,
    config_dicts,
    interpolated_config,
    json_config_dicts,
    serialize_with_inline,
)


@given(config_dicts)
def test_roundtrip(data):
    """Config.from_str(config.to_str()) should reproduce the original dict."""
    config = Config(data)
    serialized = config.to_str(interpolate=False)
    restored = Config().from_str(serialized, interpolate=False)
    assert dict_equal(restored, data)


@given(json_config_dicts())
def test_json_leaves_parse(pair):
    """Config strings with inline JSON values (lists, dicts) should parse
    to the same nested dict structure."""
    data, inline_paths = pair
    config_str = serialize_with_inline(data, inline_paths)
    restored = Config().from_str(config_str, interpolate=False)
    assert dict_equal(restored, data)


@given(interpolated_config())
def test_variable_interpolation(pair):
    """Config strings with ${section.key} variable references should resolve
    to the referenced values after interpolation."""
    config_str, expected = pair
    restored = Config().from_str(config_str, interpolate=True)
    assert dict_equal(restored, expected)


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(circular_interpolated_config())
def test_circular_interpolation_raises(config_str):
    """Circular variable references should raise an error."""
    with pytest.raises((InterpolationDepthError, Exception)):
        Config().from_str(config_str, interpolate=True)


def test_star_sections_parse():
    """[section.*.name] creates a dict under the "*" key."""
    result = Config().from_str(
        """
[section]

[section.*.first]
x = 1

[section.*.second]
x = 2
""",
        interpolate=False,
    )
    assert result["section"]["*"] == {"first": {"x": 1}, "second": {"x": 2}}


def test_star_sections_roundtrip():
    """Configs with * sections roundtrip through to_str/from_str."""
    original = Config().from_str(
        """
[section]

[section.*.a]
x = 1
y = "hello"

[section.*.b]
x = 2
y = "world"
""",
        interpolate=False,
    )
    serialized = original.to_str(interpolate=False)
    restored = Config().from_str(serialized, interpolate=False)
    assert dict_equal(restored, original)


def test_star_sections_nested():
    """* sections can appear at different levels of nesting."""
    result = Config().from_str(
        """
[top]

[top.*.item]
val = 1

[top.*.item.sub]
val = 2
""",
        interpolate=False,
    )
    assert result["top"]["*"]["item"]["val"] == 1
    assert result["top"]["*"]["item"]["sub"] == {"val": 2}


def test_star_with_interpolation():
    """Variable interpolation works across * sections."""
    result = Config().from_str(
        """
[settings]
lr = 0.001

[models]

[models.*.first]
learning_rate = ${settings.lr}
""",
        interpolate=True,
    )
    assert result["models"]["*"]["first"]["learning_rate"] == 0.001


@pytest.mark.parametrize(
    "text,expected_value,expected_type",
    [
        ("True", True, bool),
        ("False", False, bool),
        ("true", True, bool),
        ("false", False, bool),
        ("None", None, type(None)),
        ("null", None, type(None)),
    ],
)
def test_python_style_literals(text, expected_value, expected_type):
    """Python-style True/False/None must be parsed the same as JSON-style
    true/false/null. Many existing spaCy configs use the capitalised form."""
    cfg = f"[section]\nval = {text}\n"
    config = Config().from_str(cfg)
    result = config["section"]["val"]
    assert result is expected_value, (
        f"Expected {expected_value!r} ({expected_type.__name__}), "
        f"got {result!r} ({type(result).__name__})"
    )
    assert type(result) is expected_type


@pytest.mark.parametrize("text", ['"True"', '"False"', '"None"'])
def test_quoted_python_literals_stay_strings(text):
    """JSON-quoted strings like '"True"' must remain strings, not be coerced
    to booleans/None."""
    cfg = f"[section]\nval = {text}\n"
    config = Config().from_str(cfg)
    result = config["section"]["val"]
    assert isinstance(result, str), (
        f"Expected str, got {type(result).__name__}: {result!r}"
    )


def dict_equal(a, b) -> bool:
    """Recursively compare two nested dicts, treating empty dicts as equal."""
    if type(a) is not type(b) and not (isinstance(a, dict) and isinstance(b, dict)):
        return a == b
    if isinstance(a, dict):
        all_keys = set(a) | set(b)
        for key in all_keys:
            av = a.get(key, {})
            bv = b.get(key, {})
            if not dict_equal(av, bv):
                return False
        return True
    return a == b
