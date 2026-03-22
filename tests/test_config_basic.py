"""Test basic config sections.

* No variable interpolation
* No promises

Just basic structure and JSON-encoded values.
"""
from hypothesis import given

from confection import Config
import pytest
from configparser import InterpolationDepthError
from hypothesis import settings, HealthCheck
from tests.strategies import (
    config_dicts, json_config_dicts, serialize_with_inline,
    interpolated_config, circular_interpolated_config,
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
