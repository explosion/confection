"""Test error cases in config parsing and validation."""
import pytest

from confection import Config
from confection._errors import ConfigValidationError


def test_malformed_config_string():
    """Malformed config syntax should raise ConfigValidationError."""
    with pytest.raises(ConfigValidationError, match="formatted correctly"):
        Config().from_str("[[invalid", interpolate=False)


def test_malformed_missing_section_header():
    """Values without any section header are malformed."""
    with pytest.raises(ConfigValidationError, match="formatted correctly"):
        Config().from_str("key = value", interpolate=False)


def test_top_level_values_without_section():
    """Top-level values that leak into DEFAULT should be caught."""
    # configparser treats values before any section header as defaults.
    # This is caught by read_string as a MissingSectionHeaderError -> ParsingError.
    with pytest.raises(ConfigValidationError):
        Config().from_str("x = 1\n[section]\ny = 2", interpolate=False)


def test_missing_parent_section():
    """Dotted section names require all parent sections to exist."""
    with pytest.raises(ConfigValidationError, match="not defined"):
        Config().from_str("[a.b.c]\nx = 1", interpolate=False)


def test_missing_intermediate_section():
    """[a] and [a.b.c] without [a.b] should fail."""
    with pytest.raises(ConfigValidationError, match="not defined"):
        Config().from_str("[a]\nx = 1\n\n[a.b.c]\ny = 2", interpolate=False)


def test_key_subsection_conflict():
    """A key that conflicts with a subsection name should be caught."""
    with pytest.raises(ConfigValidationError, match="conflicting"):
        Config().from_str("[a]\nb = 1\n\n[a.b]\nc = 2", interpolate=False)


def test_override_without_dot():
    """Overrides must have a dotted path (section.key)."""
    with pytest.raises(ConfigValidationError, match="overrid"):
        Config().from_str("[a]\nx = 1", interpolate=False, overrides={"x": 2})


def test_override_nonexistent_section():
    """Overrides for nonexistent sections should fail."""
    with pytest.raises(ConfigValidationError, match="overrid"):
        Config().from_str("[a]\nx = 1", interpolate=False, overrides={"b.x": 2})


def test_override_applies():
    """Valid overrides should replace values."""
    result = Config().from_str("[a]\nx = 1", interpolate=False, overrides={"a.x": 99})
    assert result["a"]["x"] == 99


def test_interpolation_missing_variable():
    """Referencing a nonexistent variable should raise."""
    with pytest.raises(Exception):
        Config().from_str("[a]\nx = ${b.y}", interpolate=True)


def test_section_ref_in_string():
    """Referencing a whole section inside a string should raise."""
    with pytest.raises(ConfigValidationError, match="Can't reference whole sections"):
        Config().from_str(
            "[defaults]\nlr = 0.001\n\n[a]\nx = \"hello ${defaults}\"",
            interpolate=True,
        )


def test_section_reference_resolves():
    """${section} references should resolve to the section dict."""
    result = Config().from_str(
        "[defaults]\nlr = 0.001\n\n[a]\nsettings = ${defaults}",
        interpolate=True,
    )
    assert result["a"]["settings"] == {"lr": 0.001}


def test_uninterpolated_variable_preserved():
    """With interpolate=False, variable references should stay as strings."""
    result = Config().from_str("[a]\nx = 1\n\n[b]\ny = ${a.x}", interpolate=False)
    assert result["b"]["y"] == "${a.x}"
