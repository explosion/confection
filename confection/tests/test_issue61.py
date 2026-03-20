"""Test for issue #61: Top-level section references in interpolation.

Users want to reference a top-level section (e.g. ${logger}) from within
another section, instead of having to nest it (${logger.logger}).
Currently this causes an InterpolationDepthError because configparser's
ExtendedInterpolation only resolves ${section.key} not ${section}.
"""

import pytest

from confection import Config


def test_top_level_section_reference():
    """Referencing a top-level section via ${section} should work."""
    config_str = """
[logger]
name = "test_logger"

[training]
batch_size = 32
logger = ${logger}
"""
    # This currently raises InterpolationDepthError
    cfg = Config().from_str(config_str)
    assert cfg["training"]["logger"]["name"] == "test_logger"


def test_nested_section_reference_works():
    """The nested workaround ${section.key} should work as before."""
    config_str = """
[logger]
name = "test_logger"

[training]
batch_size = 32
logger_name = ${logger.name}
"""
    cfg = Config().from_str(config_str)
    assert cfg["training"]["logger_name"] == "test_logger"
