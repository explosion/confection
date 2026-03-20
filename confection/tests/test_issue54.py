"""Test for issue #54: Error when pydantic validator coerces str to dict.

A custom type with __get_validators__ that converts a string to a dict
should not crash _update_from_parsed.
"""

import sys

import pytest

pydantic = pytest.importorskip("pydantic")
if sys.version_info >= (3, 14):
    pytest.skip(
        "pydantic v1 is not compatible with Python 3.14+", allow_module_level=True
    )

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel  # type: ignore

from confection import Config, ConfigValidationError, registry


class CastStrAsDict:
    @classmethod
    def validate(cls, value):
        if isinstance(value, str):
            return {value: True}
        return value

    @classmethod
    def __get_validators__(cls):
        yield cls.validate


class SectionSchema(BaseModel):
    value: CastStrAsDict


class MainSchema(BaseModel):
    section: SectionSchema


def test_fill_with_validator_str_to_dict():
    """fill() should not crash when a validator coerces str -> dict."""
    cfg = Config().from_str(
        """
[section]
value = "ok"
"""
    )
    # This should not raise TypeError: string indices must be integers
    try:
        result = registry.fill(cfg, schema=MainSchema)
        # If it succeeds, the section should still be present
        assert "section" in result
    except ConfigValidationError:
        # A validation error is acceptable (the type is unusual),
        # but TypeError is not.
        pass
