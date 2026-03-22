"""Tests for backward compatibility with pydantic BaseModel schemas.

These tests verify that downstream libraries (spaCy, thinc, etc.) can
continue passing pydantic BaseModel subclasses to registry.resolve()
and registry.fill() even though confection no longer depends on pydantic.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "pydantic v1 is not compatible with Python 3.14+", allow_module_level=True
    )

pydantic = pytest.importorskip("pydantic")

try:
    from pydantic.v1 import (
        BaseModel,
        Field,
        StrictFloat,
        StrictInt,
        StrictStr,
        validator,
    )
except ImportError:
    from pydantic import (  # type: ignore
        BaseModel,
        Field,
        StrictFloat,
        StrictInt,
        StrictStr,
        validator,
    )

from confection.validation import Schema, ValidationError, ensure_schema

# --- ensure_schema conversion ---


class SimpleSchema(BaseModel):
    name: StrictStr = Field(..., title="Name")
    value: StrictInt = Field(10, title="Value")

    class Config:
        extra = "forbid"


class InnerSchema(BaseModel):
    x: StrictInt

    class Config:
        extra = "forbid"


class OuterSchema(BaseModel):
    inner: InnerSchema
    label: StrictStr = "default"

    class Config:
        extra = "forbid"


def test_converts_to_schema_subclass():
    converted = ensure_schema(SimpleSchema)
    assert issubclass(converted, Schema)


def test_extracts_fields():
    converted = ensure_schema(SimpleSchema)
    assert "name" in converted.model_fields
    assert "value" in converted.model_fields
    assert converted.model_fields["name"].is_required()
    assert not converted.model_fields["value"].is_required()
    assert converted.model_fields["value"].default == 10


def test_extracts_config():
    converted = ensure_schema(SimpleSchema)
    assert converted.model_config["extra"] == "forbid"


def test_schema_passthrough():
    class MySchema(Schema):
        x: int

    assert ensure_schema(MySchema) is MySchema


def test_caching():
    a = ensure_schema(SimpleSchema)
    b = ensure_schema(SimpleSchema)
    assert a is b


def test_nested_conversion():
    converted = ensure_schema(OuterSchema)
    inner_type = converted.model_fields["inner"].annotation
    assert issubclass(inner_type, Schema)
    assert "x" in inner_type.model_fields


# --- Validation delegates to pydantic ---


def test_validate_correct_data():
    converted = ensure_schema(SimpleSchema)
    result = converted.model_validate({"name": "test", "value": 5})
    assert result.name == "test"
    assert result.value == 5


def test_validate_fills_defaults():
    converted = ensure_schema(SimpleSchema)
    result = converted.model_validate({"name": "test"})
    assert result.value == 10


def test_strict_str_rejects_int():
    converted = ensure_schema(SimpleSchema)
    with pytest.raises(ValidationError):
        converted.model_validate({"name": 123})


def test_extra_fields_rejected():
    converted = ensure_schema(SimpleSchema)
    with pytest.raises(ValidationError):
        converted.model_validate({"name": "x", "extra": 1})


def test_pydantic_validator_works():
    class ValidatedModel(BaseModel):
        name: StrictStr

        class Config:
            extra = "forbid"

        @validator("name")
        def name_must_be_upper(cls, v):
            if v != v.upper():
                raise ValueError("must be uppercase")
            return v

    converted = ensure_schema(ValidatedModel)
    with pytest.raises(ValidationError):
        converted.model_validate({"name": "hello"})
    converted.model_validate({"name": "HELLO"})



# --- Config integration with pydantic schema ---


def test_config_from_str_with_pydantic_schema():
    """Config.from_str works with a pydantic schema for validation and defaults."""
    from confection import Config

    class MyPydanticSchema(BaseModel):
        name: StrictStr
        value: StrictInt = 10

        class Config:
            extra = "forbid"

    class TopSchema(BaseModel):
        section: MyPydanticSchema

    config = Config().from_str("""
[section]
name = "test"
""", interpolate=False, schema=TopSchema)
    assert config["section"]["name"] == "test"
    assert config["section"]["value"] == 10
