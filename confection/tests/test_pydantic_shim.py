"""Tests for backward compatibility with pydantic BaseModel schemas.

These tests verify that downstream libraries (spaCy, thinc, etc.) can
continue passing pydantic BaseModel subclasses to registry.resolve()
and registry.fill() even though confection no longer depends on pydantic.
"""

import pytest

pydantic = pytest.importorskip("pydantic")

try:
    from pydantic.v1 import (
        BaseModel,
        Field,
        StrictBool,
        StrictFloat,
        StrictInt,
        StrictStr,
        validator,
    )
except ImportError:
    from pydantic import (  # type: ignore
        BaseModel,
        Field,
        StrictBool,
        StrictFloat,
        StrictInt,
        StrictStr,
        validator,
    )

from confection import ConfigValidationError
from confection._validation import Schema, ValidationError, ensure_schema
from confection.tests.util import my_registry


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


# --- Registry integration ---


def test_registry_resolve_with_pydantic_schema():
    class RegSchema(BaseModel):
        hello: StrictInt
        world: StrictInt

        class Config:
            extra = "forbid"

    result = my_registry.resolve(
        {"hello": 1, "world": 2}, schema=RegSchema, validate=True
    )
    assert result == {"hello": 1, "world": 2}


def test_registry_resolve_rejects_bad_type():
    class RegSchema(BaseModel):
        hello: StrictInt
        world: StrictInt

        class Config:
            extra = "forbid"

    with pytest.raises(ConfigValidationError):
        my_registry.resolve(
            {"hello": "bad", "world": 2}, schema=RegSchema, validate=True
        )


def test_registry_fill_with_defaults():
    class FillSchema(BaseModel):
        required: StrictInt
        optional: StrictStr = "default_value"

        class Config:
            extra = "forbid"

    filled = my_registry.fill({"required": 42}, schema=FillSchema)
    assert filled["required"] == 42
    assert filled["optional"] == "default_value"


def test_registry_fill_rejects_extra():
    class StrictSchema(BaseModel):
        x: StrictInt

        class Config:
            extra = "forbid"

    with pytest.raises(ConfigValidationError):
        my_registry.fill({"x": 1, "extra": "bad"}, schema=StrictSchema, validate=True)


# --- Mimics spaCy-style schemas ---


def test_spacy_style_config_schema():
    """Test a schema structure similar to spaCy's ConfigSchemaTraining."""

    class TrainingSchema(BaseModel):
        train_corpus: StrictStr = Field(..., title="Training data path")
        dev_corpus: StrictStr = Field(..., title="Dev data path")
        dropout: StrictFloat = Field(..., title="Dropout rate")
        max_epochs: StrictInt = Field(..., title="Max epochs")
        seed: StrictInt = Field(0, title="Random seed")

        class Config:
            extra = "forbid"
            arbitrary_types_allowed = True

    config = {
        "train_corpus": "corpus/train",
        "dev_corpus": "corpus/dev",
        "dropout": 0.2,
        "max_epochs": 100,
    }
    filled = my_registry.fill(config, schema=TrainingSchema)
    assert filled["seed"] == 0
    assert filled["dropout"] == 0.2

    resolved = my_registry.resolve(config, schema=TrainingSchema, validate=True)
    assert resolved["train_corpus"] == "corpus/train"
    assert resolved["seed"] == 0
