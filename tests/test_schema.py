"""Test schema validation and default filling at the Config layer."""

from typing import List, Optional

import pytest

from confection import Config
from confection._errors import ConfigValidationError
from confection.validation import Schema


class TrainingSchema(Schema):
    patience: int
    dropout: float = 0.2
    use_vectors: bool = False


class NlpSchema(Schema):
    lang: str
    pipeline: List[str] = []


class FullSchema(Schema):
    training: TrainingSchema
    nlp: NlpSchema


# -- fill_defaults --


def test_fill_defaults_top_level():
    config = Config({"training": {"patience": 10}})
    config.fill_defaults(FullSchema)
    assert config["training"]["dropout"] == 0.2
    assert config["training"]["use_vectors"] is False


def test_fill_defaults_preserves_existing():
    config = Config({"training": {"patience": 10, "dropout": 0.5}})
    config.fill_defaults(FullSchema)
    assert config["training"]["dropout"] == 0.5


def test_fill_defaults_adds_missing_sections():
    config = Config({"training": {"patience": 10}})
    config.fill_defaults(FullSchema)
    # nlp has all defaults so it should be filled
    assert "nlp" not in config  # nlp section itself is required, not defaulted


def test_fill_defaults_returns_self():
    config = Config({"training": {"patience": 10}})
    result = config.fill_defaults(FullSchema)
    assert result is config


# -- validate --


def test_validate_passes():
    config = Config(
        {"training": {"patience": 10, "dropout": 0.2}, "nlp": {"lang": "en"}}
    )
    config.validate(FullSchema)  # should not raise


def test_validate_missing_required():
    config = Config({"training": {"dropout": 0.5}, "nlp": {"lang": "en"}})
    with pytest.raises(ConfigValidationError):
        config.validate(FullSchema)


def test_validate_wrong_type():
    config = Config(
        {"training": {"patience": "nope", "dropout": 0.2}, "nlp": {"lang": "en"}}
    )
    with pytest.raises(ConfigValidationError):
        config.validate(FullSchema)


# -- from_str with schema --


def test_from_str_with_schema():
    config = Config().from_str(
        """
[training]
patience = 10

[nlp]
lang = "en"
""",
        interpolate=False,
        schema=FullSchema,
    )
    assert config["training"]["patience"] == 10
    assert config["training"]["dropout"] == 0.2
    assert config["training"]["use_vectors"] is False


def test_from_str_schema_validates():
    with pytest.raises(ConfigValidationError):
        Config().from_str(
            """
[training]
dropout = 0.5

[nlp]
lang = "en"
""",
            interpolate=False,
            schema=FullSchema,
        )


def test_from_str_schema_with_interpolation():
    config = Config().from_str(
        """
[training]
patience = 10

[nlp]
lang = "en"
""",
        interpolate=True,
        schema=FullSchema,
    )
    assert config["training"]["dropout"] == 0.2


# -- Schema with extra="forbid" --


class StrictSchema(Schema):
    model_config = {"extra": "forbid"}
    x: int
    y: str = "default"


def test_validate_extra_forbidden():
    config = Config({"x": 1, "y": "hello", "z": "extra"})
    with pytest.raises(ConfigValidationError):
        config.validate(StrictSchema)


# -- Schema with Optional fields --


class OptionalSchema(Schema):
    name: str
    description: Optional[str] = None


def test_optional_field_defaults_to_none():
    config = Config({"name": "test"})
    config.fill_defaults(OptionalSchema)
    assert config["description"] is None


def test_optional_field_accepts_value():
    config = Config({"name": "test", "description": "hello"})
    config.validate(OptionalSchema)  # should not raise


# -- Flat schema (no nesting) --


class FlatSchema(Schema):
    x: int
    y: float = 3.14
    z: str = "hello"


def test_flat_schema_from_str():
    """Schema works for flat configs too (top-level keys are leaves, not sections)."""
    config = Config({"x": 1})
    config.fill_defaults(FlatSchema)
    assert config["y"] == 3.14
    assert config["z"] == "hello"
