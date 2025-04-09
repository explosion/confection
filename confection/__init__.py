import re
from typing import (
    Any,
    Dict,
    Union,
)

import srsly

from pydantic import BaseModel
from pydantic.fields import Field as ModelField

from .util import SimpleFrozenDict, SimpleFrozenList  # noqa: F401
from ._errors import ConfigValidationError
from ._registry import registry
from ._config import (
    Config,
    VARIABLE_RE,
    ARGS_FIELD,
    ARGS_FIELD_ALIAS,
    RESERVED_FIELDS,
    SECTION_PREFIX,
    JSON_EXCEPTIONS,
    try_load_json,
    try_dump_json,
)


def alias_generator(name: str) -> str:
    """Generate field aliases in promise schema."""
    # Underscore fields are not allowed in model, so use alias
    if name == ARGS_FIELD_ALIAS:
        return ARGS_FIELD
    # Auto-alias fields that shadow base model attributes
    if name in RESERVED_FIELDS:
        return RESERVED_FIELDS[name]
    return name


def copy_model_field(field: ModelField, type_: Any) -> ModelField:
    """Copy a model field and assign a new type, e.g. to accept an Any type
    even though the original value is typed differently.
    """
    return ModelField(
        name=field.name,
        type_=type_,
        class_validators=field.class_validators,
        model_config=field.model_config,
        default=field.default,
        default_factory=field.default_factory,
        required=field.required,
        alias=field.alias,
    )


class EmptySchema(BaseModel):
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True


__all__ = [
    "Config",
    "registry",
    "ConfigValidationError",
    "SimpleFrozenDict",
    "SimpleFrozenList",
]
