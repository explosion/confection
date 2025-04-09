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
from ._fill_config import registry
from ._config import (
    Config,
    VARIABLE_RE,
    ARGS_FIELD,
    ARGS_FIELD_ALIAS,
    RESERVED_FIELDS,
    SECTION_PREFIX,
    JSON_EXCEPTIONS,
)


def try_load_json(value: str) -> Any:
    """Load a JSON string if possible, otherwise default to original value."""
    try:
        return srsly.json_loads(value)
    except Exception:
        return value


def try_dump_json(value: Any, data: Union[Dict[str, dict], Config, str] = "") -> str:
    """Dump a config value as JSON and output user-friendly error if it fails."""
    # Special case if we have a variable: it's already a string so don't dump
    # to preserve ${x:y} vs. "${x:y}"
    if isinstance(value, str) and VARIABLE_RE.search(value):
        return value
    if isinstance(value, str) and value.replace(".", "", 1).isdigit():
        # Work around values that are strings but numbers
        value = f'"{value}"'
    try:
        value = srsly.json_dumps(value)
        value = re.sub(r"\$([^{])", "$$\1", value)
        value = re.sub(r"\$$", "$$", value)
        return value
    except Exception as e:
        err_msg = (
            f"Couldn't serialize config value of type {type(value)}: {e}. Make "
            f"sure all values in your config are JSON-serializable. If you want "
            f"to include Python objects, use a registered function that returns "
            f"the object instead."
        )
        raise ConfigValidationError(config=data, desc=err_msg) from e


def deep_merge_configs(
    config: Union[Dict[str, Any], Config],
    defaults: Union[Dict[str, Any], Config],
    *,
    remove_extra: bool = False,
) -> Union[Dict[str, Any], Config]:
    """Deep merge two configs."""
    if remove_extra:
        # Filter out values in the original config that are not in defaults
        keys = list(config.keys())
        for key in keys:
            if key not in defaults:
                del config[key]
    for key, value in defaults.items():
        if isinstance(value, dict):
            node = config.setdefault(key, {})
            if not isinstance(node, dict):
                continue
            value_promises = [k for k in value if k.startswith("@")]
            value_promise = value_promises[0] if value_promises else None
            node_promises = [k for k in node if k.startswith("@")] if node else []
            node_promise = node_promises[0] if node_promises else None
            # We only update the block from defaults if it refers to the same
            # registered function
            if (
                value_promise
                and node_promise
                and (
                    value_promise in node
                    and node[value_promise] != value[value_promise]
                )
            ):
                continue
            if node_promise and (
                node_promise not in value or node[node_promise] != value[node_promise]
            ):
                continue
            defaults = deep_merge_configs(node, value, remove_extra=remove_extra)
        elif key not in config:
            config[key] = value
    return config


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


class _PromiseSchemaConfig:
    extra = "forbid"
    arbitrary_types_allowed = True
    alias_generator = alias_generator


def ensure_interpolated(config: Config) -> Config:
    # If a Config was loaded with interpolate=False, we assume it needs to
    # be interpolated first, otherwise we take it at face value
    is_interpolated = not isinstance(config, Config) or config.is_interpolated
    section_order = config.section_order if isinstance(config, Config) else None
    orig_config = config
    if not is_interpolated:
        config = Config(orig_config).interpolate()
    return config


__all__ = [
    "Config",
    "registry",
    "ConfigValidationError",
    "SimpleFrozenDict",
    "SimpleFrozenList",
]
