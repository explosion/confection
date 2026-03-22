import copy
import inspect
import sys
from dataclasses import dataclass
from types import GeneratorType
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import catalogue

from ._config import Config
from ._constants import (
    ARGS_FIELD,
    ARGS_FIELD_ALIAS,
    RESERVED_FIELDS,
    RESERVED_FIELDS_REVERSE,
)
from ._errors import ConfigValidationError
from .util import is_promise
from .validation import Schema

_PromisedType = TypeVar("_PromisedType")


@dataclass
class Promise(Generic[_PromisedType]):
    registry: str
    name: str
    var_args: List[Any]
    kwargs: Dict[str, Any]
    getter: Union[Callable[..., _PromisedType], catalogue.RegistryError]

    @property
    def return_type(self) -> _PromisedType:
        if isinstance(self.getter, catalogue.RegistryError):  # pragma: no cover
            raise self.getter  # pragma: no cover
        signature = inspect.signature(self.getter)
        return signature.return_annotation

    def resolve(self) -> Any:
        if isinstance(self.getter, catalogue.RegistryError):  # pragma: no cover
            raise self.getter  # pragma: no cover
        kwargs = _recursive_resolve(self.kwargs)
        args = _recursive_resolve(self.var_args)
        args = list(args.values()) if isinstance(args, dict) else args
        return self.getter(*args, **kwargs)  # type: ignore

    @classmethod
    def from_dict(cls, registry, values) -> "Promise":
        reg_name, func_name = registry.get_constructor(values)
        var_args, kwargs = registry.parse_args(values)
        try:
            getter = registry.get(reg_name, func_name)
        except catalogue.RegistryError as e:  # pragma: no cover
            getter = e  # pragma: no cover
        output = cls(
            registry=reg_name,
            name=func_name,
            var_args=var_args,
            kwargs=kwargs,
            getter=getter,
        )
        return output


def _recursive_resolve(obj):
    if isinstance(obj, list):
        return [_recursive_resolve(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: _recursive_resolve(v) for k, v in obj.items()}
    elif isinstance(obj, Promise):
        return obj.resolve()
    else:
        return obj


class registry:
    @classmethod
    def has(cls, registry_name: str, func_name: str) -> bool:
        """Check whether a function is available in a registry."""
        if not hasattr(cls, registry_name):
            return False
        reg = getattr(cls, registry_name)
        return func_name in reg

    @classmethod
    def get(cls, registry_name: str, func_name: str) -> Callable:
        """Get a registered function from a given registry."""
        if not hasattr(cls, registry_name):
            raise ValueError(f"Unknown registry: '{registry_name}'")
        reg = getattr(cls, registry_name)
        func = reg.get(func_name)
        if func is None:  # pragma: no cover -- catalogue raises RegistryError first
            raise ValueError(f"Could not find '{func_name}' in '{registry_name}'")
        return func

    @classmethod
    def resolve(
        cls,
        config: Union[Config, Dict[str, Dict[str, Any]]],
        *,
        schema=None,
        overrides: Dict[str, Any] = {},
        validate: bool = True,
    ) -> Dict[str, Any]:
        config = cls.fill(
            config,
            schema=schema,
            overrides=overrides,
            interpolate=True,
        )
        promised = insert_promises(cls, config, resolve=True)
        resolved = resolve_promises(promised)
        fixed = fix_positionals(resolved)
        assert isinstance(fixed, dict)
        if schema is not None and validate:
            Config(fixed).validate(schema)
        return fixed

    @classmethod
    def fill(
        cls,
        config: Union[Config, Dict[str, Dict[str, Any]]],
        *,
        schema=None,
        overrides: Dict[str, Any] = {},
        interpolate: bool = False,
        validate: bool = True,
    ) -> Config:
        if cls.is_promise(config):
            err_msg = "The top-level config object can't be a reference to a registered function."
            raise ConfigValidationError(config=config, errors=[{"msg": err_msg}])
        # If a Config was loaded with interpolate=False, we assume it needs to
        # be interpolated first, otherwise we take it at face value
        is_interpolated = not isinstance(config, Config) or config.is_interpolated
        section_order = config.section_order if isinstance(config, Config) else None
        orig_config = config
        if not is_interpolated:
            config = Config(orig_config).interpolate()
        filled = fill_config(
            cls, config, overrides=overrides
        )
        filled = Config(filled, section_order=section_order)
        # Merge the original config back to preserve variables if we started
        # with a config that wasn't interpolated. Here, we prefer variables to
        # allow auto-filling a non-interpolated config without destroying
        # variable references.
        if not interpolate and not is_interpolated:
            filled = filled.merge(
                Config(orig_config, is_interpolated=False), remove_extra=True
            )
        if schema is not None:
            filled.fill_defaults(schema)
        return filled

    @classmethod
    def is_promise(cls, obj: Any) -> bool:
        """Check whether an object is a "promise", i.e. contains a reference
        to a registered function (via a key starting with `"@"`.
        """
        return is_promise(obj)

    @classmethod
    def get_constructor(cls, obj: Dict[str, Any]) -> Tuple[str, str]:
        id_keys = [k for k in obj.keys() if k.startswith("@")]
        if len(id_keys) != 1:
            err_msg = f"A block can only contain one function registry reference. Got: {id_keys}"
            raise ConfigValidationError(config=obj, errors=[{"msg": err_msg}])
        else:
            key = id_keys[0]
            value = obj[key]
            return (key[1:], value)

    @classmethod
    def parse_args(cls, obj: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        args = []
        kwargs = {}
        for key, value in obj.items():
            if not key.startswith("@"):
                if key == ARGS_FIELD:
                    args = value
                else:
                    kwargs[key] = value
        return args, kwargs


def _is_config_section(obj) -> bool:
    """Check if a dict is a config section (all string keys) vs a data value."""
    if not isinstance(obj, dict):
        return False
    return all(isinstance(k, str) for k in obj.keys())


def alias_generator(name: str) -> str:
    """Generate field aliases in promise schema."""
    # Underscore fields are not allowed in model, so use alias
    if name == ARGS_FIELD_ALIAS:
        return ARGS_FIELD
    # Auto-alias fields that shadow base model attributes
    return RESERVED_FIELDS_REVERSE.get(name, name)


def fill_config(
    registry,
    config: Dict[str, Any],
    *,
    overrides: Dict[str, Dict[str, Any]] = {},
) -> Dict[str, Any]:
    overrided = apply_overrides(dict(config), overrides)
    return _fill_defaults(registry, overrided)


def _fill_defaults(registry, config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively fill default values from registered function signatures."""
    output = dict(config)
    for key, value in output.items():
        if is_promise(value):
            # Look up the function and fill its defaults
            output[key] = _fill_promise_defaults(registry, value)
        elif isinstance(value, dict):
            output[key] = _fill_defaults(registry, value)
    return output


def _fill_promise_defaults(registry, promise_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Fill default argument values for a promise block from the function signature."""
    reg_name, func_name = registry.get_constructor(promise_dict)
    func = registry.get(reg_name, func_name)
    schema = Schema.from_function(func)
    filled = dict(promise_dict)
    # Fill in defaults from the schema
    for param_name, field in schema.model_fields.items():
        if param_name not in filled and not field.is_required():
            filled[param_name] = field.default
    # Recurse into nested values (which may themselves be promises)
    for key, value in filled.items():
        if key.startswith("@"):
            continue
        if is_promise(value):
            filled[key] = _fill_promise_defaults(registry, value)
        elif isinstance(value, dict):
            filled[key] = _fill_defaults(registry, value)
    return filled


def insert_promises(
    registry, config: Dict[str, Dict[str, Any]], resolve: bool
) -> Dict[str, Dict[str, Any]]:
    """Create a version of a config dict where promises are recognised and replaced by
    Promise dataclasses
    """
    output = {}
    for key, value in config.items():
        if registry.is_promise(value):
            value = insert_promises(registry, value, resolve=resolve)
            output[key] = Promise.from_dict(
                registry,
                value,
            )
        elif isinstance(value, dict):
            output[key] = insert_promises(
                registry, value, resolve=resolve
            )
        else:
            output[key] = value
    return output


def resolve_promises(
    config: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    output = {}
    for key, value in config.items():
        if isinstance(value, dict):
            output[key] = resolve_promises(value)
        elif isinstance(value, Promise):
            output[key] = value.resolve()
        else:
            output[key] = value
    return output


def fix_positionals(config):
    """Ensure positionals are provided as a tuple, rather than a dict."""
    if isinstance(config, dict):
        output = {}
        for key, value in config.items():
            if key == ARGS_FIELD and isinstance(value, dict):
                value = tuple(value.values())
            if isinstance(value, dict):
                value = fix_positionals(value)
            elif isinstance(value, list) or isinstance(value, tuple):
                value = fix_positionals(value)
            output[key] = value
        return output
    elif isinstance(config, list):
        return [fix_positionals(v) for v in config]
    elif isinstance(config, tuple):
        return tuple([fix_positionals(v) for v in config])
    else:
        return config


def _deep_copy_with_uncopyable(obj: Any, memo: Optional[Dict[int, Any]] = None) -> Any:
    """Deep copy that passes through objects that can't be copied (like generators)."""
    if memo is None:
        memo = {}

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    if isinstance(obj, dict):
        result = {}
        memo[obj_id] = result
        for k, v in obj.items():
            result[_deep_copy_with_uncopyable(k, memo)] = _deep_copy_with_uncopyable(
                v, memo
            )
        return result
    elif isinstance(obj, list):
        result = []
        memo[obj_id] = result
        for item in obj:
            result.append(_deep_copy_with_uncopyable(item, memo))
        return result
    elif isinstance(obj, tuple):
        # Tuples are immutable, but we still need to copy their contents
        return tuple(_deep_copy_with_uncopyable(item, memo) for item in obj)
    else:
        return obj


def apply_overrides(
    config: Dict[str, Dict[str, Any]],
    overrides: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build first representation of the config:"""
    output = dict(config)
    for key, value in overrides.items():
        path = key.split(".")
        err_title = "Error parsing config overrides"
        err_msg = "not a section value that can be overridden"
        err = [{"loc": path, "msg": err_msg}]
        node = output
        for subkey in path[:-1]:
            if not isinstance(node, dict) or subkey not in node:  # pragma: no cover -- overrides validated in _parser
                raise ConfigValidationError(errors=err, title=err_title)  # pragma: no cover
            node = node[subkey]
        if path[-1] not in node:  # pragma: no cover
            raise ConfigValidationError(errors=err, title=err_title)  # pragma: no cover
        node[path[-1]] = value  # pragma: no cover
    return output
