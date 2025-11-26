import copy
import inspect
import json
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import catalogue
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from pydantic.fields import FieldInfo

from ._config import (
    ARGS_FIELD,
    ARGS_FIELD_ALIAS,
    RESERVED_FIELDS,
    RESERVED_FIELDS_REVERSE,
    Config,
)
from ._errors import ConfigValidationError
from .util import is_promise

_PromisedType = TypeVar("_PromisedType")


class EmptySchema(BaseModel):
    model_config = {"extra": "allow", "arbitrary_types_allowed": True}


@dataclass
class Promise(Generic[_PromisedType]):
    registry: str
    name: str
    var_args: List[Any]
    kwargs: Dict[str, Any]
    getter: Union[Callable[..., _PromisedType], catalogue.RegistryError]
    schema: Optional[Type[BaseModel]]

    @property
    def return_type(self) -> _PromisedType:
        if isinstance(self.getter, catalogue.RegistryError):
            raise self.getter
        signature = inspect.signature(self.getter)
        return signature.return_annotation

    def validate(self) -> Any:
        kwargs = dict(self.kwargs)
        args = list(self.var_args)
        if args:
            kwargs[ARGS_FIELD] = args
        try:
            _ = self.schema.model_validate(kwargs)
        except ValidationError as e:
            raise ConfigValidationError(config=kwargs, errors=e.errors()) from None

    def resolve(self, validate: bool = True) -> Any:
        if isinstance(self.getter, catalogue.RegistryError):
            raise self.getter
        kwargs = _recursive_resolve(self.kwargs, validate=validate)
        args = _recursive_resolve(self.var_args, validate=validate)
        args = list(args.values()) if isinstance(args, dict) else args
        if validate:
            schema_args = dict(kwargs)
            if args:
                schema_args[ARGS_FIELD] = args
            try:
                _ = self.schema.model_validate(schema_args)
            except ValidationError as e:
                raise ConfigValidationError(config=kwargs, errors=e.errors()) from None
        return self.getter(*args, **kwargs)  # type: ignore

    @classmethod
    def from_dict(cls, registry, values, *, validate: bool = True) -> "Promise":
        reg_name, func_name = registry.get_constructor(values)
        var_args, kwargs = registry.parse_args(values)
        try:
            getter = registry.get(reg_name, func_name)
        except catalogue.RegistryError as e:
            getter = e
        if isinstance(getter, catalogue.RegistryError):
            schema = EmptySchema
        else:
            schema = make_func_schema(getter)
        if not validate:
            kwargs = remove_extra_keys(kwargs, schema)
        output = cls(
            registry=reg_name,
            name=func_name,
            var_args=var_args,
            kwargs=kwargs,
            getter=getter,
            schema=schema,
        )
        # if validate:
        #    output.validate()
        return output


def _recursive_resolve(obj, validate: bool):
    if isinstance(obj, list):
        return [_recursive_resolve(v, validate=validate) for v in obj]
    elif isinstance(obj, dict):
        return {k: _recursive_resolve(v, validate=validate) for k, v in obj.items()}
    elif isinstance(obj, Promise):
        return obj.resolve(validate=validate)
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
        if func is None:
            raise ValueError(f"Could not find '{func_name}' in '{registry_name}'")
        return func

    @classmethod
    def resolve(
        cls,
        config: Union[Config, Dict[str, Dict[str, Any]]],
        *,
        schema: Type[BaseModel] = EmptySchema,
        overrides: Dict[str, Any] = {},
        validate: bool = True,
    ) -> Dict[str, Any]:
        config = cls.fill(
            config,
            schema=schema,
            overrides=overrides,
            validate=validate,
            interpolate=True,
        )
        promised = insert_promises(cls, config, resolve=True, validate=True)
        resolved = resolve_promises(promised, validate=validate)
        fixed = fix_positionals(resolved)
        assert isinstance(fixed, dict)
        if validate:
            validate_resolved(fixed, schema)
        return fixed

    @classmethod
    def fill(
        cls,
        config: Union[Config, Dict[str, Dict[str, Any]]],
        *,
        schema: Type[BaseModel] = EmptySchema,
        overrides: Dict[str, Any] = {},
        validate: bool = True,
        interpolate: bool = False,
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
            cls, config, schema=schema, overrides=overrides, validate=validate
        )
        if validate:
            full_schema = cls._make_unresolved_schema(schema, filled)
            try:
                _ = full_schema.model_validate(filled)
            except ValidationError as e:
                raise ConfigValidationError(config=config, errors=e.errors()) from None
        filled = Config(filled, section_order=section_order)
        # Merge the original config back to preserve variables if we started
        # with a config that wasn't interpolated. Here, we prefer variables to
        # allow auto-filling a non-interpolated config without destroying
        # variable references.
        if not interpolate and not is_interpolated:
            filled = filled.merge(
                Config(orig_config, is_interpolated=False), remove_extra=True
            )
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

    @classmethod
    def make_promise_schema(
        cls, obj: Dict[str, Any], *, resolve: bool = True
    ) -> Type[BaseModel]:
        """Create a schema for a promise dict (referencing a registry function)
        by inspecting the function signature.
        """
        reg_name, func_name = cls.get_constructor(obj)
        if not resolve and not cls.has(reg_name, func_name):
            return EmptySchema
        func = cls.get(reg_name, func_name)
        return make_func_schema(func)

    @classmethod
    def _make_unresolved_schema(
        cls, schema: Type[BaseModel], config
    ) -> Type[BaseModel]:
        """Make a single schema to validate against, representing data with promises unresolved.

        When the config provides a value via a promise, we build a schema for the arguments for the
        function it references, and insert that into the schema. This subschema describes a dictionary
        that would be valid to call the referenced function.
        """
        if not schema.model_fields:
            schema = _make_dummy_schema(config)
        fields = {}
        for name, field in schema.model_fields.items():
            if name not in config:
                fields[name] = (field.annotation, Field(field.default))
            elif is_promise(config[name]):
                fields[name] = (
                    cls._make_unresolved_promise_schema(config[name]),
                    Field(field.default),
                )
            elif field.annotation is None:
                fields[name] = (Any, Field(field.default))
            elif (
                # On Python 3.10, typing.* objects were not classes
                isinstance(field.annotation, type)
                and issubclass(field.annotation, BaseModel)
            ):
                fields[name] = cls._make_unresolved_schema(
                    field.annotation, config[name]
                )
            elif isinstance(config[name], dict):
                fields[name] = cls._make_unresolved_schema(
                    _make_dummy_schema(config[name]), config
                )
            else:
                fields[name] = (field.annotation, Field(...))
        return create_model(
            "UnresolvedConfig", __config__={"extra": "forbid"}, **fields
        )

    @classmethod
    def _make_unresolved_promise_schema(cls, obj: Dict[str, Any]) -> Type[BaseModel]:
        """Create a schema for a promise dict (referencing a registry function)
        by inspecting the function signature.
        """
        reg_name, func_name = cls.get_constructor(obj)
        if not cls.has(reg_name, func_name):
            return EmptySchema
        func = cls.get(reg_name, func_name)
        fields = get_func_fields(func)
        if ARGS_FIELD_ALIAS in fields and isinstance(obj.get(ARGS_FIELD), dict):
            # You're allowed to provide variable args as a dict or a list.
            # It's a dict if the values are sections, like 'items.*.fork',
            # and a list if it's like items = ['fork']
            fields[ARGS_FIELD_ALIAS] = (Dict, fields[ARGS_FIELD_ALIAS][1])
        for name, (field_type, field_info) in list(fields.items()):
            if name in obj and is_promise(obj[name]):
                fields[name] = (
                    cls._make_unresolved_promise_schema(obj[name]),
                    Field(field_info.default),
                )
            elif name in obj and isinstance(obj[name], dict):
                fields[name] = (
                    cls._make_unresolved_schema(EmptySchema, obj[name]),
                    Field(field_info.default),
                )
        fields[f"@{reg_name}"] = (str, Field(...))
        model_config = {
            "extra": "forbid",
            "arbitrary_types_allowed": True,
            "alias_generator": alias_generator,
        }
        return create_model(f"{reg_name} {func_name} model", __config__=model_config, **fields)  # type: ignore


def _make_dummy_schema(config):
    fields = {}
    for name, value in config.items():
        fields[name] = (Any, Field(...))
    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": True,
        "alias_generator": alias_generator,
    }
    return create_model("DummyModel", __config__=model_config, **fields)


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
    schema: Type[BaseModel] = EmptySchema,
    *,
    validate: bool = True,
    overrides: Dict[str, Dict[str, Any]] = {},
) -> Dict[str, Any]:
    overrided = apply_overrides(dict(config), overrides)
    defaulted = fill_defaults(registry, overrided, schema)
    if not validate:
        defaulted = remove_extra_keys(defaulted, schema=schema)
    return defaulted


def validate_resolved(config, schema: Type[BaseModel]):
    # If value is a generator we can't validate type without
    # consuming it (which doesn't work if it's infinite – see
    # schedule for examples). So we skip it.
    config = dict(config)
    try:
        _ = schema.model_validate(config)
    except ValidationError as e:
        raise ConfigValidationError(config=config, errors=e.errors()) from None


def fill_defaults(
    registry, config: Dict[str, Any], schema: Type[BaseModel]
) -> Dict[str, Any]:
    output = dict(config)
    for name, field in schema.model_fields.items():
        # Account for the alias on variable positional args
        alias = field.alias if field.alias is not None else name
        if alias not in config and field.default != Ellipsis:
            if isinstance(field.default, BaseModel):
                output[alias] = field.default.model_dump()
            else:
                output[alias] = field.default
    for key, value in output.items():
        if registry.is_promise(value):
            schema = registry.make_promise_schema(value, resolve=False)
            value = fill_defaults(registry, value, schema=schema)
            output[key] = value
        elif isinstance(value, dict):
            output[key] = fill_defaults(registry, value, EmptySchema)
    return output


def remove_extra_keys(
    config: Dict[str, Any], schema: Type[BaseModel]
) -> Dict[str, Any]:
    """Remove keys from the config that aren't in the schema.
    This is used when validate=False
    """
    if schema.model_config.get("extra") == "allow":
        return dict(config)
    output = {}
    for field_name, field_schema in schema.model_fields.items():
        if field_name in config:
            if hasattr(field_schema.annotation, "model_fields"):
                output[field_name] = remove_extra_keys(
                    config[field_name], field_schema.annotation
                )
            else:
                output[field_name] = config[field_name]
    return output


def insert_promises(
    registry, config: Dict[str, Dict[str, Any]], resolve: bool, validate: bool
) -> Dict[str, Dict[str, Any]]:
    """Create a version of a config dict where promises are recognised and replaced by Promise
    dataclasses
    """
    output = {}
    for key, value in config.items():
        if registry.is_promise(value):
            value = insert_promises(registry, value, resolve=resolve, validate=validate)
            output[key] = Promise.from_dict(
                registry,
                value,
                validate=validate,
            )
        elif isinstance(value, dict):
            output[key] = insert_promises(
                registry, value, resolve=resolve, validate=validate
            )
        else:
            output[key] = value
    return output


def resolve_promises(
    config: Dict[str, Dict[str, Any]], validate: bool
) -> Dict[str, Dict[str, Any]]:
    output = {}
    for key, value in config.items():
        if isinstance(value, dict):
            output[key] = resolve_promises(value, validate=validate)
        elif isinstance(value, Promise):
            output[key] = value.resolve(validate=validate)
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


def apply_overrides(
    config: Dict[str, Dict[str, Any]],
    overrides: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build first representation of the config:"""
    output = copy.deepcopy(config)
    for key, value in overrides.items():
        path = key.split(".")
        err_title = "Error parsing config overrides"
        err_msg = "not a section value that can be overridden"
        err = [{"loc": path, "msg": err_msg}]
        node = output
        for subkey in path[:-1]:
            if not isinstance(node, dict) or subkey not in node:
                raise ConfigValidationError(errors=err, title=err_title)
            node = node[subkey]
        if path[-1] not in node:
            raise ConfigValidationError(errors=err, title=err_title)
        node[path[-1]] = value
    return output


def make_func_schema(func) -> Type[BaseModel]:
    fields = get_func_fields(func)
    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": True,
        "alias_generator": alias_generator,
    }
    return create_model("ArgModel", __config__=model_config, **fields)  # type: ignore


def get_func_fields(func) -> Dict[str, Tuple[Type, FieldInfo]]:
    # Read the argument annotations and defaults from the function signature
    sig_args = {}
    for param in inspect.signature(func).parameters.values():
        # If no annotation is specified assume it's anything
        annotation = param.annotation if param.annotation != param.empty else Any
        # If no default value is specified assume that it's required
        default = param.default if param.default != param.empty else ...
        # Handle spread arguments and use their annotation as Sequence[whatever]
        if param.kind == param.VAR_POSITIONAL:
            spread_annot = Sequence[annotation]  # type: ignore
            sig_args[ARGS_FIELD_ALIAS] = (spread_annot, Field(default))
        else:
            name = RESERVED_FIELDS.get(param.name, param.name)
            sig_args[name] = (annotation, Field(default))
    return sig_args


def _is_model(type_):
    return issubclass(type_, BaseModel)
