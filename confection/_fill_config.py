from typing import (
    Any,
    Union,
    Dict,
    List,
    Tuple,
    Type,
    Iterable,
    Optional,
    Sequence,
    TypeVar,
    Generic,
    Callable,
)
import inspect
from dataclasses import dataclass
import re
from pydantic import BaseModel, ValidationError, create_model
import copy


RESERVED_FIELDS = {"validate": "validate\u0020"}
ARGS_FIELD = "*"
ARGS_FIELD_ALIAS = "VARIABLE_POSITIONAL_ARGS"
SECTION_PREFIX = "__SECTION__:"
# Values that shouldn't be loaded during interpolation because it'd cause
# even explicit string values to be incorrectly parsed as bools/None etc.
JSON_EXCEPTIONS = ("true", "false", "null")
# Regex to detect whether a value contains a variable
VARIABLE_RE = re.compile(r"\$\{[\w\.:]+\}")

_PromisedType = TypeVar("_PromisedType")


class EmptySchema(BaseModel):
    model_config = {"extra": "allow", "arbitrary_types_allowed": True}


@dataclass
class Promise(Generic[_PromisedType]):
    registry: str
    name: str
    var_args: List[Any]
    kwargs: Dict[str, Any]
    getter: Callable[..., _PromisedType]
    schema: Type[BaseModel]

    @property
    def return_type(self) -> _PromisedType:
        signature = inspect.signature(self.getter)
        return signature.return_annotation

    def resolve(self, validate: bool = True) -> Any:
        kwargs = {}
        for name, kwarg in self.kwargs.items():
            if isinstance(kwarg, Promise):
                kwargs[name] = kwarg.resolve()
            else:
                kwargs[name] = kwarg
        args = []
        for arg in self.var_args:
            if isinstance(arg, Promise):
                args.append(arg.resolve())
            else:
                args.append(arg)
        if validate:
            schema_args = dict(kwargs)
            if args:
                schema_args[ARGS_FIELD_ALIAS] = args
            try:
                _ = self.schema.model_validate(schema_args)
            except ValidationError as e:
                raise ConfigValidationError(config=kwargs, errors=e.errors()) from None
        try:
            return self.getter(*args, **kwargs)
        except TypeError:
            print(self.schema.model_fields)
            print("Schema args", schema_args)
            print(self.schema.model_validate(schema_args))
            print("args", args, "kwargs", kwargs)
            raise

    @classmethod
    def from_dict(cls, registry, values, *, validate: bool = True) -> "Promise":
        reg_name, func_name = registry.get_constructor(values)
        var_args, kwargs = registry.parse_args(values)
        getter = registry.get(reg_name, func_name)
        schema = make_func_schema(getter)
        if not validate:
            kwargs = remove_extra_keys(kwargs, schema)
        return cls(
            registry=reg_name,
            name=func_name,
            var_args=var_args,
            kwargs=kwargs,
            getter=getter,
            schema=schema,
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


def validate_unresolved(registry, config, schema):
    promised = insert_promises(registry, config, resolve=True, validate=True)
    resolved = resolve_promises(promised, validate=True)
    validate_resolved(resolved, schema)
    return resolved


def fill_defaults(
    registry, config: Dict[str, Any], schema: Type[BaseModel]
) -> Dict[str, Any]:
    output = dict(config)
    for name, field in schema.model_fields.items():
        if name not in config and field.default:
            if isinstance(field.default, BaseModel):
                output[name] = field.default.model_dump()
            else:
                output[name] = field.default
    for key, value in output.items():
        if is_promise(value):
            schema = make_promise_schema(registry, value, resolve=False)
            value = fill_defaults(registry, value, schema=schema)
            output[key] = value
        elif isinstance(value, dict):
            output[key] = fill_defaults(registry, value, EmptySchema)
    return output


def remove_extra_keys(config: Dict[str, Any], schema: Type[BaseModel]) -> Dict[str, Any]:
    """Remove keys from the config that aren't in the schema.
    This is used when validate=False
    """
    if schema.model_config.get("extra") == "allow":
        return dict(config)
    output = {}
    for field_name, field_schema in schema.model_fields.items():
        if field_name in config:
            if hasattr(field_schema.annotation, "model_fields"):
                output[field_name] = remove_extra_keys(config[field_name], field_schema.annotation)
            else:
                output[field_name] = config[field_name]
    return output


def validate_resolved(config, schema: Type[BaseModel]):
    try:
        _ = schema.model_validate(config)
    except ValidationError as e:
        raise ConfigValidationError(config=config, errors=e.errors()) from None


def insert_promises(
        registry, config: Dict[str, Dict[str, Any]], resolve: bool, validate: bool
) -> Dict[str, Dict[str, Any]]:
    """Create a version of a config dict where promises are recognised and replaced by Promise
    dataclasses
    """
    output = {}
    for key, value in config.items():
        v_key = RESERVED_FIELDS.get(key, key)
        if is_promise(value):
            output[v_key] = Promise.from_dict(
                registry, insert_promises(registry, value, resolve=resolve, validate=validate), validate=validate
            )
        elif isinstance(value, dict):
            output[v_key] = insert_promises(registry, value, resolve=resolve, validate=validate)
        else:
            output[v_key] = value
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


def is_promise(obj) -> bool:
    if not hasattr(obj, "keys"):
        return False
    id_keys = [k for k in obj.keys() if k.startswith("@")]
    if len(id_keys):
        return True
    return False


def make_promise_schema(registry, obj, resolve: bool) -> Type[BaseModel]:
    """Create a schema for a promise dict (referencing a registry function)
    by inspecting the function signature.
    """
    reg_name, func_name = registry.get_constructor(obj)
    if not resolve and not registry.has(reg_name, func_name):
        return EmptySchema
    func = registry.get(reg_name, func_name)
    return make_func_schema(func)


def make_func_schema(func):
    # Read the argument annotations and defaults from the function signature
    sig_args = {}
    for param in inspect.signature(func).parameters.values():
        # If no annotation is specified assume it's anything
        annotation = param.annotation if param.annotation != param.empty else Any
        # Allow promises to stand in for their return-types
        annotation = Union[annotation, Promise[annotation]]
        # If no default value is specified assume that it's required
        default = param.default if param.default != param.empty else ...
        # Handle spread arguments and use their annotation as Sequence[whatever]
        if param.kind == param.VAR_POSITIONAL:
            spread_annot = Sequence[annotation]  # type: ignore
            sig_args[ARGS_FIELD_ALIAS] = (spread_annot, default)
        else:
            name = RESERVED_FIELDS.get(param.name, param.name)
            sig_args[name] = (annotation, default)
    model_config = {"extra": "forbid", "arbitrary_types_allowed": True, "alias_generator": alias_generator}
    return create_model("ArgModel", __config__=model_config, **sig_args)


class ConfigValidationError(ValueError):
    def __init__(
        self,
        *,
        config=None,
        errors=None,
        title: Optional[str] = "Config validation error",
        desc: Optional[str] = None,
        parent: Optional[str] = None,
        show_config: bool = True,
    ) -> None:
        """Custom error for validating configs.

        config (Union[Config, Dict[str, Dict[str, Any]], str]): The
            config the validation error refers to.
        errors (Union[Sequence[Mapping[str, Any]], Iterable[Dict[str, Any]]]):
            A list of errors as dicts with keys "loc" (list of strings
            describing the path of the value), "msg" (validation message
            to show) and optional "type" (mostly internals).
            Same format as produced by pydantic's validation error (e.errors()).
        title (str): The error title.
        desc (str): Optional error description, displayed below the title.
        parent (str): Optional parent to use as prefix for all error locations.
            For example, parent "element" will result in "element -> a -> b".
        show_config (bool): Whether to print the whole config with the error.

        ATTRIBUTES:
        config (Union[Config, Dict[str, Dict[str, Any]], str]): The config.
        errors (Iterable[Dict[str, Any]]): The errors.
        error_types (Set[str]): All "type" values defined in the errors, if
            available. This is most relevant for the pydantic errors that define
            types like "type_error.integer". This attribute makes it easy to
            check if a config validation error includes errors of a certain
            type, e.g. to log additional information or custom help messages.
        title (str): The title.
        desc (str): The description.
        parent (str): The parent.
        show_config (bool): Whether to show the config.
        text (str): The formatted error text.
        """
        self.config = config
        self.errors = errors
        self.title = title
        self.desc = desc
        self.parent = parent
        self.show_config = show_config
        self.error_types = set()
        if self.errors:
            for error in self.errors:
                err_type = error.get("type")
                if err_type:
                    self.error_types.add(err_type)
        self.text = self._format()
        ValueError.__init__(self, self.text)

    @classmethod
    def from_error(
        cls,
        err: "ConfigValidationError",
        title: Optional[str] = None,
        desc: Optional[str] = None,
        parent: Optional[str] = None,
        show_config: Optional[bool] = None,
    ) -> "ConfigValidationError":
        """Create a new ConfigValidationError based on an existing error, e.g.
        to re-raise it with different settings. If no overrides are provided,
        the values from the original error are used.

        err (ConfigValidationError): The original error.
        title (str): Overwrite error title.
        desc (str): Overwrite error description.
        parent (str): Overwrite error parent.
        show_config (bool): Overwrite whether to show config.
        RETURNS (ConfigValidationError): The new error.
        """
        return cls(
            config=err.config,
            errors=err.errors,
            title=title if title is not None else err.title,
            desc=desc if desc is not None else err.desc,
            parent=parent if parent is not None else err.parent,
            show_config=show_config if show_config is not None else err.show_config,
        )

    def _format(self) -> str:
        """Format the error message."""
        loc_divider = "->"
        data = []
        if self.errors:
            for error in self.errors:
                err_loc = f" {loc_divider} ".join([str(p) for p in error.get("loc", [])])
                if self.parent:
                    err_loc = f"{self.parent} {loc_divider} {err_loc}"
                data.append((err_loc, error.get("msg")))
        result = []
        if self.title:
            result.append(self.title)
        if self.desc:
            result.append(self.desc)
        if data:
            result.append("\n".join([f"{entry[0]}\t{entry[1]}" for entry in data]))
        if self.config and self.show_config:
            result.append(f"{self.config}")
        return "\n\n" + "\n".join(result)
