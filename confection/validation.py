"""Lightweight type validation system replacing Pydantic.

Provides Schema base class, dynamic schema creation, and type validation
for config values against function signatures.
"""

import collections.abc
import inspect
import sys
import types
from pathlib import PurePath
from types import GeneratorType
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

# === Constrained Types ===


class StrictBool:
    """Only accepts actual bool values (not int 0/1)."""

    pass


class PositiveInt:
    """Only accepts positive integers (> 0, not bool)."""

    pass


class StrictFloat:
    """Only accepts actual float values (not int)."""

    pass


# === Field Info ===


class FieldInfo:
    """Information about a schema field."""

    __slots__ = ("default", "alias", "annotation")

    def __init__(self, default=..., *, alias=None):
        self.default = default
        self.alias = alias
        self.annotation = None

    def is_required(self):
        return self.default is ...


def Field(default=..., *, alias=None):
    """Create a field definition."""
    return FieldInfo(default=default, alias=alias)


# === Validation Error ===


class ValidationError(Exception):
    """Raised when schema validation fails."""

    def __init__(self, error_list):
        self._errors = error_list
        msgs = "; ".join(e.get("msg", "") for e in error_list)
        super().__init__(msgs)

    def errors(self):
        return self._errors


# === Schema ===


class _ValidatedResult:
    """Attribute-accessible result from model_validate."""

    def __init__(self, data):
        self.__dict__.update(data)


class Schema:
    """Base class for config validation schemas. Replaces pydantic.BaseModel."""

    model_config: dict = {"extra": "allow", "arbitrary_types_allowed": True}
    model_fields: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        fields = {}
        all_hints = {}
        for base in reversed(cls.__mro__):
            base_annotations = getattr(base, "__annotations__", {})
            all_hints.update(base_annotations)

        for name, annotation in all_hints.items():
            if name in ("model_config", "model_fields") or name.startswith("_"):
                continue
            default = ...
            for klass in cls.__mro__:
                if name in klass.__dict__:
                    val = klass.__dict__[name]
                    if isinstance(val, FieldInfo):
                        default = val.default
                    elif not isinstance(
                        val, (type, classmethod, staticmethod, property)
                    ):
                        if not callable(val):
                            default = val
                    break
            field = FieldInfo(default=default)
            field.annotation = annotation
            fields[name] = field

        cls.model_fields = fields

    def __init__(self, **kwargs):
        for name, field in self.__class__.model_fields.items():
            if name in kwargs:
                setattr(self, name, kwargs[name])
            elif not field.is_required():
                setattr(self, name, field.default)

    @classmethod
    def model_validate(cls, data):
        """Validate a dict against this schema."""
        alias_gen = cls.model_config.get("alias_generator")
        errors = _validate_schema(data, cls.model_fields, cls.model_config, alias_gen)
        if errors:
            raise ValidationError(errors)
        # Build result with defaults filled in
        result_data = dict(data)
        for name, field in cls.model_fields.items():
            data_key = name
            if field.alias is not None:
                data_key = field.alias
            elif alias_gen:
                data_key = alias_gen(name)
            if data_key not in result_data and not field.is_required():
                result_data[data_key] = field.default
        return _ValidatedResult(result_data)

    @classmethod
    def from_function(
        cls,
        func,
        *,
        config=None,
    ):
        """Build a Schema subclass from a function's signature.

        Each parameter becomes a field.  The annotation is used as the type
        (defaulting to ``Any`` when missing) and the default value is
        preserved (parameters without defaults become required fields).

        ``*args`` parameters are wrapped in ``Sequence[annotation]`` and
        stored under the ``VARIABLE_POSITIONAL_ARGS`` field name.

        Forward-reference annotations are resolved via
        ``typing.get_type_hints`` against the function's module namespace.
        """
        from typing import Sequence as _Seq

        if config is None:
            config = {"extra": "forbid", "arbitrary_types_allowed": True}

        resolved = resolve_type_hints(func)
        fields = {}
        for param in inspect.signature(func).parameters.values():
            annotation = resolved.get(param.name, param.annotation)
            if annotation is inspect.Parameter.empty:
                annotation = Any
            if param.default is inspect.Parameter.empty:
                default = ...
            else:
                default = param.default
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                annotation = _Seq[annotation]  # type: ignore[valid-type]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            field = FieldInfo(default=default)
            field.annotation = annotation
            fields[param.name] = field

        return create_schema(
            func.__name__,
            __config__=config,
            **{name: (f.annotation, f) for name, f in fields.items()},
        )

    def model_dump(self):
        """Convert instance to dict."""
        result = {}
        for name in self.__class__.model_fields:
            if hasattr(self, name):
                val = getattr(self, name)
                if isinstance(val, Schema):
                    result[name] = val.model_dump()
                else:
                    result[name] = val
        return result


def create_schema(__name, __config__=None, **fields):
    """Dynamically create a Schema subclass.

    Each field value should be a (annotation, FieldInfo) tuple.
    """
    if __config__ is None:
        __config__ = {"extra": "allow"}

    processed = {}
    annotations = {}
    defaults = {}

    for name, field_def in fields.items():
        if isinstance(field_def, tuple) and len(field_def) == 2:
            annotation, field_info = field_def
            if not isinstance(field_info, FieldInfo):
                field_info = FieldInfo(default=field_info)
        else:
            raise ValueError(f"Field {name} must be (annotation, FieldInfo) tuple")

        field_info.annotation = annotation
        processed[name] = field_info
        annotations[name] = annotation
        if not field_info.is_required():
            defaults[name] = field_info.default

    namespace = {
        "__annotations__": annotations,
        "model_config": __config__,
    }
    namespace.update(defaults)

    cls = type(__name, (Schema,), namespace)
    # Apply alias_generator to fields that don't have explicit aliases
    alias_gen = __config__.get("alias_generator") if __config__ else None
    if alias_gen:
        for name, field in processed.items():
            if field.alias is None:
                field.alias = alias_gen(name)
    # Override with our processed fields (preserving aliases)
    cls.model_fields = processed
    return cls


# === Resolve forward references ===


def resolve_type_hints(func):
    """Resolve type hints for a function, handling forward references.

    Falls back to raw annotations if resolution fails.
    """
    try:
        module = sys.modules.get(getattr(func, "__module__", None))
        globalns = vars(module) if module else None
        return get_type_hints(func, globalns=globalns)
    except (NameError, AttributeError, TypeError, RecursionError):
        # NameError: unresolvable forward reference
        # AttributeError: module without expected attributes
        # TypeError: invalid annotation object
        # RecursionError: self-referential types (Python 3.13+)
        return {}


# === Type Validation ===


def _is_generator_value(value):
    """Check if value is a generator/iterator that shouldn't be consumed."""
    return isinstance(
        value, (GeneratorType, collections.abc.Iterator)
    ) and not isinstance(value, (str, bytes))


def _error_type_for(annotation):
    """Get an error type string for an annotation."""
    if annotation is int or annotation is PositiveInt:
        return "int_parsing"
    elif annotation is str:
        return "string_type"
    elif annotation is float or annotation is StrictFloat:
        return "float_parsing"
    elif annotation is bool or annotation is StrictBool:
        return "bool_type"
    return "value_error"


def validate_type(value, annotation):
    """Validate value against a type annotation.

    Returns None if valid, or an error message string if invalid.
    Generators/iterators always pass through without validation.
    """
    if annotation is Any or annotation is inspect.Parameter.empty:
        return None

    if annotation is type(None):
        return None if value is None else "Input should be None"

    # Generators always pass through (they can't be validated without consumption)
    if _is_generator_value(value):
        return None

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Annotated[X, ...] -> validate against X
    # get_origin returns None for Annotated in Python 3.10, so also check __metadata__
    if origin is Annotated or hasattr(annotation, "__metadata__"):
        return validate_type(value, get_args(annotation)[0])

    # Union / Optional  (typing.Union and Python 3.10+ X | Y syntax)
    if origin is Union or origin is types.UnionType:
        for arg in args:
            if validate_type(value, arg) is None:
                return None
        type_names = ", ".join(_type_display(a) for a in args)
        return f"Input should be valid for union type: {type_names}"

    # Literal
    if origin is Literal:
        if value in args:
            return None
        return f"Input should be {' or '.join(repr(a) for a in args)}"

    # NewType: unwrap to the supertype
    if callable(annotation) and hasattr(annotation, "__supertype__"):
        return validate_type(value, annotation.__supertype__)

    # Constrained types
    if annotation is StrictBool:
        if type(value) is not bool:
            return "Input should be a valid boolean"
        return None

    if annotation is PositiveInt:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return "Input should be a positive integer, greater than 0"
        return None

    if annotation is StrictFloat:
        if type(value) is not float:
            return "Input should be a valid float"
        return None

    # Generic types
    if origin is not None:
        return _validate_generic(value, origin, args)

    # Plain types
    if isinstance(annotation, type):
        # If annotation is a Schema subclass and value is a dict, validate against it
        if issubclass(annotation, Schema) and isinstance(value, dict):
            errors = _validate_schema(
                value,
                annotation.model_fields,
                annotation.model_config,
                annotation.model_config.get("alias_generator"),
            )
            if errors:
                return f"Input should be an instance of {annotation.__name__}"
            return None
        return _validate_plain_type(value, annotation)

    # Unknown annotation - accept
    return None


def _validate_plain_type(value, typ):
    """Validate value against a plain (non-generic) type."""
    if typ is bool:
        if not isinstance(value, bool):
            return "Input should be a valid boolean"
        return None

    if typ is int:
        if isinstance(value, bool):
            return (
                "Input should be a valid integer, unable to parse string as an integer"
            )
        if isinstance(value, int):
            return None
        if isinstance(value, str):
            try:
                int(value)
                return None
            except (ValueError, TypeError):
                pass
        return "Input should be a valid integer, unable to parse string as an integer"

    if typ is float:
        if isinstance(value, bool):
            return "Input should be a valid number"
        if isinstance(value, (int, float)):
            return None
        if isinstance(value, str):
            try:
                float(value)
                return None
            except (ValueError, TypeError):
                pass
        return "Input should be a valid number"

    if typ is str:
        if isinstance(value, str):
            return None
        return "Input should be a valid string"

    # Path: accept strings (pydantic coerces str → Path)
    if issubclass(typ, PurePath):
        if isinstance(value, (str, PurePath)):
            return None
        return "Input should be a valid path"

    # Custom class - isinstance check
    try:
        if isinstance(value, typ):
            return None
        # Types that declare custom validation via pydantic's schema protocol
        # (e.g. thinc's Floats2d).  Extract and call the validator directly
        # — the hook returns a plain dict, no pydantic import needed.
        if hasattr(typ, "__get_pydantic_core_schema__"):
            return _call_pydantic_schema_validator(typ, value)
        # pydantic v1 protocol: __get_validators__ yields single-arg
        # validator functions.
        if hasattr(typ, "__get_validators__"):
            return _call_pydantic_v1_validators(typ, value)
        # For constrained subtypes without validator hooks: if the
        # annotation inherits from the value's type, the value is
        # structurally compatible.
        if issubclass(typ, type(value)):
            return None
    except TypeError:
        return None

    return f"Input should be an instance of {getattr(typ, '__name__', str(typ))}"


class _AnySchemaHandler:
    """Minimal stand-in for pydantic's GetCoreSchemaHandler.

    Passed to __get_pydantic_core_schema__ so we can extract the
    validator function without importing pydantic.  The handler is
    only called as ``handler(source_type)`` and must return a core
    schema dict — ``{"type": "any"}`` tells pydantic "accept anything"
    which is the right inner schema for a plain validator.
    """

    def __call__(self, _source_type):
        return {"type": "any"}


def _call_pydantic_schema_validator(typ, value):
    """Call the validator from a type's __get_pydantic_core_schema__ hook.

    The hook returns a plain dict describing the schema.  We extract the
    validator function and call it directly — no pydantic import needed.
    Returns None on success, or an error message string on failure.
    """
    schema = typ.__get_pydantic_core_schema__(typ, _AnySchemaHandler())
    # Navigate the schema dict to find the validator function.
    # Typical shapes:
    #   {"type": "function-after", "function": {"type": "no-info", "function": <fn>}, ...}
    #   {"type": "function-plain", "function": {"type": "no-info", "function": <fn>}}
    fn_entry = schema.get("function", {})
    if isinstance(fn_entry, dict):
        validator = fn_entry.get("function")
    else:
        return None  # unrecognised shape — can't extract validator
    if not callable(validator):
        return None
    try:
        validator(value)
    except (ValueError, TypeError, AssertionError) as e:
        return str(e)
    return None


def _call_pydantic_v1_validators(typ, value):
    """Call validators from a type's __get_validators__ hook (pydantic v1).

    Validators may accept 1 arg (value), 2 args (value, field), or
    3 args (value, field, config).  For multi-arg validators that need
    field metadata (like number constraints), we build a minimal shim.
    """
    for validator in typ.__get_validators__():
        try:
            nparams = len(inspect.signature(validator).parameters)
        except (ValueError, TypeError):
            nparams = 1
        if nparams > 2:
            continue  # skip validators requiring pydantic config objects
        try:
            if nparams == 1:
                value = validator(value)
            else:
                value = validator(value, _PydanticV1FieldShim(typ))
        except (ValueError, TypeError, AssertionError) as e:
            return str(e)
    return None


class _PydanticV1FieldShim:
    """Minimal shim for pydantic v1 ModelField, providing just enough
    for constraint validators (number_size_validator etc.)."""

    def __init__(self, typ):
        self.type_ = typ


def _validate_generic(value, origin, args):
    """Validate value against a generic type (List[X], Dict[K,V], etc.)."""
    # Callable
    if origin is collections.abc.Callable:
        if callable(value):
            return None
        return "Input should be callable"

    # list / List[X]
    if origin is list:
        if not isinstance(value, list):
            return "Input should be a valid list"
        if args:
            for i, item in enumerate(value):
                err = validate_type(item, args[0])
                if err:
                    return f"Item {i}: {err}"
        return None

    # dict / Dict[K, V]
    if origin is dict:
        if not isinstance(value, dict):
            return "Input should be a valid dictionary"
        if args and len(args) == 2:
            for k, v in value.items():
                err = validate_type(k, args[0])
                if err:
                    return f"Key {k!r}: {err}"
                err = validate_type(v, args[1])
                if err:
                    return f"Value for {k!r}: {err}"
        return None

    # tuple / Tuple  —  Tuple[int, str] (fixed) vs Tuple[int, ...] (variable)
    if origin is tuple:
        if not isinstance(value, (tuple, list)):
            return "Input should be a valid tuple"
        if args:
            if len(args) == 2 and args[1] is Ellipsis:
                # Tuple[X, ...] — variable-length, all elements same type
                for i, item in enumerate(value):
                    err = validate_type(item, args[0])
                    if err:
                        return f"Item {i}: {err}"
            elif args != ((),):
                # Tuple[X, Y, Z] — fixed-length positional
                if len(value) != len(args):
                    return f"Expected {len(args)} items in tuple, got {len(value)}"
                for i, (item, expected) in enumerate(zip(value, args)):
                    err = validate_type(item, expected)
                    if err:
                        return f"Item {i}: {err}"
        return None

    # set / Set[X]
    if origin is set:
        if not isinstance(value, set):
            return "Input should be a valid set"
        if args:
            for item in value:
                err = validate_type(item, args[0])
                if err:
                    return f"Set item: {err}"
        return None

    # frozenset / FrozenSet[X]
    if origin is frozenset:
        if not isinstance(value, frozenset):
            return "Input should be a valid frozenset"
        if args:
            for item in value:
                err = validate_type(item, args[0])
                if err:
                    return f"Frozenset item: {err}"
        return None

    # Sequence
    if origin is collections.abc.Sequence:
        if isinstance(value, (list, tuple)):
            if args:
                for i, item in enumerate(value):
                    err = validate_type(item, args[0])
                    if err:
                        return f"Item {i}: {err}"
            return None
        if isinstance(value, str):
            return None
        return "Input should be a valid sequence"

    # Iterable
    if origin is collections.abc.Iterable:
        if hasattr(value, "__iter__"):
            return None
        return "Input should be iterable"

    # Mapping (covers Mapping, MutableMapping, OrderedDict, etc.)
    if isinstance(origin, type) and issubclass(origin, collections.abc.Mapping):
        if isinstance(value, collections.abc.Mapping):
            return None
        return "Input should be a valid mapping"

    # AbstractSet (covers Set, FrozenSet, MutableSet ABCs)
    if isinstance(origin, type) and issubclass(origin, collections.abc.Set):
        if isinstance(value, collections.abc.Set):
            return None
        return "Input should be a valid set"

    # Iterator / Generator
    if isinstance(origin, type) and issubclass(
        origin, (collections.abc.Iterator, collections.abc.Generator)
    ):
        if hasattr(value, "__next__") or hasattr(value, "__iter__"):
            return None
        return "Input should be an iterator"

    # Type[X] — check value is a class and optionally a subclass of X
    if origin is type:
        if not isinstance(value, type):
            return "Input should be a type"
        if args and args[0] is not Any:
            expected = args[0]
            try:
                if not issubclass(value, expected):
                    return (
                        f"Input should be a subclass of"
                        f" {getattr(expected, '__name__', expected)}"
                    )
            except TypeError:
                pass  # expected is not a class (e.g. Union) — skip
        return None

    # For any other generic - try isinstance against origin
    if isinstance(origin, type):
        try:
            if isinstance(value, origin):
                return None
        except TypeError:
            return None
        return f"Input should be an instance of {origin.__name__}"

    return None


def _type_display(annotation):
    """Human-readable name for a type."""
    if annotation is type(None):
        return "None"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


# === Schema Validation ===


def _validate_schema(data, fields, config, alias_generator=None):
    """Validate a data dict against schema fields.

    Returns list of error dicts (empty if valid).
    """
    errors = []
    extra_mode = config.get("extra", "allow")

    # Build mapping: data_key -> (field_name, FieldInfo)
    key_to_field = {}
    known_keys = set()

    for name, field in fields.items():
        if field.alias is not None:
            data_key = field.alias
        elif alias_generator:
            data_key = alias_generator(name)
        else:
            data_key = name

        known_keys.add(data_key)
        key_to_field[data_key] = (name, field)

    # Check extra fields
    if extra_mode == "forbid":
        for key in data:
            if key not in known_keys:
                errors.append(
                    {
                        "loc": (key,),
                        "msg": "Extra inputs are not permitted",
                        "type": "extra_forbidden",
                    }
                )

    # Validate each field
    for data_key, (name, field) in key_to_field.items():
        if data_key in data:
            value = data[data_key]
            err = validate_type(value, field.annotation)
            if err:
                errors.append(
                    {
                        "loc": (data_key,),
                        "msg": err,
                        "type": _error_type_for(field.annotation),
                    }
                )
        elif field.is_required():
            errors.append(
                {
                    "loc": (data_key,),
                    "msg": "Field required",
                    "type": "missing",
                }
            )

    return errors


# === Pydantic Compatibility Shim ===

_pydantic_cache: dict = {}


def _get_pydantic_validation_error():
    """Return the pydantic ValidationError class(es) to catch.

    Tries both pydantic.v1 and pydantic so we catch the right exception
    regardless of which API the caller's model was built with.
    """
    errors = []
    try:
        from pydantic.v1 import ValidationError as V1Err

        errors.append(V1Err)
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from pydantic import ValidationError as V2Err

        errors.append(V2Err)
    except (ImportError, ModuleNotFoundError):
        pass
    if errors:
        return tuple(errors)
    # Should never happen — we only get here if someone passed a pydantic
    # model, which means pydantic is installed.  Fall back to Exception so
    # the except clause still works rather than crashing.
    return (Exception,)  # pragma: no cover


def _is_pydantic_model(cls):
    """Check if cls is a pydantic BaseModel class (v1 or v2) without hard-depending
    on pydantic. Returns False if pydantic is not installed."""
    if not isinstance(cls, type):
        return False
    if issubclass(cls, Schema):
        return False
    # pydantic v1 compat layer (pydantic.v1) or native v1
    try:
        from pydantic.v1 import BaseModel as V1
    except (ImportError, ModuleNotFoundError):
        pass
    else:
        if issubclass(cls, V1):
            return True
    # pydantic v2 (or native v1 without the .v1 subpackage)
    try:
        from pydantic import BaseModel as V2
    except (ImportError, ModuleNotFoundError):
        pass
    else:
        if issubclass(cls, V2):
            return True
    return False


def _pydantic_instance_to_dict(obj):
    """Convert a pydantic model instance to a dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def _extract_pydantic_fields(pydantic_cls):
    """Extract field definitions from a pydantic BaseModel class (v1 or v2)."""
    fields = {}

    if hasattr(pydantic_cls, "__fields__"):
        # pydantic v1 interface
        for name, pyd_field in pydantic_cls.__fields__.items():
            annotation = pyd_field.outer_type_
            # pydantic v1 unwraps Optional[X] into outer_type_=X +
            # allow_none=True.  Re-wrap so our validator sees the Union.
            if getattr(pyd_field, "allow_none", False):
                annotation = Optional[annotation]
            if pyd_field.required:
                default = ...
            else:
                default = pyd_field.default
            alias = pyd_field.alias if pyd_field.alias != name else None

            # Recursively convert nested pydantic model annotations
            if isinstance(annotation, type) and _is_pydantic_model(annotation):
                annotation = ensure_schema(annotation)
            # Convert pydantic instance defaults to dicts
            if default is not ... and hasattr(default, "__fields__"):
                default = _pydantic_instance_to_dict(default)

            field = FieldInfo(default=default, alias=alias)
            field.annotation = annotation
            fields[name] = field

    elif hasattr(pydantic_cls, "model_fields"):
        # pydantic v2 interface
        for name, pyd_field in pydantic_cls.model_fields.items():
            annotation = pyd_field.annotation
            if pyd_field.is_required():
                default = ...
            else:
                default = pyd_field.default
            alias = pyd_field.alias

            if isinstance(annotation, type) and _is_pydantic_model(annotation):
                annotation = ensure_schema(annotation)
            if default is not ... and hasattr(default, "model_dump"):
                default = _pydantic_instance_to_dict(default)

            field = FieldInfo(default=default, alias=alias)
            field.annotation = annotation
            fields[name] = field

    return fields


def _extract_pydantic_config(pydantic_cls):
    """Extract model config from a pydantic BaseModel class (v1 or v2)."""
    config = {"extra": "allow"}

    if hasattr(pydantic_cls, "__config__"):
        # pydantic v1: inner class Config
        cfg = pydantic_cls.__config__
        extra = getattr(cfg, "extra", "allow")
        # v1 may use an enum (e.g. Extra.forbid); extract the .value
        if hasattr(extra, "value"):
            extra = extra.value
        config["extra"] = extra if isinstance(extra, str) else str(extra)
        if hasattr(cfg, "arbitrary_types_allowed"):
            config["arbitrary_types_allowed"] = cfg.arbitrary_types_allowed
    elif hasattr(pydantic_cls, "model_config") and isinstance(
        pydantic_cls.model_config, dict
    ):
        # pydantic v2: dict
        config = dict(pydantic_cls.model_config)

    return config


def ensure_schema(schema_cls):
    """Ensure *schema_cls* satisfies the Schema interface.

    If it already is a Schema subclass, return it unchanged.
    If it is a pydantic BaseModel (v1 or v2), build a thin Schema wrapper
    that exposes the same ``model_fields`` / ``model_config`` and delegates
    ``model_validate`` to the original pydantic class so that pydantic
    validators, strict types, constrained types etc. keep working.

    This allows downstream libraries (spaCy, thinc, …) to keep passing
    pydantic schemas to ``registry.resolve()`` / ``registry.fill()`` even
    though confection itself no longer depends on pydantic.
    """
    if isinstance(schema_cls, type) and issubclass(schema_cls, Schema):
        return schema_cls
    if not _is_pydantic_model(schema_cls):
        return schema_cls

    # Return cached conversion if available
    if schema_cls in _pydantic_cache:
        return _pydantic_cache[schema_cls]

    fields = _extract_pydantic_fields(schema_cls)
    config = _extract_pydantic_config(schema_cls)

    # Build wrapper class that inherits from Schema
    pyd_cls = schema_cls  # capture for closure

    wrapper = type(pydantic_cls_name(schema_cls), (Schema,), {})
    wrapper.model_fields = fields
    wrapper.model_config = config

    # Delegate model_validate to the original pydantic model so that
    # pydantic-level validators / strict types / constraints keep working.
    @classmethod  # type: ignore[misc]
    def _pydantic_model_validate(cls, data):
        # Resolve the concrete pydantic ValidationError class once so the
        # except clause is as narrow as possible.
        pyd_validation_err = _get_pydantic_validation_error()

        try:
            if hasattr(pyd_cls, "model_validate"):
                pyd_cls.model_validate(data)
            elif hasattr(pyd_cls, "parse_obj"):
                pyd_cls.parse_obj(data)
            else:
                pyd_cls(**data)
        except pyd_validation_err as e:
            raise ValidationError(e.errors()) from None
        # Return attribute-accessible result with defaults filled in
        result_data = dict(data)
        for name, field in cls.model_fields.items():
            data_key = field.alias if field.alias is not None else name
            if data_key not in result_data and not field.is_required():
                result_data[data_key] = field.default
        return _ValidatedResult(result_data)

    wrapper.model_validate = _pydantic_model_validate

    _pydantic_cache[schema_cls] = wrapper
    return wrapper


def pydantic_cls_name(cls):
    return getattr(cls, "__name__", "PydanticSchema")
