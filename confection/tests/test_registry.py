import inspect
import pickle
import platform
from types import GeneratorType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    Sequence,
    Literal,
)

import catalogue
import pytest

from pydantic import BaseModel, StrictFloat, PositiveInt, constr
from pydantic.types import StrictBool

from confection import Config, ConfigValidationError, EmptySchema
from confection.tests.util import Cat, make_tempdir, my_registry
from confection.util import Generator, partial


class IntsSchema(BaseModel):
    int1: int
    int2: int
    model_config = {"extra": "forbid"}


class DefaultsSchema(BaseModel):
    required: int
    optional: str = "default value"
    model_config = {"extra": "forbid"}


class LooseSchema(BaseModel):
    required: int
    optional: str = "default value"
    model_config = {"extra": "allow"}


class ComplexSchema(BaseModel):
    outer_req: int
    outer_opt: str = "default value"

    level2_req: IntsSchema
    level2_opt: DefaultsSchema = DefaultsSchema(required=1)


@my_registry.cats("var_args.v1")
def cats_var_args(*args: str) -> str:
    return " ".join(args)


@my_registry.cats("var_args_optional.v1")
def cats_var_args_optional(*args: str, foo: str = "hi"):
    return " ".join(args) + f"foo={foo}"


@my_registry.cats("no_args.v1")
def cats_no_args() -> str:
    return "(empty)"


@my_registry.cats("str_arg.v1")
def cats_str_arg(hi: str) -> str:
    return hi


@my_registry.cats("optional_str_arg.v1")
def cats_optional_str_arg(hi: str = "default value") -> str:
    return hi


@my_registry.cats("return_int_optional_str.v1")
def cats_return_int(hi: str = "default value") -> int:
    return 0


@pytest.mark.parametrize(
    "config,schema,expected",
    [
        ({"int1": 1, "int2": 2}, IntsSchema, "unchanged"),
        ({"required": 1, "optional": "provided"}, DefaultsSchema, "unchanged"),
        ({"required": 1, "optional": ""}, DefaultsSchema, "unchanged"),
        ({"required": 1}, DefaultsSchema, {"required": 1, "optional": "default value"}),
        (
            {
                "outer_req": 1,
                "outer_opt": "provided",
                "level2_req": {"int1": 1, "int2": 2},
                "level2_opt": {"required": 1, "optional": "provided"},
            },
            ComplexSchema,
            "unchanged",
        ),
        (
            {"outer_req": 1, "level2_req": {"int1": 1, "int2": 2}},
            ComplexSchema,
            {
                "outer_req": 1,
                "outer_opt": "default value",
                "level2_req": {"int1": 1, "int2": 2},
                "level2_opt": {"required": 1, "optional": "default value"},
            },
        ),
        (
            {
                "outer_req": 1,
                "outer_opt": "provided",
                "level2_req": {"int1": 1, "int2": 2},
            },
            ComplexSchema,
            {
                "outer_req": 1,
                "outer_opt": "provided",
                "level2_req": {"int1": 1, "int2": 2},
                "level2_opt": {"required": 1, "optional": "default value"},
            },
        ),
    ],
)
def test_fill_from_schema(config, schema, expected):
    """Basic tests filling config with defaults from a schema, but not from promises."""
    f = my_registry.fill(config, schema=schema)
    if expected == "unchanged":
        assert f == config
    else:
        assert f != config
        assert f == expected


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"required": {"@cats": "no_args.v1"}}, "unchanged"),
        (
            {"required": {"@cats": "catsie.v1", "evil": False, "cute": False}},
            "unchanged",
        ),
        (
            {"required": {"@cats": "catsie.v1", "evil": False, "cute": False}},
            "unchanged",
        ),
        (
            {"required": {"@cats": "catsie.v1", "evil": False}},
            {"required": {"@cats": "catsie.v1", "evil": False, "cute": True}},
        ),
        (
            {
                "required": {
                    "@cats": "optional_str_arg.v1",
                    "hi": {"@cats": "no_args.v1"},
                }
            },
            "unchanged",
        ),
        (
            {"required": {"@cats": "optional_str_arg.v1"}},
            {"required": {"@cats": "optional_str_arg.v1", "hi": "default value"}},
        ),
    ],
)
def test_fill_from_promises(config, expected):
    filled = my_registry.fill(config)
    if expected == "unchanged":
        assert filled == config
    else:
        assert filled != config
        assert filled == expected


@pytest.mark.parametrize(
    "config,schema,expected",
    [
        (
            {"required": 1, "optional": {"@cats": "optional_str_arg.v1"}},
            DefaultsSchema,
            {
                "required": 1,
                "optional": {"@cats": "optional_str_arg.v1", "hi": "default value"},
            },
        ),
        (
            {"required": {"@cats": "return_int_optional_str.v1", "hi": "provided"}},
            DefaultsSchema,
            {
                "required": {"@cats": "return_int_optional_str.v1", "hi": "provided"},
                "optional": "default value",
            },
        ),
        (
            {"required": {"@cats": "return_int_optional_str.v1"}},
            DefaultsSchema,
            {
                "required": {
                    "@cats": "return_int_optional_str.v1",
                    "hi": "default value",
                },
                "optional": "default value",
            },
        ),
    ],
)
def test_fill_from_both(config, schema, expected):
    filled = my_registry.fill(config, schema=schema)
    if expected == "unchanged":
        assert filled == config
    else:
        assert filled != config
        assert filled == expected


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"hello": 1, "world": 2}, "unchanged"),
        ({"config": {"@cats": "no_args.v1"}}, {"config": "(empty)"}),
        ({"required": {"@cats": "optional_str_arg.v1"}}, {"required": "default value"}),
        (
            {"required": {"@cats": "optional_str_arg.v1", "hi": "provided"}},
            {"required": "provided"},
        ),
        (
            {
                "required": {
                    "@cats": "optional_str_arg.v1",
                    "hi": {"@cats": "str_arg.v1", "hi": "nested"},
                }
            },
            {"required": "nested"},
        ),
    ],
)
def test_resolve(config, expected):
    resolved = my_registry.resolve(config)
    if expected == "unchanged":
        assert resolved == config
    else:
        assert resolved != config
        assert resolved == expected


@pytest.mark.parametrize(
    "config,schema,expected",
    [
        ({"required": "hi", "optional": 1}, DefaultsSchema, "unchanged"),
        (
            {"required": {"@cats": "no_args.v1"}, "optional": 1},
            DefaultsSchema,
            "unchanged",
        ),
        (
            {"required": {"@cats": "no_args.v1", "extra_arg": True}, "optional": 1},
            DefaultsSchema,
            "unchanged",
        ),
        # Drop extra args if we have a schema and we're not validating
        (
            {"required": "hi", "optional": 1, "extra_arg": True},
            DefaultsSchema,
            {"required": "hi", "optional": 1},
        ),
        # Keep the extra args if the schema says extra is allowed
        (
            {"required": "hi", "optional": 1, "extra_arg": True},
            LooseSchema,
            "unchanged",
        ),
    ],
)
def test_fill_allow_invalid(config, schema, expected):
    filled = my_registry.fill(config, schema=schema, validate=False)
    if expected == "unchanged":
        assert filled == config
    else:
        assert filled != config
        assert filled == expected


@pytest.mark.parametrize(
    "config,schema",
    [
        ({"int1": "str", "int2": 2}, IntsSchema),
        ({"required": "hi", "optional": 1}, DefaultsSchema),
        (
            {"required": {"@cats": "no_args.v1"}, "optional": 1},
            DefaultsSchema,
        ),
        (
            {"required": {"@cats": "no_args.v1", "extra_arg": True}, "optional": 1},
            DefaultsSchema,
        ),
        # Drop extra args if we have a schema and we're not validating
        (
            {"required": "hi", "optional": 1, "extra_arg": True},
            DefaultsSchema,
        ),
        # Keep the extra args if the schema says extra is allowed
        (
            {"required": "hi", "optional": 1, "extra_arg": True},
            LooseSchema,
        ),
    ],
)
def test_fill_raise_invalid(config, schema):
    with pytest.raises(ConfigValidationError):
        my_registry.fill(config, schema=schema, validate=True)


@pytest.mark.parametrize(
    "config,schema,expected",
    [
        ({"int1": 1, "int2": "bah"}, IntsSchema, "unchanged"),
        ({"required": "hi", "optional": 1}, DefaultsSchema, "unchanged"),
        (
            {"required": {"@cats": "no_args.v1"}, "optional": 1},
            DefaultsSchema,
            {"required": "(empty)", "optional": 1},
        ),
        (
            {"required": {"@cats": "no_args.v1", "extra_arg": True}, "optional": 1},
            DefaultsSchema,
            {"required": "(empty)", "optional": 1},
        ),
        # Drop extra args if we have a schema and we're not validating
        (
            {"required": "hi", "optional": 1, "extra_arg": True},
            DefaultsSchema,
            {"required": "hi", "optional": 1},
        ),
        # Keep the extra args if the schema says extra is allowed
        (
            {"required": "hi", "optional": 1, "extra_arg": True},
            LooseSchema,
            "unchanged",
        ),
    ],
)
def test_resolve_allow_invalid(config, schema, expected):
    resolved = my_registry.resolve(config, schema=schema, validate=False)
    if expected == "unchanged":
        assert resolved == config
    else:
        assert resolved != config
        assert resolved == expected


@pytest.mark.parametrize(
    "config,schema",
    [
        ({"int1": 1, "int2": "bah"}, IntsSchema),
        ({"required": "hi", "optional": 1}, DefaultsSchema),
        (
            {"required": {"@cats": "no_args.v1"}, "optional": 1},
            DefaultsSchema,
        ),
        (
            {"required": {"@cats": "no_args.v1", "extra_arg": True}, "optional": 1},
            DefaultsSchema,
        ),
        (
            {"required": "hi", "optional": 1, "extra_arg": True},
            DefaultsSchema,
        ),
        (
            {"required": "hi", "optional": 1, "extra_arg": True},
            LooseSchema,
        ),
    ],
)
def test_resolve_raise_invalid(config, schema):
    with pytest.raises(ConfigValidationError):
        my_registry.resolve(config, schema=schema, validate=True)
