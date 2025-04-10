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


class StrsSchema(BaseModel):
    str1: str
    str2: str
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


good_catsie = {"@cats": "catsie.v1", "evil": False, "cute": True}
ok_catsie = {"@cats": "catsie.v1", "evil": False, "cute": False}
bad_catsie = {"@cats": "catsie.v1", "evil": True, "cute": True}
worst_catsie = {"@cats": "catsie.v1", "evil": True, "cute": False}


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


@my_registry.cats("var_str_args.v1")
def cats_var_str_args(*args: str) -> str:
    return " ".join(args)


@my_registry.cats("dict_arg.v1")
def cats_dict_arg(schedules: Dict[str, int]) -> int:
    return schedules["rate"]


@my_registry.cats("generic_cat.v1")
def cat_generic(cat: Cat[int, int]) -> Cat[int, int]:
    cat.name = "generic_cat"
    return cat


@pytest.mark.parametrize(
    "config,schema,expected",
    [
        ({"int1": 1, "int2": 2}, IntsSchema, "unchanged"),
        ({"str1": "1", "str2": "2"}, StrsSchema, "unchanged"),
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
        ({"str1": "1", "str2": {"@cats": "var_str_args.v1", "*": ["a1", "a2"]}}, StrsSchema, "unchanged"),
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
        (
            {"required": {"@cats": "dict_arg.v1", "schedules": {"rate": {"@cats": "no_args.v1"}}}},
            "unchanged"
        ),
        (
            {'a': {'@cats': 'var_args.v1', '*': {'foo': {'@cats': 'no_args.v1'}}}},
            "unchanged"
        )
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
        # Should we allow extra args in a promise block? I think no, right?
        (
            {"required": {"@cats": "no_args.v1"}, "optional": 1},
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


def test_is_promise():
    assert my_registry.is_promise(good_catsie)
    assert not my_registry.is_promise({"hello": "world"})
    assert not my_registry.is_promise(1)
    invalid = {"@complex": "complex.v1", "rate": 1.0, "@cats": "catsie.v1"}
    assert my_registry.is_promise(invalid)


def test_get_constructor():
    assert my_registry.get_constructor(good_catsie) == ("cats", "catsie.v1")


def test_parse_args():
    args, kwargs = my_registry.parse_args(bad_catsie)
    assert args == []
    assert kwargs == {"evil": True, "cute": True}


def test_make_promise_schema():
    schema = my_registry.make_promise_schema(good_catsie, resolve=True)
    assert "evil" in schema.model_fields
    assert "cute" in schema.model_fields


def test_create_registry():
    my_registry.dogs = catalogue.create(
        my_registry.namespace, "dogs", entry_points=False
    )
    assert hasattr(my_registry, "dogs")
    assert len(my_registry.dogs.get_all()) == 0
    my_registry.dogs.register("good_boy.v1", func=lambda x: x)
    assert len(my_registry.dogs.get_all()) == 1


def test_registry_methods():
    with pytest.raises(ValueError):
        my_registry.get("dfkoofkds", "catsie.v1")
    my_registry.cats.register("catsie.v123")(None)
    with pytest.raises(ValueError):
        my_registry.get("cats", "catsie.v123")


def test_resolve_schema():
    class TestBaseSubSchema(BaseModel):
        three: str
        model_config = {"extra": "forbid"}

    class TestBaseSchema(BaseModel):
        one: PositiveInt
        two: TestBaseSubSchema
        model_config = {"extra": "forbid"}

    class TestSchema(BaseModel):
        cfg: TestBaseSchema
        model_config = {"extra": "forbid"}

    config = {"one": 1, "two": {"three": {"@cats": "catsie.v1", "evil": True}}}
    my_registry.resolve({"three": {"@cats": "catsie.v1", "evil": True}}, schema=TestBaseSubSchema)
    config = {"one": -1, "two": {"three": {"@cats": "catsie.v1", "evil": True}}}
    with pytest.raises(ConfigValidationError):
        # "one" is not a positive int
        my_registry.resolve({"cfg": config}, schema=TestSchema)
    config = {"one": 1, "two": {"four": {"@cats": "catsie.v1", "evil": True}}}
    with pytest.raises(ConfigValidationError):
        # "three" is required in subschema
        my_registry.resolve({"cfg": config}, schema=TestSchema)


def test_make_config_positional_args():
    @my_registry.cats("catsie.v567")
    def catsie_567(*args: Optional[str], foo: str = "bar"):
        assert args[0] == "^_^"
        assert args[1] == "^(*.*)^"
        assert foo == "baz"
        return args[0]

    args = ["^_^", "^(*.*)^"]
    cfg = {"config": {"@cats": "catsie.v567", "foo": "baz", "*": args}}
    assert my_registry.resolve(cfg)["config"] == "^_^"


def test_make_config_positional_args_complex():
    @my_registry.cats("catsie.v890")
    def catsie_890(*args: Optional[Union[StrictBool, PositiveInt]]):
        assert args[0] == 123
        return args[0]

    cfg = {"config": {"@cats": "catsie.v890", "*": [123, True, 1, False]}}
    assert my_registry.resolve(cfg)["config"] == 123
    cfg = {"config": {"@cats": "catsie.v890", "*": [123, "True"]}}
    with pytest.raises(ConfigValidationError):
        # "True" is not a valid boolean or positive int
        my_registry.resolve(cfg)


def test_validation_no_validate():
    config = {"one": 1, "two": {"three": {"@cats": "catsie.v1", "evil": "false"}}}
    result = my_registry.resolve({"cfg": config}, validate=False)
    filled = my_registry.fill({"cfg": config}, validate=False)
    assert result["cfg"]["one"] == 1
    assert result["cfg"]["two"] == {"three": "scratch!"}
    assert filled["cfg"]["two"]["three"]["evil"] == "false"
    assert filled["cfg"]["two"]["three"]["cute"] is True


def test_validation_generators_iterable():
    @my_registry.optimizers("test_optimizer.v1")
    def test_optimizer_v1(rate: float) -> None:
        return None

    @my_registry.schedules("test_schedule.v1")
    def test_schedule_v1(some_value: float = 1.0) -> Iterable[float]:
        while True:
            yield some_value

    config = {"optimizer": {"@optimizers": "test_optimizer.v1", "rate": 0.1}}
    my_registry.resolve(config)


def test_validation_unset_type_hints():
    """Test that unset type hints are handled correctly (and treated as Any)."""

    @my_registry.optimizers("test_optimizer.v2")
    def test_optimizer_v2(rate, steps: int = 10) -> None:
        return None

    config = {"test": {"@optimizers": "test_optimizer.v2", "rate": 0.1, "steps": 20}}
    my_registry.resolve(config)


def test_validation_bad_function():
    @my_registry.optimizers("bad.v1")
    def bad() -> None:
        raise ValueError("This is an error in the function")

    @my_registry.optimizers("good.v1")
    def good() -> None:
        return None

    # Bad function
    config = {"test": {"@optimizers": "bad.v1"}}
    with pytest.raises(ValueError):
        my_registry.resolve(config)
    # Bad function call
    config = {"test": {"@optimizers": "good.v1", "invalid_arg": 1}}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve(config)


def test_objects_from_config():
    config = {
        "optimizer": {
            "@optimizers": "my_cool_optimizer.v1",
            "beta1": 0.2,
            "learn_rate": {
                "@schedules": "my_cool_repetitive_schedule.v1",
                "base_rate": 0.001,
                "repeat": 4,
            },
        }
    }

    optimizer = my_registry.resolve(config)["optimizer"]
    assert optimizer.beta1 == 0.2
    assert optimizer.learn_rate == [0.001] * 4


def test_partials_from_config():
    """Test that functions registered with partial applications are handled
    correctly (e.g. initializers)."""
    numpy = pytest.importorskip("numpy")

    def uniform_init(
        shape: Tuple[int, ...], *, lo: float = -0.1, hi: float = 0.1
    ) -> List[float]:
        return numpy.random.uniform(lo, hi, shape).tolist()

    @my_registry.initializers("uniform_init.v1")
    def configure_uniform_init(
        *, lo: float = -0.1, hi: float = 0.1
    ) -> Callable[[List[float]], List[float]]:
        return partial(uniform_init, lo=lo, hi=hi)

    name = "uniform_init.v1"
    cfg = {"test": {"@initializers": name, "lo": -0.2}}
    func = my_registry.resolve(cfg)["test"]
    assert hasattr(func, "__call__")
    # The partial will still have lo as an arg, just with default
    assert len(inspect.signature(func).parameters) == 3
    # Make sure returned partial function has correct value set
    assert inspect.signature(func).parameters["lo"].default == -0.2
    # Actually call the function and verify
    assert numpy.asarray(func((2, 3))).shape == (2, 3)
    # Make sure validation still works
    bad_cfg = {"test": {"@initializers": name, "lo": [0.5]}}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve(bad_cfg)
    bad_cfg = {"test": {"@initializers": name, "lo": -0.2, "other": 10}}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve(bad_cfg)


def test_partials_from_config_nested():
    """Test that partial functions are passed correctly to other registered
    functions that consume them (e.g. initializers -> layers)."""

    def test_initializer(a: int, b: int = 1) -> int:
        return a * b

    @my_registry.initializers("test_initializer.v1")
    def configure_test_initializer(b: int = 1) -> Callable[[int], int]:
        return partial(test_initializer, b=b)

    @my_registry.layers("test_layer.v1")
    def test_layer(init: Callable[[int], int], c: int = 1) -> Callable[[int], int]:
        return lambda x: x + init(c)

    cfg = {
        "@layers": "test_layer.v1",
        "c": 5,
        "init": {"@initializers": "test_initializer.v1", "b": 10},
    }
    func = my_registry.resolve({"test": cfg})["test"]
    assert func(1) == 51
    assert func(100) == 150


@my_registry.schedules("schedule.v1")
def schedule1():
    while True:
        yield 10


@my_registry.optimizers("optimizer.v1")
def optimizer1(rate: Generator) -> Generator:
    return rate


@my_registry.optimizers("optimizer2.v1")
def optimizer2(schedules: Dict[str, Generator]) -> Generator:
    return schedules["rate"]


@pytest.mark.parametrize("config,expected", [
    ({"test": {"@schedules": "schedule.v1"}}, "unchanged"),
    ({"test": {"@optimizers": "optimizer2.v1", "schedules": {"rate": {"@schedules": "schedule.v1"}}}}, "unchanged")
])
def test_fill_validate_generator(config, expected):
    result = my_registry.fill(config, validate=True)
    if expected == "unchanged":
        assert result == config
    else:
        assert result != config
        assert result == expected


@pytest.mark.parametrize("config,paths", [
    ({"test": {"@schedules": "schedule.v1"}}, [("test",)]),
    ({"test": {"@optimizers": "optimizer.v1", "rate": {"@schedules": "schedule.v1"}}}, [("test",)]),
    ({"test": {"@optimizers": "optimizer2.v1", "schedules": {"rate": {"@schedules": "schedule.v1"}}}}, [("test",)])
])
def test_resolve_validate_generator(config, paths):
    result = my_registry.resolve(config, validate=True)
    for path in paths:
        node = result
        for x in path:
            node = node[x]
        assert isinstance(node, GeneratorType)


def test_handle_generic_type():
    """Test that validation can handle checks against arbitrary generic
    types in function argument annotations."""

    cfg = {"@cats": "generic_cat.v1", "cat": {"@cats": "int_cat.v1", "value_in": 3}}
    output = my_registry.resolve({"test": cfg})
    cat = output["test"]
    assert isinstance(cat, Cat)
    assert cat.value_in == 3
    assert cat.value_out is None
    assert cat.name == "generic_cat"


def test_fill_config_dict_return_type():
    """Test that a registered function returning a dict is handled correctly."""

    @my_registry.cats.register("catsie_with_dict.v1")
    def catsie_with_dict(evil: StrictBool) -> Dict[str, bool]:
        return {"not_evil": not evil}

    config = {"test": {"@cats": "catsie_with_dict.v1", "evil": False}, "foo": 10}
    result = my_registry.fill({"cfg": config}, validate=True)["cfg"]["test"]
    assert result["evil"] is False
    assert "not_evil" not in result
    result = my_registry.resolve({"cfg": config}, validate=True)["cfg"]["test"]
    assert result["not_evil"] is True


@my_registry.cats("catsie.with_alias")
def catsie_with_alias(validate: StrictBool = False):
    return validate

@my_registry.cats("catsie.with_model_alias")
def catsie_with_model_alias(model_config: str = "default"):
    return model_config


@pytest.mark.parametrize("config,filled,resolved", [
    (
        {"test": {"@cats": "catsie.with_alias", "validate": True}}, "unchanged", {"test": True}
    ),
    (
        {"test": {"@cats": "catsie.with_model_alias", "model_config": "hi"}}, "unchanged", {"test": "hi"}
    ),
    (
        {"test": {"@cats": "catsie.with_model_alias"}}, {"test": {"@cats": "catsie.with_model_alias", "model_config": "default"}}, {"test": "default"}
    ),
])
def test_reserved_aliases(config, filled, resolved):
    """Test that the auto-generated pydantic schemas auto-alias reserved
    attributes like "validate" that would otherwise cause NameError."""
    f = my_registry.fill(config)
    r = my_registry.resolve(config)
    if filled == "unchanged":
        assert f == config
    else:
        assert f != config
        assert f == filled
    if resolved == "unchanged":
        assert r == config
    else:
        assert r != config
        assert r == resolved


def test_config_validation_error_custom():
    class Schema(BaseModel):
        hello: int
        world: int

    config = {"hello": 1, "world": "hi!"}
    with pytest.raises(ConfigValidationError) as exc_info:
        my_registry.resolve(config, schema=Schema, validate=True)
    e1 = exc_info.value
    assert e1.title == "Config validation error"
    assert e1.desc is None
    assert not e1.parent
    assert e1.show_config is True
    assert len(e1.errors) == 1
    assert e1.errors[0]["loc"] == ("world",)
    assert e1.errors[0]["msg"] == "Input should be a valid integer, unable to parse string as an integer"
    assert e1.errors[0]["type"] == "int_parsing"
    assert e1.error_types == set(["int_parsing"])
    # Create a new error with overrides
    title = "Custom error"
    desc = "Some error description here"
    e2 = ConfigValidationError.from_error(e1, title=title, desc=desc, show_config=False)
    assert e2.errors == e1.errors
    assert e2.error_types == e1.error_types
    assert e2.title == title
    assert e2.desc == desc
    assert e2.show_config is False
    assert e1.text != e2.text


def test_config_fill_without_resolve():
    class BaseSchema(BaseModel):
        catsie: int

    config = {"catsie": {"@cats": "catsie.v1", "evil": False}}
    filled = my_registry.fill(config)
    resolved = my_registry.resolve(config)
    assert resolved["catsie"] == "meow"
    assert filled["catsie"]["cute"] is True
    with pytest.raises(ConfigValidationError):
        my_registry.resolve(config, schema=BaseSchema)
    filled2 = my_registry.fill(config, schema=BaseSchema)
    assert filled2["catsie"]["cute"] is True
    resolved = my_registry.resolve(filled2)
    assert resolved["catsie"] == "meow"

    # With unavailable function
    class BaseSchema2(BaseModel):
        catsie: Any
        other: int = 12

    config = {"catsie": {"@cats": "dog", "evil": False}}
    filled3 = my_registry.fill(config, schema=BaseSchema2)
    assert filled3["catsie"] == config["catsie"]
    assert filled3["other"] == 12
