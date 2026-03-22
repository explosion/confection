"""Test registry.resolve() and the Promise lifecycle."""
import catalogue
import pytest
from typing import Callable, List
from functools import partial

from confection import Config, registry
from confection._errors import ConfigValidationError
from confection._registry import (
    Promise,
    insert_promises,
    resolve_promises,
    fix_positionals,
    _is_config_section,
    alias_generator,
    _deep_copy_with_uncopyable,
)
from confection._constants import ARGS_FIELD_ALIAS, RESERVED_FIELDS_REVERSE


# --- Test registry setup ---

class _test_registry(registry):
    namespace = "test_resolve"
    cats = catalogue.create(namespace, "cats", entry_points=False)
    optimizers = catalogue.create(namespace, "optimizers", entry_points=False)
    schedules = catalogue.create(namespace, "schedules", entry_points=False)
    layers = catalogue.create(namespace, "layers", entry_points=False)


@_test_registry.cats.register("catsie.v1")
def catsie(evil: bool, cute: bool = True) -> str:
    if evil:
        return "scratch!"
    return "meow"


@_test_registry.optimizers.register("Adam.v1")
def adam(learn_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999):
    return {"learn_rate": learn_rate, "beta1": beta1, "beta2": beta2}


@_test_registry.schedules.register("decay.v1")
def decay(base_rate: float, repeat: int) -> List[float]:
    return [base_rate] * repeat


@_test_registry.optimizers.register("cool.v1")
def cool_optimizer(learn_rate, beta1: float = 0.9):
    return {"learn_rate": learn_rate, "beta1": beta1}


@_test_registry.layers.register("linear.v1")
def linear_layer(width: int, init: Callable = lambda: None):
    return {"width": width, "init": init}


# --- resolve basic ---


def test_resolve_simple():
    config = Config({"cat": {"@cats": "catsie.v1", "evil": True}})
    result = _test_registry.resolve(config)
    assert result["cat"] == "scratch!"


def test_resolve_with_defaults():
    config = Config({"cat": {"@cats": "catsie.v1", "evil": False}})
    result = _test_registry.resolve(config)
    assert result["cat"] == "meow"


def test_resolve_nested_promise():
    """A promise arg can be another promise."""
    config = Config({
        "optimizer": {
            "@optimizers": "cool.v1",
            "learn_rate": {
                "@schedules": "decay.v1",
                "base_rate": 0.001,
                "repeat": 4,
            },
        }
    })
    result = _test_registry.resolve(config)
    assert result["optimizer"]["learn_rate"] == [0.001] * 4
    assert result["optimizer"]["beta1"] == 0.9


def test_resolve_non_promise_passthrough():
    config = Config({"training": {"epochs": 10}, "cat": {"@cats": "catsie.v1", "evil": True}})
    result = _test_registry.resolve(config)
    assert result["training"] == {"epochs": 10}
    assert result["cat"] == "scratch!"


def test_resolve_from_str():
    config = Config().from_str("""
[optimizer]
@optimizers = "Adam.v1"
learn_rate = 0.01
""")
    result = _test_registry.resolve(config)
    assert result["optimizer"]["learn_rate"] == 0.01
    assert result["optimizer"]["beta1"] == 0.9


# --- Promise class ---


def test_promise_from_dict():
    values = {"@cats": "catsie.v1", "evil": True, "cute": False}
    promise = Promise.from_dict(_test_registry, values)
    assert promise.registry == "cats"
    assert promise.name == "catsie.v1"
    assert promise.kwargs == {"evil": True, "cute": False}


def test_promise_resolve():
    values = {"@cats": "catsie.v1", "evil": True}
    promise = Promise.from_dict(_test_registry, values)
    assert promise.resolve() == "scratch!"


def test_promise_return_type():
    values = {"@cats": "catsie.v1", "evil": True}
    promise = Promise.from_dict(_test_registry, values)
    assert promise.return_type is str


# --- insert_promises ---


def test_insert_promises():
    config = {"cat": {"@cats": "catsie.v1", "evil": True}, "x": 1}
    result = insert_promises(_test_registry, config, resolve=True)
    assert isinstance(result["cat"], Promise)
    assert result["x"] == 1


def test_insert_promises_nested():
    config = {
        "section": {
            "cat": {"@cats": "catsie.v1", "evil": False},
            "val": 42,
        }
    }
    result = insert_promises(_test_registry, config, resolve=True)
    assert isinstance(result["section"]["cat"], Promise)
    assert result["section"]["val"] == 42


# --- resolve_promises ---


def test_resolve_promises():
    config = {"cat": {"@cats": "catsie.v1", "evil": True}, "x": 1}
    promised = insert_promises(_test_registry, config, resolve=True)
    resolved = resolve_promises(promised)
    assert resolved["cat"] == "scratch!"
    assert resolved["x"] == 1


# --- fix_positionals ---


def test_fix_positionals_dict_to_tuple():
    config = {"*": {"0": "a", "1": "b"}, "x": 1}
    result = fix_positionals(config)
    assert result["*"] == ("a", "b")
    assert result["x"] == 1


def test_fix_positionals_nested():
    config = {"section": {"*": {"0": "a"}, "val": 1}}
    result = fix_positionals(config)
    assert result["section"]["*"] == ("a",)


def test_fix_positionals_list():
    config = [{"*": {"0": 1}}, {"x": 2}]
    result = fix_positionals(config)
    assert result[0]["*"] == (1,)
    assert result[1] == {"x": 2}


def test_fix_positionals_tuple():
    config = ({"*": {"0": 1}},)
    result = fix_positionals(config)
    assert result[0]["*"] == (1,)
    assert isinstance(result, tuple)


def test_fix_positionals_scalar():
    assert fix_positionals(42) == 42
    assert fix_positionals("hello") == "hello"


# --- _is_config_section ---


def test_is_config_section():
    assert _is_config_section({"a": 1, "b": 2})
    assert not _is_config_section([1, 2])
    assert not _is_config_section("hello")
    assert _is_config_section({})


# --- alias_generator ---


def test_alias_generator_args_field():
    assert alias_generator(ARGS_FIELD_ALIAS) == "*"


def test_alias_generator_reserved():
    for alias, original in RESERVED_FIELDS_REVERSE.items():
        assert alias_generator(alias) == original


def test_alias_generator_normal():
    assert alias_generator("some_field") == "some_field"


# --- _deep_copy_with_uncopyable ---


def test_deep_copy_dict():
    original = {"a": [1, 2], "b": {"c": 3}}
    copied = _deep_copy_with_uncopyable(original)
    assert copied == original
    copied["a"].append(3)
    assert original["a"] == [1, 2]


def test_deep_copy_list():
    original = [{"a": 1}, {"b": 2}]
    copied = _deep_copy_with_uncopyable(original)
    assert copied == original
    copied[0]["a"] = 99
    assert original[0]["a"] == 1


def test_deep_copy_tuple():
    original = ({"a": 1},)
    copied = _deep_copy_with_uncopyable(original)
    assert copied == original
    assert isinstance(copied, tuple)


def test_deep_copy_scalar():
    assert _deep_copy_with_uncopyable(42) == 42
    assert _deep_copy_with_uncopyable("hello") == "hello"


def test_deep_copy_generator():
    """Generators can't be deepcopied — should pass through."""
    def gen():
        yield 1
    g = gen()
    copied = _deep_copy_with_uncopyable(g)
    assert copied is g  # same object, not copied


def test_deep_copy_memo():
    """Shared references should be preserved."""
    shared = [1, 2, 3]
    original = {"a": shared, "b": shared}
    copied = _deep_copy_with_uncopyable(original)
    assert copied["a"] is copied["b"]  # still shared
    assert copied["a"] is not original["a"]  # but different from original


# --- registry.has / registry.get ---


def test_registry_has():
    assert _test_registry.has("cats", "catsie.v1")
    assert not _test_registry.has("cats", "nonexistent")
    assert not _test_registry.has("nonexistent_registry", "anything")


def test_registry_get():
    func = _test_registry.get("cats", "catsie.v1")
    assert func is catsie


def test_registry_get_unknown_registry():
    with pytest.raises(ValueError, match="Unknown registry"):
        _test_registry.get("nonexistent", "anything")


def test_registry_get_unknown_func():
    with pytest.raises((ValueError, catalogue.RegistryError)):
        _test_registry.get("cats", "nonexistent")


# --- get_constructor / parse_args ---


def test_get_constructor():
    assert _test_registry.get_constructor({"@cats": "catsie.v1", "evil": True}) == ("cats", "catsie.v1")


def test_get_constructor_multiple_refs():
    with pytest.raises(ConfigValidationError):
        _test_registry.get_constructor({"@cats": "catsie.v1", "@dogs": "doggo.v1"})


def test_parse_args():
    args, kwargs = _test_registry.parse_args({"@cats": "catsie.v1", "evil": True, "cute": False})
    assert args == []
    assert kwargs == {"evil": True, "cute": False}


def test_parse_args_with_positionals():
    args, kwargs = _test_registry.parse_args({"@cats": "catsie.v1", "*": [1, 2], "evil": True})
    assert args == [1, 2]
    assert kwargs == {"evil": True}


# --- resolve top-level promise raises ---


def test_resolve_top_level_promise():
    with pytest.raises(ConfigValidationError):
        _test_registry.resolve({"@cats": "catsie.v1", "evil": True})
