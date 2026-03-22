"""Test registry.fill() default-filling from function signatures."""

import catalogue
import pytest

from confection import Config, registry
from confection._errors import ConfigValidationError

# --- Test registry setup ---


class _test_registry(registry):
    optimizers = catalogue.create("test_fill", "optimizers", entry_points=False)
    schedules = catalogue.create("test_fill", "schedules", entry_points=False)
    models = catalogue.create("test_fill", "models", entry_points=False)


@_test_registry.optimizers.register("Adam.v1")
def adam(learn_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999):
    return {"learn_rate": learn_rate, "beta1": beta1, "beta2": beta2}


@_test_registry.optimizers.register("SGD.v1")
def sgd(learn_rate: float = 0.01, momentum: float = 0.9):
    return {"learn_rate": learn_rate, "momentum": momentum}


@_test_registry.schedules.register("linear.v1")
def linear(start: float = 0.0, end: float = 1.0, steps: int = 100):
    return [start + (end - start) * i / steps for i in range(steps)]


@_test_registry.models.register("cnn.v1")
def cnn(width: int = 128, depth: int = 3, dropout: float = 0.1):
    return {"width": width, "depth": depth, "dropout": dropout}


@_test_registry.models.register("no_defaults.v1")
def no_defaults(width: int, depth: int):
    return {"width": width, "depth": depth}


# --- Tests ---


def test_fill_basic():
    """Fill adds missing defaults from the registered function."""
    config = Config().from_str(
        """
[optimizer]
@optimizers = "Adam.v1"
learn_rate = 0.01
""",
        interpolate=False,
    )
    filled = _test_registry.fill(config)
    assert filled["optimizer"]["learn_rate"] == 0.01
    assert filled["optimizer"]["beta1"] == 0.9
    assert filled["optimizer"]["beta2"] == 0.999


def test_fill_preserves_provided():
    """Values explicitly provided should not be overwritten."""
    config = Config().from_str(
        """
[optimizer]
@optimizers = "Adam.v1"
learn_rate = 0.05
beta1 = 0.8
beta2 = 0.99
""",
        interpolate=False,
    )
    filled = _test_registry.fill(config)
    assert filled["optimizer"]["learn_rate"] == 0.05
    assert filled["optimizer"]["beta1"] == 0.8
    assert filled["optimizer"]["beta2"] == 0.99


def test_fill_all_defaults():
    """When no args are provided, all defaults should be filled."""
    config = Config().from_str(
        """
[optimizer]
@optimizers = "Adam.v1"
""",
        interpolate=False,
    )
    filled = _test_registry.fill(config)
    assert filled["optimizer"]["learn_rate"] == 0.001
    assert filled["optimizer"]["beta1"] == 0.9
    assert filled["optimizer"]["beta2"] == 0.999


def test_fill_no_defaults():
    """Function with no defaults should not add anything."""
    config = Config().from_str(
        """
[model]
@models = "no_defaults.v1"
width = 64
depth = 2
""",
        interpolate=False,
    )
    filled = _test_registry.fill(config)
    assert filled["model"]["width"] == 64
    assert filled["model"]["depth"] == 2


def test_fill_nested_promise():
    """Defaults are filled recursively into nested promises."""
    config = Config().from_str(
        """
[optimizer]
@optimizers = "SGD.v1"

[optimizer.learn_rate]
@schedules = "linear.v1"
end = 0.5
""",
        interpolate=False,
    )
    filled = _test_registry.fill(config)
    # Outer promise filled
    assert filled["optimizer"]["momentum"] == 0.9
    # Inner promise filled
    lr = filled["optimizer"]["learn_rate"]
    assert lr["start"] == 0.0
    assert lr["end"] == 0.5  # provided, not overwritten
    assert lr["steps"] == 100


def test_fill_non_promise_sections():
    """Non-promise sections are passed through unchanged."""
    config = Config().from_str(
        """
[training]
epochs = 10

[optimizer]
@optimizers = "Adam.v1"
""",
        interpolate=False,
    )
    filled = _test_registry.fill(config)
    assert filled["training"]["epochs"] == 10
    assert filled["optimizer"]["beta1"] == 0.9


def test_fill_preserves_registry_key():
    """The @registry key should be preserved in the filled config."""
    config = Config().from_str(
        """
[optimizer]
@optimizers = "Adam.v1"
""",
        interpolate=False,
    )
    filled = _test_registry.fill(config)
    assert filled["optimizer"]["@optimizers"] == "Adam.v1"


def test_fill_returns_config():
    """fill() should return a Config object."""
    config = Config().from_str(
        """
[optimizer]
@optimizers = "Adam.v1"
""",
        interpolate=False,
    )
    filled = _test_registry.fill(config)
    assert isinstance(filled, Config)


def test_fill_top_level_promise_raises():
    """Top-level config can't be a promise."""
    with pytest.raises(ConfigValidationError):
        _test_registry.fill({"@optimizers": "Adam.v1"})


def test_fill_with_overrides():
    """Overrides should be applied before filling defaults."""
    config = Config().from_str(
        """
[optimizer]
@optimizers = "Adam.v1"
learn_rate = 0.01
""",
        interpolate=False,
    )
    filled = _test_registry.fill(
        config, overrides={"optimizer.learn_rate": 0.1}, interpolate=True
    )
    assert filled["optimizer"]["learn_rate"] == 0.1
    assert filled["optimizer"]["beta1"] == 0.9


def test_fill_promise_rejects_extra_args():
    """Extra args not in the function signature should be rejected."""
    config = Config(
        {
            "optimizer": {
                "@optimizers": "Adam.v1",
                "learn_rate": 0.01,
                "metadata": {"nested_model": {"@models": "cnn.v1"}},
            }
        }
    )
    with pytest.raises(ConfigValidationError):
        _test_registry.fill(config)


def test_fill_nested_non_promise_dict():
    """Non-promise dicts nested under a section with promises should recurse."""
    config = Config().from_str(
        """
[section]

[section.sub]
x = 1

[section.model]
@models = "cnn.v1"
width = 64
""",
        interpolate=False,
    )
    filled = _test_registry.fill(config)
    assert filled["section"]["sub"] == {"x": 1}
    assert filled["section"]["model"]["depth"] == 3


def test_fill_skips_unknown_functions():
    """fill() should skip promises referencing unknown functions, not crash."""
    config = Config(
        {
            "section": {
                "known": {"@optimizers": "Adam.v1"},
                "unknown": {"@optimizers": "nonexistent.v1"},
            }
        }
    )
    filled = _test_registry.fill(config)
    # Known function gets defaults filled
    assert filled["section"]["known"]["beta1"] == 0.9
    # Unknown function passes through unchanged
    assert filled["section"]["unknown"] == {"@optimizers": "nonexistent.v1"}


def test_fill_strips_extra_fields_with_schema():
    """fill() with a schema that has extra='forbid' should strip unknown keys."""
    from confection.validation import Schema

    class TrainingSchema(Schema):
        model_config = {"extra": "forbid"}
        patience: int = 10
        dropout: float = 0.2

    class MySchema(Schema):
        model_config = {"extra": "forbid"}
        training: TrainingSchema

    config = Config({"training": {"patience": 5, "extra_field": "hello"}})
    filled = _test_registry.fill(config, schema=MySchema)
    assert filled["training"]["patience"] == 5
    assert filled["training"]["dropout"] == 0.2
    assert "extra_field" not in filled["training"]


def test_fill_positional_args():
    """Functions with *args should have positional args filled under '*'."""

    @_test_registry.models.register("chain.v1")
    def chain(*layers):
        return list(layers)

    config = Config(
        {
            "model": {
                "@models": "chain.v1",
                "*": {
                    "layer1": {"@models": "cnn.v1", "width": 64},
                    "layer2": {"@models": "cnn.v1"},
                },
            }
        }
    )
    filled = _test_registry.fill(config)
    # Positional args should pass through, nested promise defaults filled
    assert filled["model"]["*"]["layer2"]["depth"] == 3
    assert filled["model"]["*"]["layer1"]["width"] == 64


def test_fill_positional_args_not_flagged_as_unexpected():
    """The '*' key should not be flagged as an unexpected argument."""
    config = Config(
        {
            "model": {
                "@models": "chain.v1",
                "*": {"a": {"@models": "cnn.v1"}},
            }
        }
    )
    # Should not raise ConfigValidationError about unexpected '*'
    filled = _test_registry.fill(config)
    assert "*" in filled["model"]


def test_fill_with_interpolation():
    """fill() with interpolate=True should resolve variables."""
    config = Config().from_str(
        """
[hyper]
lr = 0.01

[optimizer]
@optimizers = "Adam.v1"
learn_rate = ${hyper.lr}
""",
        interpolate=False,
    )
    filled = _test_registry.fill(config, interpolate=True)
    assert filled["optimizer"]["learn_rate"] == 0.01
    assert filled["optimizer"]["beta1"] == 0.9
