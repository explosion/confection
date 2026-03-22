"""Tests for typechecker edge cases."""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any

from confection.typechecker import check_type
from confection.validation import Field, Schema

# --- custom_handlers ---


def test_custom_handler():
    def handle_str(value, annotation, handlers, ctx):
        return value == "magic"

    assert check_type("magic", int, custom_handlers={str: handle_str})
    assert not check_type("other", int, custom_handlers={str: handle_str})


# --- Annotated with Strict() ---

try:
    from pydantic import Strict
except ImportError:

    class Strict:  # type: ignore
        strict = True


def test_strict_int():
    assert check_type(42, Annotated[int, Strict()])
    assert not check_type(True, Annotated[int, Strict()])
    assert not check_type("42", Annotated[int, Strict()])


def test_strict_float():
    assert check_type(3.14, Annotated[float, Strict()])
    assert not check_type(42, Annotated[float, Strict()])


def test_strict_str():
    assert check_type("hi", Annotated[str, Strict()])
    assert not check_type(42, Annotated[str, Strict()])


def test_strict_bool():
    assert check_type(True, Annotated[bool, Strict()])
    assert not check_type(1, Annotated[bool, Strict()])


def test_strict_other():
    assert check_type([1], Annotated[list, Strict()])


# --- String forward reference ---


def test_string_annotation_accepts_anything():
    assert check_type(42, "SomeForwardRef")
    assert check_type("hi", "SomeForwardRef")


# --- String enum ---


class Color(str, Enum):
    RED = "red"
    GREEN = "green"


def test_string_enum():
    assert check_type("red", Color)
    assert not check_type("blue", Color)


# --- float with non-numeric ---


def test_float_rejects_list():
    assert not check_type([], float)


# --- Type[X] with non-class generic arg ---


def test_type_non_class_arg():
    """Type[X] where X causes TypeError in issubclass should accept."""
    from typing import Type

    # This would raise TypeError in issubclass
    assert check_type(int, Type[Any])


# --- Dataclass with dict ---


@dataclass
class Point:
    x: int
    y: int


def test_dataclass_accepts_dict():
    assert check_type({"x": 1, "y": 2}, Point)


def test_dataclass_dict_validates_fields():
    assert not check_type({"x": "bad", "y": 2}, Point)


def test_dataclass_with_dataclass_value():
    p = Point(x=1, y=2)
    assert check_type(p, Point)


def test_dataclass_value_validates_fields():
    p = Point(x=1, y=2)
    # Manually set bad type to test field validation
    p.x = "bad"  # type: ignore
    assert not check_type(p, Point)


# --- Schema with aliased field in decompose ---


class AliasedSchema(Schema):
    x: int = Field(default=..., alias="x_alias")


def test_schema_decompose_alias():
    assert check_type({"x_alias": 42}, AliasedSchema)
    assert not check_type({"x_alias": "bad"}, AliasedSchema)


# --- Fall-through return False ---


def test_unknown_annotation():
    """An annotation that doesn't match any branch returns False."""
    # A module object isn't a type, not a TypeVar, not a string, etc.
    import os

    assert not check_type(42, os)


# --- Pydantic v1 validator edges ---


def test_pydantic_v1_validator_many_params():
    """Validators with >2 params are skipped."""

    class MyType:
        @classmethod
        def __get_validators__(cls):
            def three_params(v, field, config):
                return v

            yield three_params

    # Should still pass (validator is skipped)
    assert check_type(42, MyType)


def test_pydantic_v1_validator_sig_error():
    """Validators with un-inspectable signatures default to 1 param."""

    class MyType:
        @classmethod
        def __get_validators__(cls):
            yield len  # len's signature can be inspected, but let's test the path

    # len(42) will raise TypeError
    assert not check_type(42, MyType)


# --- Pydantic v2 match edges ---


def test_pydantic_v2_no_function_key():
    """Schema without 'function' key in result."""

    class MyType:
        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            return {"type": "any"}  # no function key

    assert not check_type(42, MyType)


def test_pydantic_v2_isinstance_shortcut():
    """If value is already an instance, skip validator."""

    class MyType:
        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            return {"type": "any"}

    assert check_type(MyType(), MyType)


# --- Parameter.empty ---


def test_parameter_empty():
    import inspect

    assert check_type(42, inspect.Parameter.empty)
    assert check_type("anything", inspect.Parameter.empty)


# --- Type[Union[...]] ---


def test_type_union_arg():
    """Type[Union[int, str]] — issubclass raises TypeError, should accept."""
    from typing import Type, Union

    assert check_type(int, Type[Union[int, str]])


# --- _pydantic_v1_match isinstance shortcut ---


def test_pydantic_v1_isinstance_shortcut():
    class MyType:
        @classmethod
        def __get_validators__(cls):
            yield lambda v: v

    assert check_type(MyType(), MyType)


# --- _pydantic_v1_match uninspectable signature ---


def test_pydantic_v1_uninspectable_sig():
    """Validator with uninspectable signature defaults to 1 param."""

    class MyType:
        @classmethod
        def __get_validators__(cls):
            # A builtin with no inspectable sig in some Python versions
            class Uninspectable:
                def __call__(self, v):
                    if not isinstance(v, int):
                        raise ValueError
                    return v

                # Make signature() raise
                __signature__ = property(lambda self: (_ for _ in ()).throw(ValueError))

            yield Uninspectable()

    assert check_type(42, MyType)
    assert not check_type("hi", MyType)


# --- ctx default ---


def test_check_type_creates_default_ctx():
    """check_type works without explicit ctx."""
    assert check_type(42, int)
    assert not check_type("hi", int)
