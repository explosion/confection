"""Tests for validate_type covering all type branches."""

from pathlib import Path, PurePath
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    NewType,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import pytest

from confection.validation import (
    Field,
    FieldInfo,
    Schema,
    _validate_schema,
    create_schema,
    ensure_schema,
    validate_type,
)

# === None ===


def test_none():
    assert validate_type(None, type(None)) is None
    assert validate_type(42, type(None)) is not None


# === Annotated ===

try:
    from typing import Annotated
except ImportError:
    Annotated = None  # type: ignore


@pytest.mark.skipif(Annotated is None, reason="Annotated not available")
def test_annotated():
    assert validate_type(42, Annotated[int, "metadata"]) is None
    assert validate_type("hi", Annotated[int, "metadata"]) is not None


# === Union / Optional ===


def test_union():
    assert validate_type(42, Union[int, str]) is None
    assert validate_type("hi", Union[int, str]) is None
    assert validate_type(3.14, Union[int, str]) is not None


def test_optional():
    assert validate_type(None, Optional[int]) is None
    assert validate_type(42, Optional[int]) is None
    assert validate_type("hi", Optional[int]) is not None


def test_union_pipe_syntax():
    assert validate_type(42, int | str) is None
    assert validate_type("hi", int | str) is None
    assert validate_type(3.14, int | str) is not None


# === Literal ===


def test_literal():
    assert validate_type("a", Literal["a", "b"]) is None
    assert validate_type("b", Literal["a", "b"]) is None
    assert validate_type("c", Literal["a", "b"]) is not None
    assert validate_type(1, Literal[1, 2]) is None
    assert validate_type(3, Literal[1, 2]) is not None


# === NewType ===


def test_newtype():
    UserId = NewType("UserId", int)
    assert validate_type(42, UserId) is None
    assert validate_type("hi", UserId) is not None


# === TypeVar ===


def test_typevar_unbound():
    T = TypeVar("T")
    assert validate_type("anything", T) is None


def test_typevar_bound():
    T = TypeVar("T", bound=int)
    assert validate_type(42, T) is None
    assert validate_type("hi", T) is not None


def test_typevar_bound_union():
    T = TypeVar("T", bound=Union[str, int])
    assert validate_type("hello", T) is None
    assert validate_type(42, T) is None
    assert validate_type(3.14, T) is not None


def test_typevar_bound_union_negative():
    T = TypeVar("T", bound=Union[None, float])
    assert validate_type(None, T) is None
    assert validate_type(3.14, T) is None
    assert validate_type("hello", T) is not None


def test_typevar_constraints():
    T = TypeVar("T", int, str)
    assert validate_type(42, T) is None
    assert validate_type("hi", T) is None
    assert validate_type(3.14, T) is not None


# === Plain types ===


def test_bool():
    assert validate_type(True, bool) is None
    assert validate_type(1, bool) is not None


def test_int():
    assert validate_type(42, int) is None
    assert validate_type(True, int) is not None  # bool is not accepted as int
    assert validate_type("123", int) is None  # string coercion
    assert validate_type("abc", int) is not None


def test_float():
    assert validate_type(3.14, float) is None
    assert validate_type(42, float) is None  # int accepted for float
    assert validate_type(True, float) is not None
    assert validate_type("3.14", float) is None  # string coercion
    assert validate_type("abc", float) is not None


def test_str():
    assert validate_type("hello", str) is None
    assert validate_type(42, str) is not None


def test_path():
    assert validate_type(Path("/tmp"), Path) is None
    assert validate_type("/tmp/foo", Path) is None  # string coercion
    assert validate_type(PurePath("/tmp"), PurePath) is None
    assert validate_type("/tmp", PurePath) is None
    assert validate_type(42, Path) is not None


# === Callable ===


def test_callable():
    assert validate_type(lambda: None, Callable) is None
    assert validate_type(len, Callable) is None
    assert validate_type(42, Callable) is not None


# === List ===


def test_list():
    assert validate_type([1, 2, 3], list) is None
    assert validate_type([1, 2, 3], List[int]) is None
    assert validate_type([1, "a"], List[int]) is not None
    assert validate_type("not a list", list) is not None


# === Dict ===


def test_dict():
    assert validate_type({"a": 1}, dict) is None
    assert validate_type({"a": 1}, Dict[str, int]) is None
    assert validate_type({"a": "b"}, Dict[str, int]) is not None
    assert validate_type({1: "a"}, Dict[str, int]) is not None


# === Tuple ===


def test_tuple_bare():
    assert validate_type((1, 2), tuple) is None
    assert validate_type("hi", tuple) is not None


def test_tuple_fixed():
    assert validate_type((1, "a"), Tuple[int, str]) is None
    assert validate_type((1, 2), Tuple[int, str]) is not None
    assert validate_type((1,), Tuple[int, str]) is not None  # wrong length
    assert validate_type((1, "a", 3), Tuple[int, str]) is not None


def test_tuple_variable():
    assert validate_type((1, 2, 3), Tuple[int, ...]) is None
    assert validate_type((), Tuple[int, ...]) is None
    assert validate_type((1, "a"), Tuple[int, ...]) is not None


# === Set / FrozenSet ===


def test_set():
    assert validate_type({1, 2}, set) is None
    assert validate_type({1, 2}, Set[int]) is None
    assert validate_type({1, "a"}, Set[int]) is not None
    assert validate_type([1, 2], set) is not None


def test_frozenset():
    assert validate_type(frozenset([1, 2]), frozenset) is None
    assert validate_type(frozenset([1, 2]), FrozenSet[int]) is None
    assert validate_type(frozenset([1, "a"]), FrozenSet[int]) is not None
    assert validate_type({1, 2}, FrozenSet[int]) is not None


# === Sequence ===


def test_sequence():
    assert validate_type([1, 2], Sequence[int]) is None
    assert validate_type((1, 2), Sequence[int]) is None
    assert validate_type("hello", Sequence) is None  # str is a Sequence
    assert validate_type([1, "a"], Sequence[int]) is not None
    assert validate_type(42, Sequence) is not None


# === Iterable ===


def test_iterable():
    assert validate_type([1, 2], Iterable) is None
    assert validate_type("hi", Iterable) is None
    assert validate_type(42, Iterable) is not None


# === Mapping ===


def test_mapping():
    assert validate_type({"a": 1}, Mapping[str, int]) is None
    assert validate_type(42, Mapping) is not None


# === Iterator ===


def test_iterator():
    assert validate_type(iter([1, 2]), Iterator) is None
    assert validate_type(42, Iterator) is not None


# === Type[X] ===


def test_type():
    assert validate_type(int, Type[int]) is None
    assert validate_type(bool, Type[int]) is None  # subclass
    assert validate_type(str, Type[int]) is not None
    assert validate_type(42, Type[int]) is not None
    assert validate_type(int, Type[Any]) is None


# === Schema-as-dict validation ===


def test_schema_as_dict():
    MySchema = create_schema(
        "MySchema",
        __config__={"extra": "forbid"},
        x=(int, Field(...)),
        y=(str, Field("default")),
    )
    assert validate_type({"x": 1}, MySchema) is None
    assert validate_type({"x": 1, "y": "hi"}, MySchema) is None
    assert validate_type(42, MySchema) is not None  # not a dict


# === Pydantic hooks ===


def test_pydantic_core_schema_hook():
    """Types with __get_pydantic_core_schema__ get their validator called."""

    class MyType:
        @classmethod
        def __get_pydantic_core_schema__(cls, source_type, handler):
            def validate(v):
                if not isinstance(v, int):
                    raise ValueError("expected int")
                return v

            return {
                "type": "function-plain",
                "function": {"type": "no-info", "function": validate},
            }

    assert validate_type(42, MyType) is None
    assert validate_type("hi", MyType) is not None


def test_pydantic_v1_validators_hook():
    """Types with __get_validators__ get their validators called."""

    class MyType:
        @classmethod
        def __get_validators__(cls):
            def check_positive(v):
                if not isinstance(v, int) or v <= 0:
                    raise ValueError("must be positive int")
                return v

            yield check_positive

    assert validate_type(5, MyType) is None
    assert validate_type(-1, MyType) is not None
    assert validate_type("hi", MyType) is not None


# === Generator passthrough ===


def test_generator_passthrough():
    def gen():
        yield 1

    g = gen()
    assert validate_type(g, int) is None  # generators always pass
    assert next(g) == 1  # not consumed


# === _validate_schema ===


def test_validate_schema_extra_forbid():
    fields = {"x": FieldInfo(default=...)}
    fields["x"].annotation = int
    config = {"extra": "forbid"}
    errors = _validate_schema({"x": 1, "extra_key": 2}, fields, config, None)
    assert any("Extra inputs" in e["msg"] for e in errors)


def test_validate_schema_extra_allow():
    fields = {"x": FieldInfo(default=...)}
    fields["x"].annotation = int
    config = {"extra": "allow"}
    errors = _validate_schema({"x": 1, "extra_key": 2}, fields, config, None)
    assert not errors


def test_validate_schema_missing_required():
    fields = {"x": FieldInfo(default=...)}
    fields["x"].annotation = int
    config = {"extra": "forbid"}
    errors = _validate_schema({}, fields, config, None)
    assert any("required" in e["msg"].lower() for e in errors)


def test_validate_schema_alias():
    f = FieldInfo(default=..., alias="x_alias")
    f.annotation = int
    fields = {"x": f}
    config = {"extra": "forbid"}
    errors = _validate_schema({"x_alias": 1}, fields, config, None)
    assert not errors


def test_validate_schema_alias_generator():
    f = FieldInfo(default=...)
    f.annotation = int
    fields = {"my_field": f}
    config = {"extra": "forbid"}
    errors = _validate_schema(
        {"MY_FIELD": 1}, fields, config, lambda name: name.upper()
    )
    assert not errors


# === ensure_schema ===


def test_ensure_schema_passthrough():
    """Schema subclass passes through unchanged."""

    class MySchema(Schema):
        model_config = {"extra": "forbid"}

    assert ensure_schema(MySchema) is MySchema


def test_ensure_schema_from_our_schema():
    """create_schema output passes through."""
    s = create_schema("Test", __config__={"extra": "forbid"}, x=(int, Field(...)))
    assert ensure_schema(s) is s


# === model_dump ===


def test_model_dump():
    MySchema = create_schema(
        "Test",
        __config__={"extra": "forbid"},
        x=(int, Field(...)),
        y=(str, Field("default")),
    )
    instance = MySchema(x=1)
    assert instance.model_dump() == {"x": 1, "y": "default"}


# === Schema.from_function ===


def test_from_function_basic():
    def my_func(x: int, y: str = "hello"):
        pass

    schema = Schema.from_function(my_func)
    assert "x" in schema.model_fields
    assert "y" in schema.model_fields
    assert schema.model_fields["x"].annotation is int
    assert schema.model_fields["y"].default == "hello"


def test_from_function_var_positional():
    def my_func(*args: int):
        pass

    schema = Schema.from_function(my_func)
    assert "args" in schema.model_fields
    # Should be Sequence[int]
    ann = schema.model_fields["args"].annotation
    assert hasattr(ann, "__origin__")  # is generic


def test_from_function_var_keyword_skipped():
    def my_func(x: int, **kwargs):
        pass

    schema = Schema.from_function(my_func)
    assert "x" in schema.model_fields
    assert "kwargs" not in schema.model_fields


def test_from_function_no_annotations():
    def my_func(x, y=10):
        pass

    schema = Schema.from_function(my_func)
    assert schema.model_fields["x"].annotation is Any
    assert schema.model_fields["y"].default == 10
