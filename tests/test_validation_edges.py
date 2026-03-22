"""Tests for edge cases in validation.py."""

from typing import Optional

import pytest

from confection.validation import (
    Field,
    FieldInfo,
    Schema,
    _is_pydantic_model,
    _pydantic_instance_to_dict,
    _validate_schema,
    create_schema,
    ensure_schema,
    validate_type,
)

# --- Schema with FieldInfo as class default ---


class SchemaWithFieldInfo(Schema):
    x: int = Field(42)
    y: str = Field("hello")


def test_field_info_as_class_default():
    assert SchemaWithFieldInfo.model_fields["x"].default == 42
    assert SchemaWithFieldInfo.model_fields["y"].default == "hello"


def test_field_info_required():
    class RequiredSchema(Schema):
        x: int = Field(...)

    assert RequiredSchema.model_fields["x"].is_required()


# --- Schema field aliases ---


class AliasedSchema(Schema):
    model_config = {"extra": "forbid"}
    x: int = Field(default=..., alias="x_alias")


def test_model_validate_with_alias():
    result = AliasedSchema.model_validate({"x_alias": 42})
    assert result.x_alias == 42


def test_model_validate_alias_fills_default():
    class AliasDefault(Schema):
        x: int = Field(default=10, alias="x_alias")

    result = AliasDefault.model_validate({})
    assert result.x_alias == 10


# --- Schema with alias_generator ---


class AliasGenSchema(Schema):
    model_config = {"extra": "forbid", "alias_generator": lambda name: name.upper()}
    my_field: int


def test_model_validate_with_alias_generator():
    result = AliasGenSchema.model_validate({"MY_FIELD": 42})
    assert result.MY_FIELD == 42


def test_model_validate_alias_generator_fills_default():
    class AliasGenDefault(Schema):
        model_config = {"alias_generator": lambda name: name.upper()}
        x: int = 99

    result = AliasGenDefault.model_validate({})
    assert result.X == 99


# --- model_dump with nested Schema ---


class Inner(Schema):
    a: int


class Outer(Schema):
    inner: Inner
    b: str = "hi"


def test_model_dump_nested():
    instance = Outer(inner=Inner(a=1))
    dumped = instance.model_dump()
    assert dumped == {"inner": {"a": 1}, "b": "hi"}


# --- create_schema edge cases ---


def test_create_schema_no_config():
    s = create_schema("NoConfig", x=(int, Field(...)))
    assert s.model_config == {"extra": "allow"}


def test_create_schema_bad_field():
    with pytest.raises(ValueError, match="must be"):
        create_schema("Bad", x="not a tuple")


def test_create_schema_with_alias_generator():
    s = create_schema(
        "AliasGen",
        __config__={"extra": "forbid", "alias_generator": lambda n: n.upper()},
        my_field=(int, Field(...)),
    )
    assert s.model_fields["my_field"].alias == "MY_FIELD"


def test_create_schema_plain_default():
    """Non-FieldInfo second element in tuple gets wrapped."""
    s = create_schema("PlainDefault", x=(int, 42))
    assert s.model_fields["x"].default == 42


# --- resolve_type_hints fallback ---


def test_from_function_unresolvable_forward_ref():
    """Forward refs that can't be resolved fall back to raw annotations."""

    # Create a function with an annotation that can't be resolved
    def func(x: "NonExistentType") -> None:  # noqa: F821
        pass

    schema = Schema.from_function(func)
    # Should still create the schema, just with Any or the raw annotation
    assert "x" in schema.model_fields


# --- _error_type_for branches ---


def test_validate_type_str_error():
    err = validate_type(42, str)
    assert err is not None


def test_validate_type_float_error():
    err = validate_type("abc", float)
    assert err is not None


def test_validate_type_bool_error():
    err = validate_type(1, bool)
    assert err is not None


def test_validate_type_complex_error():
    """Non-primitive type should give value_error."""
    from typing import List

    err = validate_type("not a list", List[int])
    assert err is not None


# --- _validate_schema error types ---


def test_validate_schema_str_error_type():
    f = FieldInfo(default=...)
    f.annotation = str
    errors = _validate_schema({"x": 42}, {"x": f}, {"extra": "allow"}, None)
    assert errors[0]["type"] == "string_type"


def test_validate_schema_float_error_type():
    f = FieldInfo(default=...)
    f.annotation = float
    errors = _validate_schema({"x": "abc"}, {"x": f}, {"extra": "allow"}, None)
    assert errors[0]["type"] == "float_parsing"


def test_validate_schema_bool_error_type():
    f = FieldInfo(default=...)
    f.annotation = bool
    errors = _validate_schema({"x": 1}, {"x": f}, {"extra": "allow"}, None)
    assert errors[0]["type"] == "bool_type"


def test_validate_schema_positive_int_error_type():
    from confection.validation import PositiveInt

    f = FieldInfo(default=...)
    f.annotation = PositiveInt
    errors = _validate_schema({"x": -1}, {"x": f}, {"extra": "allow"}, None)
    assert errors[0]["type"] == "int_parsing"


def test_validate_schema_strict_float_error_type():
    from confection.validation import StrictFloat

    f = FieldInfo(default=...)
    f.annotation = StrictFloat
    errors = _validate_schema({"x": "abc"}, {"x": f}, {"extra": "allow"}, None)
    assert errors[0]["type"] == "float_parsing"


def test_validate_schema_generic_error_type():
    from typing import List

    f = FieldInfo(default=...)
    f.annotation = List[int]
    errors = _validate_schema({"x": "nope"}, {"x": f}, {"extra": "allow"}, None)
    assert errors[0]["type"] == "value_error"


# --- _is_pydantic_model edge cases ---


def test_is_pydantic_model_non_type():
    assert _is_pydantic_model("not a type") is False


def test_is_pydantic_model_schema_subclass():
    class MySchema(Schema):
        x: int

    assert _is_pydantic_model(MySchema) is False


# --- _pydantic_instance_to_dict ---


def test_pydantic_instance_to_dict_v2():
    import pydantic

    class M(pydantic.BaseModel):
        x: int = 1

    assert _pydantic_instance_to_dict(M()) == {"x": 1}


def test_pydantic_instance_to_dict_v1():
    from pydantic.v1 import BaseModel

    class M(BaseModel):
        x: int = 1

    assert _pydantic_instance_to_dict(M()) == {"x": 1}


def test_pydantic_instance_to_dict_plain():
    assert _pydantic_instance_to_dict(42) == 42


# --- ensure_schema passthrough for non-pydantic ---


def test_ensure_schema_plain_class():
    class NotAModel:
        pass

    assert ensure_schema(NotAModel) is NotAModel


# --- v1 allow_none (Optional) ---


def test_v1_optional_field():
    from pydantic.v1 import BaseModel as V1Model

    class M(V1Model):
        x: Optional[int] = None

    converted = ensure_schema(M)
    # Should accept None
    result = converted.model_validate({"x": None})
    assert result.x is None


# --- v2 model with pydantic instance default ---


def test_v2_pydantic_instance_default():
    import pydantic

    class Inner(pydantic.BaseModel):
        x: int = 1

    class Outer(pydantic.BaseModel):
        inner: Inner = Inner()

    converted = ensure_schema(Outer)
    # Default should be converted from pydantic instance to dict
    assert converted.model_fields["inner"].default == {"x": 1}


# --- v1 model with pydantic instance default ---


def test_v1_pydantic_instance_default():
    from pydantic.v1 import BaseModel as V1Model

    class Inner(V1Model):
        x: int = 1

    class Outer(V1Model):
        inner: Inner = Inner()

    converted = ensure_schema(Outer)
    assert converted.model_fields["inner"].default == {"x": 1}


# --- Schema inheritance ---


class BaseSchema(Schema):
    x: int = 1


class DerivedSchema(BaseSchema):
    y: str = "hello"


def test_schema_inheritance():
    assert "x" in DerivedSchema.model_fields
    assert "y" in DerivedSchema.model_fields
    assert DerivedSchema.model_fields["x"].default == 1


def test_schema_inheritance_override():
    class Override(BaseSchema):
        x: int = 99

    assert Override.model_fields["x"].default == 99
