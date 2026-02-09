"""Focused tests to understand Pydantic v2's generator/iterator consumption behavior."""

from typing import Generator, Iterable, Iterator, List, Union

import pytest
from pydantic import BaseModel, Field, create_model


def make_generator():
    """Create a simple generator for testing."""
    yield 0.1
    yield 0.2
    yield 0.3


def assert_not_consumed(gen, expected_first=0.1):
    """Assert that a generator has not been consumed."""
    val = next(gen)
    assert val == expected_first, f"Expected {expected_first}, got {val}"


def assert_consumed(gen):
    """Assert that a generator has been consumed."""
    with pytest.raises(StopIteration):
        next(gen)


class TestPydanticGeneratorBehavior:
    """Test how Pydantic handles generators with different type annotations."""

    def test_generator_annotation(self):
        """Generator annotation alone - not consumed."""
        gen = make_generator()
        Model = create_model("M", field=(Generator, ...))
        result = Model.model_validate({"field": gen})
        assert_not_consumed(result.field)

    def test_iterator_annotation_not_supported(self):
        """Iterator annotation alone - NOT SUPPORTED by Pydantic without arbitrary_types_allowed."""
        import pydantic

        gen = make_generator()
        with pytest.raises(pydantic.errors.PydanticSchemaGenerationError):
            create_model("M", field=(Iterator, ...))

    def test_iterable_annotation(self):
        """Iterable annotation alone - not consumed."""
        gen = make_generator()
        Model = create_model("M", field=(Iterable, ...))
        result = Model.model_validate({"field": gen})
        assert_not_consumed(result.field)

    def test_list_annotation(self):
        """List annotation - CONSUMED (converted to list)."""
        gen = make_generator()
        Model = create_model("M", field=(List[float], ...))
        result = Model.model_validate({"field": gen})
        # List consumes and converts
        assert result.field == [0.1, 0.2, 0.3]
        assert_consumed(gen)


class TestPydanticUnionBehavior:
    """Test how Pydantic handles generators in Union types.

    KEY FINDING: Order matters! When a Sequence type (List) comes before
    an iterator type, the generator gets consumed.
    """

    def test_union_generator_first(self):
        """Union with Generator listed first - NOT consumed."""
        gen = make_generator()
        Model = create_model("M", field=(Union[Generator, List[float], float], ...))
        result = Model.model_validate({"field": gen})
        assert_not_consumed(result.field)

    def test_union_generator_last(self):
        """Union with Generator listed last - CONSUMED (List tried first)."""
        gen = make_generator()
        Model = create_model("M", field=(Union[float, List[float], Generator], ...))
        result = Model.model_validate({"field": gen})
        # Generator is consumed because List[float] is tried first
        assert_consumed(gen)

    def test_union_iterator_not_supported(self):
        """Iterator in Union - NOT SUPPORTED by Pydantic."""
        import pydantic

        with pytest.raises(pydantic.errors.PydanticSchemaGenerationError):
            create_model("M", field=(Union[Iterator, List[float], float], ...))

    def test_union_iterable_first(self):
        """Union with Iterable listed first - NOT consumed."""
        gen = make_generator()
        Model = create_model("M", field=(Union[Iterable, List[float], float], ...))
        result = Model.model_validate({"field": gen})
        assert_not_consumed(result.field)

    def test_union_iterable_last(self):
        """Union with Iterable listed last - CONSUMED (List tried first)."""
        gen = make_generator()
        Model = create_model("M", field=(Union[float, List[float], Iterable], ...))
        result = Model.model_validate({"field": gen})
        # Generator is consumed because List[float] is tried first
        assert_consumed(gen)


class TestPydanticParameterizedTypes:
    """Test parameterized generator/iterator types."""

    def test_generator_parameterized(self):
        """Generator[YieldType, SendType, ReturnType] - not consumed."""
        gen = make_generator()
        Model = create_model("M", field=(Generator[float, None, None], ...))
        result = Model.model_validate({"field": gen})
        assert_not_consumed(result.field)

    def test_iterator_parameterized_not_supported(self):
        """Iterator[YieldType] - NOT SUPPORTED by Pydantic."""
        import pydantic

        with pytest.raises(pydantic.errors.PydanticSchemaGenerationError):
            create_model("M", field=(Iterator[float], ...))

    def test_iterable_parameterized(self):
        """Iterable[YieldType] - not consumed."""
        gen = make_generator()
        Model = create_model("M", field=(Iterable[float], ...))
        result = Model.model_validate({"field": gen})
        assert_not_consumed(result.field)


class TestPydanticUnionParameterized:
    """Test parameterized types in Unions."""

    def test_union_generator_parameterized_last(self):
        """Union with parameterized Generator listed last - CONSUMED."""
        gen = make_generator()
        Model = create_model(
            "M", field=(Union[float, List[float], Generator[float, None, None]], ...)
        )
        result = Model.model_validate({"field": gen})
        # This FAILS - generator is consumed because List comes first
        assert_consumed(gen)

    def test_union_generator_parameterized_first(self):
        """Union with parameterized Generator listed first - NOT consumed."""
        gen = make_generator()
        Model = create_model(
            "M", field=(Union[Generator[float, None, None], List[float], float], ...)
        )
        result = Model.model_validate({"field": gen})
        assert_not_consumed(result.field)

    def test_union_iterable_parameterized_last(self):
        """Union with parameterized Iterable listed last - CONSUMED."""
        gen = make_generator()
        Model = create_model(
            "M", field=(Union[float, List[float], Iterable[float]], ...)
        )
        result = Model.model_validate({"field": gen})
        # This FAILS - generator is consumed because List comes first
        assert_consumed(gen)

    def test_union_iterable_parameterized_first(self):
        """Union with parameterized Iterable listed first - NOT consumed."""
        gen = make_generator()
        Model = create_model(
            "M", field=(Union[Iterable[float], List[float], float], ...)
        )
        result = Model.model_validate({"field": gen})
        assert_not_consumed(result.field)


class TestUnionOrderMatters:
    """Demonstrate that Union order is the key factor."""

    def test_generator_before_list_ok(self):
        """Generator before List[float] - NOT consumed."""
        gen = make_generator()
        Model = create_model("M", field=(Union[float, Generator, List[float]], ...))
        result = Model.model_validate({"field": gen})
        assert_not_consumed(result.field)

    def test_generator_after_list_consumed(self):
        """Generator after List[float] - CONSUMED."""
        gen = make_generator()
        Model = create_model("M", field=(Union[float, List[float], Generator], ...))
        result = Model.model_validate({"field": gen})
        assert_consumed(gen)

    def test_iterable_before_list_ok(self):
        """Iterable before List[float] - NOT consumed."""
        gen = make_generator()
        Model = create_model("M", field=(Union[float, Iterable, List[float]], ...))
        result = Model.model_validate({"field": gen})
        assert_not_consumed(result.field)

    def test_iterable_after_list_consumed(self):
        """Iterable after List[float] - CONSUMED."""
        gen = make_generator()
        Model = create_model("M", field=(Union[float, List[float], Iterable], ...))
        result = Model.model_validate({"field": gen})
        assert_consumed(gen)


class TestGeneratorSafeWrapper:
    """Test the _make_generator_safe wrapper prevents consumption."""

    def test_generator_safe_wrapper_prevents_consumption(self):
        """Generator-safe wrapper should prevent consumption even with bad Union order."""
        from confection._registry import _make_generator_safe

        # This annotation would normally consume generators (List comes before Generator)
        bad_order = Union[float, List[float], Generator]
        safe_annotation = _make_generator_safe(bad_order)

        gen = make_generator()
        Model = create_model("M", field=(safe_annotation, ...))
        result = Model.model_validate({"field": gen})

        # With the wrapper, generator should NOT be consumed
        assert_not_consumed(result.field)

    def test_generator_safe_wrapper_allows_other_types(self):
        """Generator-safe wrapper should still allow validation of other types."""
        from confection._registry import _make_generator_safe

        annotation = Union[float, List[float], Generator]
        safe_annotation = _make_generator_safe(annotation)

        Model = create_model("M", field=(safe_annotation, ...))

        # Float should work
        result = Model.model_validate({"field": 3.14})
        assert result.field == 3.14

        # List should work
        result = Model.model_validate({"field": [1.0, 2.0, 3.0]})
        assert result.field == [1.0, 2.0, 3.0]

    def test_make_func_schema_with_generator_union(self):
        """make_func_schema should produce a schema that doesn't consume generators."""
        from confection._registry import make_func_schema

        def func_with_generator(schedule: Union[float, List[float], Generator]) -> None:
            pass

        schema = make_func_schema(func_with_generator)

        gen = make_generator()
        result = schema.model_validate({"schedule": gen})

        # Generator should NOT be consumed
        assert_not_consumed(result.schedule)
