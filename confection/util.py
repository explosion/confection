import functools
from copy import deepcopy
from typing import Any, Callable, Iterator, Protocol, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

_DIn = TypeVar("_DIn")


class Decorator(Protocol):
    """Protocol to mark a function as returning its child with identical signature."""

    def __call__(self, name: str) -> Callable[[_DIn], _DIn]: ...


# This is how functools.partials seems to do it, too, to retain the return type
PartialT = TypeVar("PartialT")


def partial(
    func: Callable[..., PartialT], *args: Any, **kwargs: Any
) -> Callable[..., PartialT]:
    """Wrapper around functools.partial that retains docstrings and can include
    other workarounds if needed.
    """
    partial_func = functools.partial(func, *args, **kwargs)
    partial_func.__doc__ = func.__doc__
    return partial_func


class Generator(Iterator):
    """Custom generator type. Used to annotate function arguments that accept
    generators so they can be validated by pydantic (which doesn't support
    iterators/iterables otherwise).
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_plain_validator_function(cls.__validate__)

    @classmethod
    def __validate__(cls, v, info):
        if not hasattr(v, "__iter__") and not hasattr(v, "__next__"):
            raise TypeError("not a valid iterator")
        else:
            return v


DEFAULT_FROZEN_DICT_ERROR = (
    "Can't write to frozen dictionary. This is likely an internal "
    "error. Are you writing to a default function argument?"
)

DEFAULT_FROZEN_LIST_ERROR = (
    "Can't write to frozen list. Maybe you're trying to modify a computed "
    "property or default function argument?"
)


class SimpleFrozenDict(dict):
    """Simplified implementation of a frozen dict, mainly used as default
    function or method argument (for arguments that should default to empty
    dictionary). Will raise an error if the user attempts to add to dict.
    """

    def __init__(
        self,
        *args,
        error: str = DEFAULT_FROZEN_DICT_ERROR,
        **kwargs,
    ) -> None:
        """Initialize the frozen dict. Can be initialized with pre-defined
        values.

        error (str): The error message when user tries to assign to dict.
        """
        super().__init__(*args, **kwargs)
        self.error = error

    def __setitem__(self, key, value):
        raise NotImplementedError(self.error)

    def pop(self, key, default=None):
        raise NotImplementedError(self.error)

    def update(self, other):
        raise NotImplementedError(self.error)

    def __deepcopy__(self, memo):
        return self.__class__(deepcopy({k: v for k, v in self.items()}))


class SimpleFrozenList(list):
    """Wrapper class around a list that lets us raise custom errors if certain
    attributes/methods are accessed. Mostly used for properties that return an
    immutable list (and that we don't want to convert to a tuple to not break
    too much backwards compatibility). If a user accidentally calls
    frozen_list.append(), we can raise a more helpful error.
    """

    def __init__(
        self,
        *args,
        error: str = DEFAULT_FROZEN_LIST_ERROR,
    ) -> None:
        """Initialize the frozen list.

        error (str): The error message when user tries to mutate the list.
        """
        self.error = error
        super().__init__(*args)

    def append(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def clear(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def extend(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def insert(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def pop(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def remove(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def reverse(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def sort(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def __deepcopy__(self, memo):
        return self.__class__(deepcopy(v) for v in self)


def is_promise(obj) -> bool:
    if not hasattr(obj, "keys"):
        return False
    id_keys = [k for k in obj.keys() if k.startswith("@")]
    if len(id_keys):
        return True
    return False
