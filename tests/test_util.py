"""Test util.py: frozen collections, partial, try_dump_json."""
import copy
import pytest

from confection.util import (
    SimpleFrozenDict,
    SimpleFrozenList,
    partial,
    try_dump_json,
)
from confection._errors import ConfigValidationError


# --- SimpleFrozenDict ---


def test_frozen_dict_init():
    d = SimpleFrozenDict({"a": 1, "b": 2})
    assert d["a"] == 1


def test_frozen_dict_setitem():
    d = SimpleFrozenDict()
    with pytest.raises(NotImplementedError):
        d["x"] = 1


def test_frozen_dict_pop():
    d = SimpleFrozenDict({"a": 1})
    with pytest.raises(NotImplementedError):
        d.pop("a")


def test_frozen_dict_update():
    d = SimpleFrozenDict()
    with pytest.raises(NotImplementedError):
        d.update({"x": 1})


def test_frozen_dict_deepcopy():
    d = SimpleFrozenDict({"a": [1, 2]})
    d2 = copy.deepcopy(d)
    assert isinstance(d2, SimpleFrozenDict)
    assert d2["a"] == [1, 2]
    assert d2["a"] is not d["a"]


def test_frozen_dict_custom_error():
    d = SimpleFrozenDict(error="custom error")
    with pytest.raises(NotImplementedError, match="custom error"):
        d["x"] = 1


# --- SimpleFrozenList ---


def test_frozen_list_init():
    lst = SimpleFrozenList([1, 2, 3])
    assert lst[0] == 1


def test_frozen_list_append():
    lst = SimpleFrozenList()
    with pytest.raises(NotImplementedError):
        lst.append(1)


def test_frozen_list_clear():
    lst = SimpleFrozenList([1])
    with pytest.raises(NotImplementedError):
        lst.clear()


def test_frozen_list_extend():
    lst = SimpleFrozenList()
    with pytest.raises(NotImplementedError):
        lst.extend([1])


def test_frozen_list_insert():
    lst = SimpleFrozenList()
    with pytest.raises(NotImplementedError):
        lst.insert(0, 1)


def test_frozen_list_pop():
    lst = SimpleFrozenList([1])
    with pytest.raises(NotImplementedError):
        lst.pop()


def test_frozen_list_remove():
    lst = SimpleFrozenList([1])
    with pytest.raises(NotImplementedError):
        lst.remove(1)


def test_frozen_list_reverse():
    lst = SimpleFrozenList([1, 2])
    with pytest.raises(NotImplementedError):
        lst.reverse()


def test_frozen_list_sort():
    lst = SimpleFrozenList([2, 1])
    with pytest.raises(NotImplementedError):
        lst.sort()


def test_frozen_list_deepcopy():
    lst = SimpleFrozenList([[1, 2], [3]])
    lst2 = copy.deepcopy(lst)
    assert isinstance(lst2, SimpleFrozenList)
    assert lst2 == [[1, 2], [3]]
    assert lst2[0] is not lst[0]


def test_frozen_list_custom_error():
    lst = SimpleFrozenList(error="nope")
    with pytest.raises(NotImplementedError, match="nope"):
        lst.append(1)


# --- partial ---


def test_partial_basic():
    def add(a, b):
        """Add two numbers."""
        return a + b

    p = partial(add, 1)
    assert p(2) == 3
    assert p.__doc__ == "Add two numbers."


# --- try_dump_json error ---


def test_try_dump_json_unserializable():
    with pytest.raises(ConfigValidationError, match="Couldn't serialize"):
        try_dump_json(object())
