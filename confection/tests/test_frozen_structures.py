from typing import Dict, List

import catalogue
import pytest
from pytest import raises

from confection import Config, SimpleFrozenDict, SimpleFrozenList, registry


def test_frozen_list():
    frozen = SimpleFrozenList(range(10))

    for k in range(10):
        assert frozen[k] == k

    with raises(NotImplementedError, match="frozen list"):
        frozen.append(5)

    with raises(NotImplementedError, match="frozen list"):
        frozen.reverse()

    with raises(NotImplementedError, match="frozen list"):
        frozen.pop(0)


def test_frozen_dict():
    frozen = SimpleFrozenDict({k: k for k in range(10)})

    for k in range(10):
        assert frozen[k] == k

    with raises(NotImplementedError, match="frozen dictionary"):
        frozen[0] = 1

    with raises(NotImplementedError, match="frozen dictionary"):
        frozen[10] = 1


@pytest.mark.parametrize("frozen_type", ("dict", "list"))
def test_frozen_struct_deepcopy(frozen_type):
    """Test whether setting default values for a FrozenDict/FrozenList works within a config, which utilizes
    deepcopy."""
    registry.bar = catalogue.create("confection", "bar", entry_points=False)

    @registry.bar.register("foo_dict.v1")
    def make_dict(values: Dict[str, int] = SimpleFrozenDict(x=3)):
        return values

    @registry.bar.register("foo_list.v1")
    def make_list(values: List[int] = SimpleFrozenList([1, 2, 3])):
        return values

    cfg = Config()
    resolved = registry.resolve(
        cfg.from_str(
            f"""
            [something]
            @bar = "foo_{frozen_type}.v1"        
            """
        )
    )

    assert isinstance(resolved["something"], Dict if frozen_type == "dict" else List)
