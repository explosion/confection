from pytest import raises

from confection import SimpleFrozenDict, SimpleFrozenList, registry, Config
import dataclasses
from typing import Dict
import catalogue


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


def test_frozen_dict_deepcopy():
    """Test whether setting default values for a FrozenDict works within a config, which utilizes deepcopy."""
    registry.bar = catalogue.create("confection", "bar", entry_points=False)

    @dataclasses.dataclass
    class Foo:
        a: int

    # Load the config file from disk, resolve it and fetch the instantiated optimizer object.
    @registry.bar.register("foo.v1")
    def make_smth(values: Dict[str, int] = SimpleFrozenDict(x=3)):
        return Foo(a=3)

    cfg = Config()
    resolved = registry.resolve(
        cfg.from_str(
            """
            [something]
            @bar = "foo.v1"        
            """
        )
    )

    assert isinstance(resolved["something"], Foo)
    assert resolved["something"].a == 3
